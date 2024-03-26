#!/usr/bin/env python
import math


import numpy as np
import pyopencl as cl
import pyopencl.array as array

import time


from matpy import *

#Doc
#https://documen.tician.de/pyopencl/
#https://documen.tician.de/pyopencl/array.html

#Get list of available opencl devices
def get_device_list():
    platforms = cl.get_platforms()
    devices = []
    for pf in platforms:
        devices.extend(pf.get_devices())

    #return devices
        
    devices_gpu = [dev for dev in devices if dev.type & cl.device_type.GPU != 0]
    devices = [dev for dev in devices if dev.type & cl.device_type.GPU == 0]
    devices_acc = [dev for dev in devices if dev.type & cl.device_type.ACCELERATOR != 0]
    devices = [dev for dev in devices if dev.type & cl.device_type.ACCELERATOR == 0]
    devices_cpu = [dev for dev in devices if dev.type & cl.device_type.CPU != 0]
    devices = [dev for dev in devices if dev.type & cl.device_type.CPU == 0]
    return devices_gpu+devices_acc+devices_cpu+devices
    
#Class to store OpenCL programs    
class Program(object):
    def __init__(self, code):
    
        self.code = code

    def build(self,ctx):
        self._cl_prg = cl.Program(ctx, self.code)
        self._cl_prg.build()
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
            self.__dict__[kernel.function_name] = kernel
        
#Class to store the OpenCL context
class ContextStore(object):
    def __init__(self):
        self.contexts = {}
        return

    def __getitem__(self, dev):
        if not dev in self.contexts:
            self.contexts[dev] = cl.Context(devices=[dev])
        return self.contexts[dev]    


# Gradient with periodic boundary extension
def grad_per(img):

    grad = np.zeros(img.shape + (2,))
    # Dx
    grad[:-1,:,0] = img[1:,:] - img[:-1,:]
    grad[-1,:,0] = img[0,:] - img[-1,:]
    
    # Dy
    grad[:,:-1,1] = img[:,1:] - img[:,:-1]
    grad[:,-1,1] = img[:,0] - img[:,-1]
    
    return grad

#Define OpenCL kernel code (in "C" language)
prgs = Program("""
#define INIT_INDICES \\
  size_t Nx = get_global_size(0); \\
  size_t Ny = get_global_size(1); \\
  size_t x = get_global_id(0); \\
  size_t y = get_global_id(1); \\
  size_t i = y*Nx + x;


__kernel void update_p(__global float2 *p, __global float *u,const float sig) {
  INIT_INDICES

  float2 p0;
  float pabs = 0.0f;

    // gradient
    float2 val = -u[i];
    if (x < Nx-1) val.s0 += u[i+1];  else val.s0 += u[y*Nx];
    if (y < Ny-1) val.s1 += u[i+Nx]; else val.s1 += u[x];

    // step
    val = p[i] + sig*(val);
    
    // Linfty-prox
    p[i] = val / fmax(fabs(val),(float2)(1.0f, 1.0f));
    
    
   
    // pabs = dot(val, val);
  
    // reproject
    // if (pabs > 1.0f) {
    //    float fac = rsqrt(pabs);
    //    p[i] = val*fac;
    //} else {
    //    p[i] = val;
    //}
    
    
}


// fd = (fd + np.sqrt(N)*np.sqrt(M)*par.ld*mask*d0)/(1.0+par.ld*mask)

__kernel void update_d_l2(__global float2 *fd, __global float2 *d0, __global int *mask, const float ldtau, const float rNM) {

  INIT_INDICES

  fd[i].x = (fd[i].x + rNM*ldtau*mask[i]*d0[i].x)/(1.0f+ldtau*mask[i]);
  
  fd[i].y = (fd[i].y + rNM*ldtau*mask[i]*d0[i].y)/(1.0f+ldtau*mask[i]);
}

__kernel void update_d_exact(__global float2 *fd, __global float2 *d0, __global int *mask, const float rNM) {

  INIT_INDICES


  if (mask[i]==1) {
  
    fd[i] = rNM*d0[i];
    
  }
}


__kernel void update_extra(__global float *u, __global float *ux) {
  INIT_INDICES

    u[i] = 2.0f*ux[i] - u[i];
    
}


__kernel void testcopy(__global float2 *arout, __global float2 *arin,const float ldtau) {
  INIT_INDICES

    // arout[i] = arin[i];
    arout[i].x = ldtau*arin[i].x;
    arout[i].y = ldtau*arin[i].y;
    
}


__kernel void update_u(__global float *u, __global float *ux,
                             __global float2 *p, const float tau) {
  INIT_INDICES

  float2 val;
  
    // divergence
    val = p[i];
    if (x == 0) val.s0 -= p[y*Nx + Nx-1].s0;
    if (x > 0) val.s0  -= p[i-1].s0;
    if (y == 0) val.s1 -= p[(Ny-1)*Nx + x].s1;
    if (y > 0) val.s1  -= p[i-Nx].s1;

    // linear step
    ux[i] = u[i] + tau*(val.s0 + val.s1);

}



"""
)

#Store OpenCL information
cl_contexts = ContextStore()
cl_devices = get_device_list()



def tv_recon(**par_in):


    #Initialize parameters and data input
    par = parameter({})
    data_in = data_input({})
    data_in.mask = 0 #Inpaint requires a mask

    ##Set data
    data_in.u0 = 0 #Direct image input
    # optional direct data input
    data_in.d0_input = False    

    #Version information:
    par.version='Version 1' # Removed initialization of image with u0 (ground truth), initialized with rec0 instead
    

    par.imname = ''

    par.niter = 10

    par.s_t_ratio = 1.0
    
    par.ld = 10
    
    par.check=100 #The higher check is, the faster

    par.show_every = False

    #Select which OpenCL device to use
    par.cl_device_nr = 0

    # Set data fidelity mode
    par.dmode = 'exact' # 'exact' or 'l2'
        
    #Maximal magnitute of fourier freq, i,e., we sample frequences f with |f|<nf, in total 2(nf-1)+1 frequencies
    par.nf = 2
    

    
    par.verbose=False
    
    par.variance = 0 # variance = 0.5*|f^dagger - f^0|^2 when noisy data f^0 is used

    ##Data and parameter parsing
    
    #Set parameters according to par_in
    par_parse(par_in,[par,data_in])

    #Read image
    
    if par.imname and isinstance(data_in.u0,np.ndarray):
        raise ValueError('Both image name and u0 given as input')
    if par.imname and not isinstance(data_in.u0,np.ndarray):
        data_in.u0 = imread(par.imname)
    if not par.imname and not isinstance(data_in.u0,np.ndarray) and not isinstance(data_in.d0_input,np.ndarray):
        raise ValueError('No input given, neigher imname nor u0')


    #Stepsize
    Ln = np.sqrt(8.0)
    par.sig = 0.5*par.s_t_ratio/Ln
    par.tau = 0.5*1.0/(par.s_t_ratio*Ln)
    
    if par.verbose:
        print('Stepsizes: ' + str(par.sig)+',' + str(par.tau))

        print('Using the following data term: '  + par.dmode)

    

    #d0 = np.sqrt(N*M)*np.fft.fft2(u0)
    if np.any(data_in.d0_input):
        d0 = np.copy(data_in.d0_input)
        
        u0 = np.zeros(d0.shape)

    else:
        ## Data initilaization
        u0 = np.copy(data_in.u0)

        d0 = np.fft.fft2(u0)/np.sqrt(u0.shape[0]*u0.shape[1]) #We use the orthogonal dft
    
    # Store noise-free data (will be weighted by mask below)
    d0_noise_free = np.copy(d0)
    #Set variables
    N,M = u0.shape

    
    if par.variance:
        if par.dmode=='exact':
            print('Warning: Using noisy data with exact data term')
            
        # Note: we sample (2(nf-1)+1)^2 frequencies, due to symmetry, we don't use a factor 2 for the imaginary part
        # this should be consistent in expectation since the mean of two gaussans with same variance is gaussian with half variance
        ncoeff = float((2*(par.nf-1) + 1)*(2*(par.nf-1) + 1))
        
        d0.real += np.random.normal(loc=0.0,scale=np.sqrt(2*par.variance/ncoeff),size=d0.shape)
        d0.imag += np.random.normal(loc=0.0,scale=np.sqrt(2*par.variance/ncoeff),size=d0.shape)

    
    mask = np.zeros(d0.shape)
    
    for i in range(-par.nf + 1,par.nf):
        for j in range(-par.nf + 1,par.nf):
        
            mask[i,j] = 1.0
            
    d0 *= mask
    
    # store noise-free coefficients
    d0_noise_free *= mask
    
    d0_true = d0
    
    tmp = np.zeros(d0.shape,dtype=complex)
    #Symmetrize data
    for i in range(0,N):
        for j in range(0,M):
            tmp[i,j] = 0.5*(d0[i,j] + d0[-i,-j].conj()) # variance

    d0 = tmp
    
    # Zero-fill recon
    rec0 = (np.fft.ifft2(d0)*np.sqrt(N*M)).real #We use the orthogonal dft


    ## Initialize OpenCL
    if par.verbose:
        print('Available OpenCL devices:')
        for i in range(len(cl_devices)):
            print('[' + str(i) + ']: ' + cl_devices[i].name)
        print('Choosing: ' + cl_devices[par.cl_device_nr].name)    
    cl_device = cl_devices[par.cl_device_nr]
    #Create context and queue
    ctx = cl_contexts[cl_device]
    queue = cl.CommandQueue(ctx)
    #Build programs   
    prgs.build(ctx)

    ############################################################
    ############################################################

    import reikna.cluda as cluda
    from reikna.core import Annotation, Type, Transformation, Parameter
    from reikna.fft import FFT
    from reikna.cluda import dtypes
    #https://groups.google.com/g/reikna/c/q_4afKsG880
    
    
    api = cluda.ocl_api()
    thr = api.Thread(queue)
    rthr = api.Thread(queue)
    
    # A transformation that transforms a real array to a complex one
    # by adding a zero imaginary part
    def get_complex_trf(arr):
        complex_dtype = dtypes.complex_for(arr.dtype)
        return Transformation(
            [Parameter('output', Annotation(Type(complex_dtype, arr.shape), 'o')),
            Parameter('input', Annotation(arr, 'i'))],
            """
            ${output.store_same}(
                COMPLEX_CTR(${output.ctype})(
                    ${input.load_same},
                    0));
            """)
    
    # A transformation that transforms a complex array to a real one
    def get_real_trf(arr):
        real_dtype = dtypes.real_for(arr.dtype)
        return Transformation(
            [Parameter('output', Annotation(Type(real_dtype, arr.shape), 'o')),
            Parameter('input', Annotation(arr, 'i'))],
             """
            ${output.store_same}(${input.load_same}.x);
            """)

    
    arr =  rec0.astype(np.float32, order='C')
    fd_arr =  rec0.astype(np.complex64, order='C')



    # Create the FFT computation and attach the transformation above to its input.
    trf = get_complex_trf(arr)
    fft = FFT(trf.output) # (A shortcut: using the array type saved in the transformation)
    fft.parameter.input.connect(trf, trf.output, new_input=trf.input)
    cfft = fft.compile(thr)

    # Create FFT with real output
    rtrf = get_real_trf(fd_arr)
    ifft = FFT(fd_arr)
    ifft.parameter.output.connect(rtrf, rtrf.input, new_output=rtrf.output)
    cifft = ifft.compile(rthr)


    if 0: # Test real-output fft
        arr_dev = array.to_device(queue, fd_arr.astype(np.complex64, order='C'))
        
        res_dev = thr.array(arr.shape, np.float32)
        cifft(res_dev, arr_dev,1) # input 0 for forward, 1 for inverse 
        result = res_dev.get()

        reference = np.fft.ifftn(fd_arr).real
        

        #assert np.linalg.norm(result - reference) / np.linalg.norm(reference) < 1e-6
        print('Error:')
        print(np.linalg.norm(result - reference))



    if 0: # Test real input fft
        # Run the computation
        #arr_dev = thr.to_device(arr)
        arr_dev = array.to_device(queue, rec0.astype(np.float32, order='C'))
        
        res_dev = thr.array(arr.shape, np.complex64)
        cfft(res_dev, arr_dev,0) # input 0 for forward, 1 for inverse 
        result = res_dev.get()

        reference = np.fft.fftn(arr)
        

        #assert np.linalg.norm(result - reference) / np.linalg.norm(reference) < 1e-6
        print('Error:')
        print(np.linalg.norm(result - reference))
    
    ############################################################
    ############################################################


    #Initialize OpenCL variables on device
    u0d =  array.to_device(queue, rec0.astype(np.float32, order='C'))

    #maskd = array.to_device(queue, mask_inpainting.astype(np.int32, order='F'))

    
    ud =  array.to_device(queue, rec0.astype(np.float32, order='C'))
    
    maskd =  array.to_device(queue, mask.astype(np.intc, order='C')) # thr.to_device(mask) 
    #d0d =  array.to_device(queue, d0.astype(np.complex64, order='F'))
    d0d =  array.to_device(queue, d0.astype(np.complex64, order='C')) # thr.to_device(d0
    
    #uxd = array.to_device(queue, u0.astype(np.float32, order='F'))
    # for some reason , we need to use thr to generate this array such that it works with reikna
    # it is still open if it does not make a difference to use thr or rthr
    uxd = thr.array(ud.shape, np.float32) 

    #For float2 variables on device, vector components need to be stored in first dimension
    pd = array.zeros(queue,(2,N,M),dtype=np.float32, order='C') 


    #Host variable to transfer image data from device
    u = np.zeros([N,M])
    ux_tmp = np.zeros([N,M])

    #Host functions from matpy only to evaluate objective functional
    grad = gradient(u0.shape)

    ob_val = np.zeros([ int((par.niter-1)/par.check)+1  ,2])
    
    ob_val[0,0] = l1nrm(grad.fwd(u),vdims=2)
    ob_val[0,1] = 0.5*par.ld*np.square(np.abs(np.fft.fftn(u,norm='ortho')*mask-mask*d0)).sum()


    #Host container for fourier data
    fd = np.zeros(u.shape,dtype=complex)
    #Device container for fourier data
    # for some reason , we need to use thr to generate this array such that it works with reikna
    # it is still open if it does not make a difference to use thr or rthr
    fd_dev = thr.array(arr.shape, np.complex64) #array.to_device(queue, u0.astype(np.complex64, order='F'))

    #Define wrapper for OpenCL programs
    #Dual step
    def update_p(p,u,sig):
        return prgs.update_p(p.queue, u.shape, None , p.data , u.data , np.float32(sig))

    # Primal step
    def update_u(u, ux, p, tau):
        return prgs.update_u(ux.queue, u.shape, None, u.data, ux.data,
                                      p.data, np.float32(tau))

    #Extragradient step
    def update_extra(u, ux):
        return prgs.update_extra(u.queue, u.shape, None, u.data, ux.data)
        
    #Data prox
    def update_d_l2(fd, d0, mask,ldtau,rNM):
        return prgs.update_d_l2(fd.queue, fd.shape, None, fd.data, d0.data,
                                      mask.data, np.float32(ldtau), np.float32(rNM))
    def update_d_exact(fd, d0, mask,rNM):
        return prgs.update_d_exact(fd.queue, fd.shape, None, fd.data, d0.data,
                                      mask.data, np.float32(rNM))

    def testcopy(arout, arin, ldtau):
        return prgs.testcopy(arout.queue, arout.shape, None, arout.data, arin.data,np.float32(ldtau))



    start_time = time.time()

    for k in range(par.niter):
        
        #Dual
        #p = p + par.sig*( grad.fwd(ux) )
        #p = proxl1s(p,1.0,vdims=2)

        update_p(pd,uxd,par.sig)


        #Primal
        #ux = prox( u - par.tau*(grad.adj(p))   ,ppar=par.tau)    
        #update_u_inpaint(ud, uxd, pd, u0d, maskd, par.tau)
        
        update_u(ud, uxd, pd, par.tau)

        if 0: # 29.96 seconds for 10000 iterations
            ux_tmp = uxd.get()
            fd = np.fft.fftn(ux_tmp)
            
            if par.dmode == 'l2':
                fd = (fd + np.sqrt(N)*np.sqrt(M)*par.ld*par.tau*mask*d0)/(1.0+par.ld*par.tau*mask)
            elif par.dmode == 'exact':
                fd[mask==1] = np.sqrt(N)*np.sqrt(M)*d0[mask==1]
            else:
                raise ValueError('Unknown data fidelity mode.')
            
            
            ux_tmp = np.fft.ifftn(fd).real
            uxd = array.to_device(queue, ux_tmp.astype(np.float32, order='C'))


        if 1: # 6.23 seconds for 10000 iterations
            cfft(fd_dev, uxd,0)
            
            if par.dmode == 'l2':
                update_d_l2(fd_dev, d0d, maskd,par.ld*par.tau,np.sqrt(N*M))
            elif par.dmode == 'exact':
                update_d_exact(fd_dev, d0d, maskd,np.sqrt(N*M))
            else:
                raise ValueError('Unknown data fidelity mode.')
            cifft(uxd, fd_dev,1)
        

        #Extragradient
        #u = 2.0*ux - u
        #[u,ux] = [ux,u]
        update_extra(ud, uxd)
        (ud, uxd) = (uxd, ud)
        

        if k>0 and (np.remainder(k,par.check) == 0):
            #Getting the objective value is costly
            u = ud.get() #Copy current image from device
            ob_val[int((k)/par.check),0] = l1nrm(grad_per(u),vdims=())
            ob_val[int((k)/par.check),1] = 0.5*par.ld*np.square(np.abs(np.fft.fftn(u,norm='ortho')*mask-mask*d0)).sum()


            print('Iteration: ' + str(k) + ' / Ob-val: ' + str(ob_val[int((k+1)/par.check)].sum()))

        if par.show_every:
            if np.remainder(k+1,par.show_every) == 0:
                closefig()
                u = ud.get()
                imshow(u,title='Iter: '+str(k+1))
                plt.pause(1)

    if par.verbose:
        print("--- %s seconds ---" % (time.time() - start_time))

    #Initialize output class
    res = output(par)

    #Set original input
    res.orig = data_in.u0
    res.u0 = u0
    res.rec0 = rec0
    res.d0 = d0
    res.d0_true = d0_true
    res.d0_noise_free = d0_noise_free

    res.d0d = d0d.get()
        
    res.u = ud.get() #ud.get() copies device variables to host
    res.p = np.moveaxis(pd.get(),0,-1) #Changes axis alignment 
    res.ob_val = ob_val    
    
    #Save parameters and input data
    res.par = par
    res.par_in = par_in
    res.data_in = data_in
    
    return res




