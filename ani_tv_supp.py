import numpy as np

import random
import itertools
import matpy as mp
import matplotlib.pyplot as plt

import tv_recon_ocl as tvcl



# Randomly select positions of jump points with minimal distances in given delta bins
def get_jump_points_bin(delta_bin,npoints = 1,maxtries=5e05,maxjumps=4,imgsz=120,seed=None):
    

    np.random.seed(seed)
    i = 0

    pointlist = []    

    
    while i<maxtries*npoints and len(pointlist)<npoints:
    
        M = np.random.randint(2,maxjumps+1)

        # select random points. note: equal points will be rejected below
        points_x = np.sort(np.random.randint(0,imgsz,size=(M)))
    
        difs = np.concatenate( [points_x[1:] - points_x[0:-1], np.array([(imgsz - points_x[-1]) + points_x[0] ])],axis=0)
        

        if delta_bin[0] <= difs.min()/float(imgsz) < delta_bin[1]:
        
            N = np.random.randint(2,maxjumps+1)
            points_y = np.sort(np.random.randint(0,imgsz,size=(N)))
        
            difs = np.concatenate( [points_y[1:] - points_y[0:-1], np.array([imgsz - points_y[-1] + points_y[0] ])],axis=0)
            
            if delta_bin[0] <= difs.min()/float(imgsz) < delta_bin[1]:
                
                pointlist.append([points_x.tolist(),points_y.tolist()])
        
        i += 1
        
    if len(pointlist)<npoints:
        raise Warning("Found only " + str(len(pointlist)) + " of " + str(npoints) +" points for delta-bin: " + str(delta_bin))
    

    return pointlist

# Input: array of jump point distances, must be equally spaced and at least two deltas
# Domain size is regarded as 1
def get_jump_points(deltas,npoints =1,maxtries=5e05,maxjumps = 4,imgsz=120,seed=None):


    w = (deltas[1] - deltas[0])/2
    
    
    delta_bins = [ [delta-w,delta + w] for delta in deltas]
    
    
    if  delta_bins[0][1] < 1.0/float(imgsz):
        print('Delta bin: ' + str(delta_bins[0]))
        raise Warning("Delta-bin impossible to match: Pixel distance < " + str(imgsz*delta_bins[0][0]))
    if  delta_bins[-1][0] >= 0.33:
        print('Delta bin: ' + str(delta_bins[-1]))
        raise Warning("Delta-bin impossible to match: Pixel distance > " + str(delta_bins[-1][-1]*imgsz))
    for delta_bin in delta_bins:
        if np.ceil(delta_bin[0]*imgsz) == np.ceil(delta_bin[1]*imgsz):
            print('Delta bin: ' + str(delta_bin))
            raise Warning("Delta-bin impossible to match: Interval: " + str(delta_bin[0]*imgsz) + ' / ' +  str(delta_bin[1]*imgsz))
    
    data = {}
    
    # Get lists of jump points
    for i in range(len(delta_bins)):
        
        data[deltas[i]] = get_jump_points_bin(delta_bins[i],npoints = npoints,maxtries=maxtries,maxjumps=maxjumps,imgsz=imgsz,seed=seed)
        print('Finished bin: '  + str(delta_bins[i]))
    
    return data


# Get valid pixel values of size MxN
# Optional input: Image to be corrected, in this case, (M,N) = img.shape
def get_valid_values(M,N,seed=None,img=False):

    np.random.seed(seed)
    
    grad_passed = False
    counter = 0
    
    eps = 1e-08
    
    while not grad_passed and counter < 10:
        
        if counter>0:
            print('Try. nr: ' + str(counter))
        
        if np.any(img):
            (M,N) = img.shape

        points = list(itertools.product(range(M), range(N)))
        random.shuffle(points)
        
        eps = 1e-09 # small tolerance to avoid exact equality
        
        dims = (M,N) # point dimensions
        
        vals = -np.ones(dims) # container for values, -1 means unset
        grad = [np.zeros(M),np.zeros(N)] # container for gradients
        
        
        
        # Stancil for value comparison (to be shuffled)
        stencil = [[1,0],[-1,0],[0,1],[0,-1]]

        for point in points:
        

            # Define possible value range
            mx = 1.0
            mn = 0.0
            
            random.shuffle(stencil) # randomly select order of stencil points
            for dx in stencil: #loop over stencil points

                idx = ((point[0]+dx[0])%M,(point[1]+dx[1])%N) #index of neighboring pixel

                if vals[idx] != -1: #if value is already set
                    
                    ax = 0 if dx[0] !=0 else 1 #set axis of stencil
                    # compare along axis ax
                    if grad[ax][ (point[ax]+min(dx[ax],0))%dims[ax] ] == dx[ax]: # neighboring pixel must be larger
                        mx_tmp = mx
                        mx = min(mx,vals[idx])
                        # correct if upper bound is too small
                        if mn>mx:
                            i = 0
                            while (grad[ax][ (point[ax]+min(dx[ax],0)+i*dx[ax])%dims[ax] ] == dx[ax]) & (i<dims[ax]):
                                pos = (point[ax]+(i+1)*dx[ax])%dims[ax] #current position
                                if ax==0: #first axis
                                    setvals = vals[pos,:] != -1
                                    vals[pos,setvals] += (mn-mx+eps)
                                    vals[pos,setvals] = np.clip(vals[pos,setvals],0.0,1.0)
                                else: #second axis
                                    setvals = vals[:,pos] != -1
                                    vals[setvals,pos] += (mn-mx+eps)
                                    vals[setvals,pos] = np.clip(vals[setvals,pos],0.0,1.0)
                                i += 1  
                            mx = min(mx_tmp,vals[idx]) #new maximum
                            
                    if grad[ax][ (point[ax]+min(dx[ax],0))%dims[ax] ] == - dx[ax]: # neighboring pixel must be smaller
                        mn_tmp = mn
                        mn = max(mn,vals[idx])
                        #correct if lower bound is too high
                        if mn>mx:
                            i = 0
                            while (grad[ax][ (point[ax]+min(dx[ax],0)+i*dx[ax])%dims[ax] ] == -dx[ax]) & (i<dims[ax]):
                                pos = (point[ax]+(i+1)*dx[ax])%dims[ax] #current position
                                if ax==0: #first axis
                                    setvals = vals[pos,:] != -1
                                    vals[pos,setvals] -= (mn-mx+eps)
                                    vals[pos,setvals] = np.clip(vals[pos,setvals],0.0,1.0)
                                else: #second axis
                                    setvals = vals[:,pos] != -1
                                    vals[setvals,pos] -= (mn-mx+eps)
                                    vals[setvals,pos] = np.clip(vals[setvals,pos],0.0,1.0)
                                i += 1
                            mn = max(mn_tmp,vals[idx]) #new minimum
       
            #Set value
            if not np.any(img):
                vals[point] = np.random.uniform(mn,mx)
                
            else:
                vals[point] = np.clip(img[point],mn,mx)
            
            if mn>mx:
                print('problem with mx/mn: ' + str(mx) + ' / ' + str(mn))
                print('value: ' + str(vals[point]))
            
            
            # Define resulting gradients
            for dx in stencil: #loop over neighboring pixels
                idx = ((point[0]+dx[0])%M,(point[1]+dx[1])%N) #index of neighboring pixel
                
                if vals[idx] != -1: #if value is already set
                    
                    ax = 0 if dx[0] !=0 else 1 #set axis of stencil
                    
                    if not grad[ax][ (point[ax]+min(dx[ax],0))%dims[ax] ]: # if gradient is not yet set
                        grad[ax][(point[ax]+min(dx[ax],0))%dims[ax]] = np.sign(vals[idx] - vals[point])*dx[ax]
    
        grad_passed = test_grad(vals)[0]
        
        grad_mag = np.abs(grad_per(vals)).sum()/(N*M)
        if grad_mag < eps:
            grad_passed = False
            print('Dedected zero gradient - retrying')
        
        
        counter +=1


    if not grad_passed:
            
        grad_passed = test_grad(vals,verbose=True)[0]
        raise Warning("Gradients not valid")
        
    return vals


def color_image(data,points,imgsz=120):

    lx = len(points[0])
    ly = len(points[1])

    u = np.zeros((imgsz,imgsz))

    for idx in range(lx-1) :
        for idy in range(ly-1):
        
            u[points[0][idx]:points[0][idx+1],points[1][idy]:points[1][idy+1]] = data[idx,idy]
        
        u[points[0][idx]:points[0][idx+1],:points[1][0]] = data[idx,-1]
        u[points[0][idx]:points[0][idx+1],points[1][-1]:] = data[idx,-1]

    for idy in range(ly-1):
    
        u[:points[0][0],points[1][idy]:points[1][idy+1]] = data[-1,idy]
        u[points[0][-1]:,points[1][idy]:points[1][idy+1]] = data[-1,idy]
    
    u[:points[0][0],:points[1][0]] = data[-1,-1]
    u[:points[0][0],points[1][-1]:] = data[-1,-1]
    u[points[0][-1]:,:points[1][0]] = data[-1,-1]
    u[points[0][-1]:,points[1][-1]:] = data[-1,-1]

    return u

def get_image(points,imgsz=120,valid=True,seed=None):

    
    lx = len(points[0])
    ly = len(points[1])


    if valid:
        data = get_valid_values(lx,ly,seed=seed)
    else:
        np.random.seed(seed)
        data = np.random.uniform(0,1,size=(lx,ly))

    return color_image(data,points,imgsz=imgsz)
        
def get_image_database(point_data,imgsz=120,valid=True,seed=None):

    
    img_data = {}
    
    # Get images according to jump points
    for key in point_data.keys():
        img_data[key] = []
        for idx,points in enumerate(point_data[key]):
            img_data[key].append(get_image(points,imgsz=120,valid=valid,seed=None))
            print('Finished coloring of image ' + str(idx) + ' of delta ' + str(key))
    
    return img_data

def test_recon(img_data,niter=10000,nf=50):


    
    result = {}
    
    for key in img_data.keys():
        result[key] = []
        for idx,u0 in enumerate(img_data[key]):
        
            tv_res = tvcl.tv_recon(niter=niter,u0=u0,dmode='exact',nf=nf,check=niter)
            result[key].append(tv_res.u)
            print('Finished reconstructing image ' + str(idx) + ' of delta ' + str(key))

    # Export ani-tv parameters
    res = mp.output(tv_res.par)
    
    res.imgs = result
            
    return res


def test_cutoff_image(u0,niter=10000,nfs=np.linspace(6,54,17)):


    imgs = []
    l1errs = []
    rec0s = []
    
    
    for nf in list(nfs):
        
            tv_res = tvcl.tv_recon(u0=u0,niter=niter,dmode='exact',nf=nf,check=niter)
            imgs.append(tv_res.u)
            rec0s.append(tv_res.rec0)
            l1errs.append(np.abs(tv_res.u - tv_res.orig).sum()/np.prod(tv_res.u.shape))
            
            print('Finished reconstructing nf=' + str(nf))

    # Export ani-tv parameters
    res = mp.output(tv_res.par)
    
    res.imgs = imgs
    res.rec0s = rec0s
    res.l1errs = l1errs
            
    return res



def test_cutoff_dataset(img_data,niter=10000,nfs=np.linspace(6,54,17)):


    imgs = {}
    l1errs = {}

    
    for key in img_data.keys():
        imgs[key] = []
        l1errs[key] = []
        

        for u0 in img_data[key]:
            for nf in list(nfs):
                
                tv_res = tvcl.tv_recon(u0=u0,niter=niter,dmode='exact',nf=nf,check=niter)
                imgs[key].append(tv_res.u)
                l1errs[key].append(np.abs(tv_res.u - tv_res.orig).sum()/np.prod(tv_res.u.shape))

                print('Finished reconstructing delta ' + str(key) + ' with nf ' + str(nf))

    # Export ani-tv parameters
    res = mp.output(tv_res.par)
    
    res.imgs = imgs
    res.l1errs = l1errs
            
    return res




def grad_exact(points,img):


    return 0


def grad_per(img):

    grad = np.zeros(img.shape + (2,))
    # Dx
    grad[:-1,:,0] = img[1:,:] - img[:-1,:]
    grad[-1,:,0] = img[0,:] - img[-1,:]
    
    # Dy
    grad[:,:-1,1] = img[:,1:] - img[:,:-1]
    grad[:,-1,1] = img[:,0] - img[:,-1]
    
    return grad
    


def eval_result(res,tol=1e-5,show=False,export=False,title=False):

    #tol =  error tolerance ofr exact evaluations
    
    # Set data
    point_data = res.point_data
    img_data = res.img_data
    imgs = res.imgs
    
    outfigname = res.output_name(folder='images',outpars = ['nf','niter'])
    


    tmp = list(point_data.keys())
    
    l1error = np.zeros((len(tmp),len(point_data[tmp[0]])))

    gradpass = np.zeros(l1error.shape)
    colpass = np.zeros(l1error.shape)
    allpass = np.zeros(l1error.shape)
    
    # Get l1 error and passing information
    for i,key in enumerate(point_data.keys()):
        for j in range(len(point_data[key])):
            
            
            #l1error[i,j] =  np.log(np.abs(img_data[key][j] - imgs[key][j]).sum()/np.prod(imgs[key][j].shape))
            l1error[i,j] =  np.abs(img_data[key][j] - imgs[key][j]).sum()/np.prod(imgs[key][j].shape)
            
            
            
            # Check exactness of gradient            
            grad = grad_per(imgs[key][j])
            grad_true = grad_per(img_data[key][j])
            
            gradpass[i,j] = np.all( np.abs(grad - grad_true) < tol)
            
            # Check exactness of coloring
            
            pointlist = list(itertools.product( point_data[key][j][0],point_data[key][j][1]))
            pointlist = tuple(np.array(pointlist).T) # conversion for correct numpy indexing
            
            colpass[i,j] = np.all( np.abs(imgs[key][j][pointlist] - img_data[key][j][pointlist] ) < tol)
            
            allpass[i,j] = gradpass[i,j] and colpass[i,j]
    
    
    


    deltas = list(point_data.keys())

    # Show l1 error
    mean_error = l1error.sum(axis=1)/l1error.shape[1]
    # Compute non-symmetric averate deviation
    err = np.zeros( (2,) + mean_error.shape )
    for i in range(l1error.shape[0]):
        
        dif = l1error[i,:] - mean_error[i]
    
        idpos = dif>=0
        idneg = dif<0
        
        err[0,i] = np.sqrt( np.square(dif[idneg]).sum()/float(np.count_nonzero(idneg)))
        err[1,i] = np.sqrt( np.square(dif[idpos]).sum()/float(np.count_nonzero(idpos)))
    
    plt.rcParams.update({'font.size': 14})
    fig, ax1 = plt.subplots(figsize=(8,5))


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    width = 0.006
    #ax2.bar([ delta - width for delta in deltas],gradpass.sum(axis=1)/gradpass.shape[1],width=width,label='gradient')
    ax2.bar(deltas,100*allpass.sum(axis=1)/gradpass.shape[1],width=width,label='color',color='tab:blue',alpha=0.2)
    #ax1.bar([ delta + width for delta in deltas],allpass.sum(axis=1)/gradpass.shape[1],width=width,label='all')
    ax2.set_ylim(0,100)
    
    ax2.xaxis.set_ticks(deltas)
    ax2.set_ylabel('Percentage of exact reconstruction')


    ax1.errorbar(deltas, mean_error, yerr=err, fmt='-o',capsize=4,color='tab:red')#marker='o')      
    
    ax1.set_yscale('log')
    ax1.set_ylim(1e-07,1e-01)
    ax1.set_xlabel('Jump-point distance $\Delta$')
    ax1.set_ylabel('L1-error')

    
    if not title:
        ax1.set_title('As1-error and percentage of exact reconstruction')
    else:
        ax1.set_title(title)

    plt.rcParams.update({'font.size': 15})
    if export:
        clean_outfigname = outfigname.replace(', ','_')
        clean_outfigname = clean_outfigname.replace('[','')
        clean_outfigname = clean_outfigname.replace(']','')
        plt.savefig(clean_outfigname + '_fig_l1_perc' + '.pdf',format='pdf')

    # Show correct reconstruction rate
    # Figure size
    plt.figure(figsize=(10,5))
    # Plotting
    width = 0.0025
    plt.bar([ delta - width for delta in deltas],gradpass.sum(axis=1)/gradpass.shape[1],width=width,label='gradient')
    plt.bar(deltas,colpass.sum(axis=1)/gradpass.shape[1],width=width,label='color')
    plt.bar([ delta + width for delta in deltas],allpass.sum(axis=1)/gradpass.shape[1],width=width,label='all')
    plt.ylim(0,1)

    plt.xlabel('Delta')
    plt.title('Percentage of exact reconstruction')
    plt.xlabel('Jump-point distance $\Delta$')
    plt.legend(loc='best')




    error = mp.output({})
    
    error.l1error = l1error
    error.allpass = allpass
    error.colpass = colpass
    error.gradpass =gradpass
    
    return error
            

def eval_results_rate(res_rate,export=False,):

    l1error = res_rate.l1error

    # Average over images
    l1_img_avg = np.zeros((l1error.shape[0],l1error.shape[2]))

    for ndelta in range(l1error.shape[0]):
        for nvar in range(l1error.shape[2]):
            
            # Number of images
            nnz = np.count_nonzero(l1error[ndelta,:,nvar] != -1)
            
            l1_img_avg[ndelta,nvar] = ((l1error[ndelta,:,nvar])[l1error[ndelta,:,nvar]!=-1]).sum()/nnz



    l1avg = l1_img_avg.sum(axis=0)/float(l1_img_avg.shape[0])


    variances = res_rate.par.variances
    plt.rcParams.update({'font.size': 14})    
    fig = plt.figure(figsize=(10,5))
    plt.loglog(variances,l1avg)


    # Shift for loglog-plot
    nn = int(0.5*l1avg.shape[0])
    tmp = np.exp( np.log(np.sqrt(np.sqrt(variances))) - np.log(np.sqrt(np.sqrt(variances[nn]))) + np.log(l1avg[nn]))
    tmp2 = np.exp( np.log(np.sqrt(variances)) - np.log(np.sqrt(variances[nn])) + np.log(l1avg[nn]))

    plt.loglog(variances,tmp,linestyle='--')
    plt.loglog(variances,tmp2,linestyle=':')

    plt.legend(['$||u - u^\dagger ||_1$','$\delta^{1/4}$','$\delta^{1/2}$'],prop={'size': 14})
    plt.xlabel('Variance')
    plt.ylabel('L1-error')

    if export:
        outfigname = res_rate.output_name(folder='images')
        plt.savefig(outfigname + '_fig_rate' + '.pdf',format='pdf')




def eval_result_cutoff(res_real,res_data,export=False):


    
    # Save all images
    for idx,nf in enumerate(list(res_real.nfs)):
        mp.imsave('../images/cameraman/cameraman_nf_' + str(nf) + '.png',res_real.imgs[idx])


    l1error_real = np.array(res_real.l1errs)
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(res_real.nfs,l1error_real)

    legend = ['Cameraman']

    for key in res_data.imgs.keys():
        for idx,nf in enumerate(list(res_data.par.nfs)):
            mp.imsave('../images/synthetic/img_delta_' + str(key) + '_nf_' + str(nf) + '.png',res_data.imgs[key][idx])
        
        plt.plot(res_data.par.nfs,res_data.l1errs[key])
        legend.append('$\Delta$ = ' + str(key))
        


    plt.xlabel('$\Phi$')
    plt.ylabel('L1-error')
    plt.yscale('log')
    plt.title('L1 error for reconstructing different images in dependence on $\Phi$')
    plt.legend(legend)
    if export:
        plt.savefig('../images/cutoff_test_l1_error_plot' + '.pdf',format='pdf')





def visualize_result(res,error,deltas= [],nimg=2,correct=True,ind_dict = {}):


    if ind_dict:
        print('Warning: Indices given, ignoring correct flag')

    if not deltas:
        deltas = list(res.point_data.keys())
        
    for delta in deltas:
    
        j = list(res.point_data.keys()).index(delta)
        
        indices = [ i for i in range(error.gradpass.shape[1]) if error.gradpass[j,i] == correct ]
        
        
        if delta not in ind_dict:
            if len(indices)>=nimg:
                imns = list(np.random.choice(indices,replace=False,size=(nimg)))
            else:
                imns = indices
            ind_dict[delta] = imns
            
        else:        
            imns = ind_dict[delta]
            
        for i in range(len(imns)):
            fig, axes = plt.subplots(1, 2,figsize=(10,20))
            

            #fig.suptitle('Delta: '+  str(delta))
            im = axes[0].imshow(res.imgs[delta][imns[i]])
            plt.colorbar(im,ax = axes[0],fraction=0.046, pad=0.04)
            axes[0].set_title('Delta: '+  str(delta) + ', index: ' + str(imns[i]))
            
            im2 = axes[1].imshow(np.abs( res.imgs[delta][imns[i]] - res.img_data[delta][imns[i]]))
            plt.colorbar(im2,ax = axes[1],fraction=0.046, pad=0.04)
            axes[1].set_title('Error')

    return ind_dict



def compare_result(res1,res2,error1,error2,deltas= [],nimg=2,correct=True,complementary = False,export=False):


    if not deltas:
        deltas = list(res1.point_data.keys())
        
        
    fig, axes = plt.subplots( len(deltas)*nimg, 5,figsize=(20,len(deltas)*4*nimg))            
    for count,delta in enumerate(deltas):
    

        j = list(res1.point_data.keys()).index(delta)
        
        if not complementary:
            indices = [ i for i in range(error1.gradpass.shape[1]) if error1.gradpass[j,i] == correct ]
        else:
            indices = [ i for i in range(error1.gradpass.shape[1]) if (error1.gradpass[j,i] == correct) and (error2.gradpass[j,i] != correct)]
        
        if len(indices)>=nimg:
            imns = list(np.random.choice(indices,replace=False,size=(nimg)))
        else:
            imns = indices
        
        

        for i in range(len(imns)):
            
            fpos = nimg*count + i
            
            im = axes[fpos,0].imshow(res1.imgs[delta][imns[i]],interpolation=None)
            plt.colorbar(im,ax = axes[fpos,0],fraction=0.046, pad=0.02)
            axes[fpos,0].set_title('Res 1, Delta: '+  str(delta) + ', index: ' + str(imns[i]))
            axes[fpos,0].axis('off')
            
            im = axes[fpos,1].imshow(res2.imgs[delta][imns[i]],interpolation=None)
            plt.colorbar(im,ax = axes[fpos,1],fraction=0.046, pad=0.02)
            axes[fpos,1].set_title('Res 2, Delta: '+  str(delta) + ', index: ' + str(imns[i]))
            axes[fpos,1].axis('off')
            
            im = axes[fpos,2].imshow(np.abs(res1.img_data[delta][imns[i]]-res2.img_data[delta][imns[i]]),interpolation=None)
            plt.colorbar(im,ax = axes[fpos,2],fraction=0.046, pad=0.02)
            axes[fpos,2].set_title('Difference')
            axes[fpos,2].axis('off')
            
            im2 = axes[fpos,3].imshow(np.abs( res1.imgs[delta][imns[i]] - res1.img_data[delta][imns[i]]),interpolation=None)
            plt.colorbar(im2,ax = axes[fpos,3],fraction=0.046, pad=0.02)
            axes[fpos,3].set_title('Error res1')
            axes[fpos,3].axis('off')
            
            im2 = axes[fpos,4].imshow(np.abs( res2.imgs[delta][imns[i]] - res2.img_data[delta][imns[i]]),interpolation=None)
            plt.colorbar(im2,ax = axes[fpos,4],fraction=0.046, pad=0.02)
            axes[fpos,4].set_title('Error res2')
            axes[fpos,4].axis('off')
            
    if export:
        outfigname = res1.output_name(folder='results',outpars = ['nf','niter'])
        plt.savefig(outfigname + '_compare_compl_' + str(complementary) + '.pdf',format='pdf')


# Function to change the coloring in order to make the gradient condition invalid
def make_invalid(img_data,point_data,nflips=1,maxtries=int(1e4),seed=False):
    
    eps = 1e-8
    
    np.random.seed(seed)
    
    # Loop over all images
    for delta in point_data.keys():
        for count,points in enumerate(point_data[delta]):
            
            M = len(points[0])
            N = len(points[1])

            # Recover values on regular grid            
            vals = np.zeros((M,N))
            
            for i in range(M):
                for j in range(N):
                    vals[i,j] = img_data[delta][count][points[0][i],points[1][j]]


            # Get gradient                    
            grad = grad_per(vals)
            
            gxnz = np.where((np.abs(grad[...,0])>eps))
            gynz = np.where((np.abs(grad[...,1])>eps))
            
            # Flip values
            invalid = False
            try_nr = 0
            while not invalid and try_nr < maxtries:

                # List of points with non-zero gradient
                gnzpoints = [ [(gxnz[0][i],gxnz[1][i]),((gxnz[0][i]+1)%M,gxnz[1][i])] for i in range(len(gxnz[0]))]
                gnzpoints += [ [(gynz[0][i],gynz[1][i]),(gynz[0][i],(gynz[1][i]+1)%N)] for i in range(len(gynz[0]))]

                np.random.shuffle(gnzpoints)
                
                vals_try = np.copy(vals)
                i = 0
                while (i < nflips) and (len(gnzpoints)>0):
                                
                    point = gnzpoints.pop()               
                    
                    tmp = vals_try[point[0]]
                    vals_try[point[0]] = vals_try[point[1]]
                    vals_try[point[1]] = tmp
                    i+=1
                
                if i<nflips:
                    print('Delta: ' + str(delta) + ', count: ' + str(count))
                    print(' i = ' + str(i) + ', nflips = ' + str(nflips))
                    print(gxnz)
                    print(gynz)
                    raise Warning('Not enough non-zero gradients to flip values')
                
                # Test if we have at least nflips invalid gradient
                invalid = (test_grad(vals_try)[1] >= nflips)
                try_nr += 1
                
            if try_nr == maxtries:
                    print('Delta: ' + str(delta) + ', count: ' + str(count))
                    print(' i = ' + str(i) + ', nflips = ' + str(nflips))
                    raise Warning('Could not find invalid values')
            
            # Update image data            
            img_data[delta][count] = color_image(vals_try,points,imgsz=img_data[delta][count].shape[0])

            
    return img_data            
            
            
    
                                        


            
        
def test_grad(vals,show=False,verbose=False):

    grad = grad_per(vals)
    gradx = grad[...,0]
    grady = grad[...,1]

    gradx = np.sign(gradx)
    grady = np.sign(grady)

    
    passed = True

    #Test gradient y
    gxpos = np.sign(np.maximum(gradx,0).sum(axis=1))
    gxneg = np.sign(np.minimum(gradx,0).sum(axis=1))
    
    gx_test = -5*np.ones(gxpos.shape) #
    
    # non-admissible values
    gx_test[gxpos == 1] = -1
    gx_test[gxneg == -1] = 1

    gx_test = gx_test[:,np.newaxis]

    n_gx_invalid = np.count_nonzero(gradx == gx_test)
    if n_gx_invalid:
        if verbose:
            print('Problem with gx')
            res = mp.output({})
            res.vals = vals
            res.gradx = gradx
            res.gx_test = gx_test
            mp.psave('wrong_grad',res)
        
        passed = False

    #Test gradient y
    gypos = np.sign(np.maximum(grady,0).sum(axis=0))
    gyneg = np.sign(np.minimum(grady,0).sum(axis=0))
    
    gy_test = -5*np.ones(gypos.shape) #
    
    # non-admissible values
    gy_test[gypos == 1] = -1
    gy_test[gyneg == -1] = 1

    gy_test = gy_test[np.newaxis,:]
    
    n_gy_invalid = np.count_nonzero(grady == gy_test)
    if np.any(grady == gy_test):
        if verbose:
            print('Problem with gy')
            if passed:
                res = mp.output({})
                res.vals = vals
                res.grady = grady
                res.gy_test = gy_test
                mp.psave('wrong_grad',res)

        passed = False

    if show:
        mp.imshow(gradx)
        mp.imshow(grady)
        

    if passed and verbose:
        print('Gradient valid.')
    
    n_g_invalid = n_gx_invalid + n_gx_invalid
    return passed,n_g_invalid
                
#########################################################################################################
# Functions for testing the convergence result
#########################################################################################################


def select_working_subset(res,tol=1e-05,max_len=2,store=False):

    par = mp.parameter({})

    par.tol = tol
    par.max_len = max_len
    
    

    # Get error measures
    error = eval_result(res,tol=par.tol,show=False,export=False)
    
    # Extract images    
    imgs = {}
    for i,key in enumerate(res.img_data.keys()):
        for j in range(len(res.img_data[key])):
            
            if key not in imgs:
                imgs[key] = []
            if len(imgs[key]) < par.max_len and error.allpass[i,j] == True:
                imgs[key].append(res.img_data[key][j])

    data = mp.output({})
    
    data.par = par
    data.imgs = imgs
    data.nf = res.par.nf
    
    data.fname = res.fname + '_selected'
    
    if store:
        data.save(outpars=['max_len'],folder='data')
    
    
    return data



def find_opt_constant(imgs,variances = [0.5,1,5,10],nf=18,niter=15,C=np.linspace(0.1,1,10)):

    par = mp.parameter({})

    par.niter = niter
    par.nf = nf

    

    l1error = np.zeros((C.shape[0],len(variances),len(imgs.keys())))

    for cidx in range(C.shape[0]):
        for varnr,var in enumerate(variances):
            for ndelta,delta in enumerate(imgs.keys()):
                # Average over all images with given delta distance
                err = 0
                for img in imgs[delta]:
                    tv_res = tvcl.tv_recon(niter=par.niter,u0=img,dmode='l2',nf=par.nf,ld=1.0/(np.sqrt(var)*C[cidx]),variance=var,check=par.niter)
                    err += np.abs(img - tv_res.u).sum()/np.prod(img.shape)
                err /= float(len(imgs[delta]))
            
                l1error[cidx,varnr,ndelta] = err
        print('Finished constant C= ' + str(C[cidx]))            
    return l1error
    

def get_rate_img(img,variances=[0.5,1,5,10],C = 0.028,niter=10,nf=18):


    l1error = np.zeros(len(variances))
    
    for vcount,var in enumerate(variances):
    
        tv_res = tvcl.tv_recon(niter=niter,u0=img,dmode='l2',nf=nf,ld=1.0/(np.sqrt(var)*C),variance=var,check=niter)
        
        l1error[vcount] = np.abs(img - tv_res.u).sum()/np.prod(img.shape)


    return l1error


def get_rate(imgs,variances=[0.5,1,5,10],C = 0.028,niter=10,nf=18):

    #Get maximal number of available images
    mxlen = 0
    for key in imgs.keys():
        if len(imgs[key])>mxlen:
            mxlen = len(imgs[key]) 

    # Note: -1 means no result
    l1error = -np.ones((len(imgs.keys()),mxlen,len(variances)))
    
    

    for ndelta,delta in enumerate(imgs.keys()):
        for nimg,img in enumerate(imgs[delta]):
        
            l1error[ndelta,nimg,:] = get_rate_img(img,variances=variances,C=C,niter=niter,nf=nf)
            
        print('Finished delta = ' + str(delta))            
        
    return l1error



    l1error = np.zeros(len(variances))
    
    for vcount,var in enumerate(variances):
    
        tv_res = tvcl.tv_recon(niter=niter,u0=img,dmode='l2',nf=nf,ld=1.0/(np.sqrt(var)*C),variance=var,check=niter)
        
        l1error[vcount] = np.abs(img - tv_res.u).sum()/np.prod(img.shape)


    return l1error




def create_counterexample_images(shape=(12,10000),nf=12,nshifts=10,mode='single_block_shift'):

    #Options for mode are:
    # 'single_block_shift': one pixel shift of a single edge
    # 'single_block_up' single block with value one, value shifted up by dx
    # 'single_block_down' single block with value one, value shifted down by dx
    # 'image_noise': random noise on the image
    # 'two_blocks_shift' two blocks with shift of two edges
    # 'two_blocks_value' two blocks with value change
    # 'data_noise' random noise on the data
    # 'data_noise_square' random noise on the data, suqare image
    
    par = mp.parameter({})
    par.nf = nf
    par.nshifts = nshifts


    #True data
    if (mode[:7] == 'single_') or mode == ('image_noise') or mode == ('data_noise'):
        u_true = np.zeros(shape)
        u_true[:,100:5000] = 1.0
    elif mode[:11] == 'two_blocks_':
        u_true = np.zeros(shape)
        u_true[:,2000:4000] = 1.0
        u_true[:,6000:8000] = 1.0    
    elif mode == 'orig_example':
        u_true = np.zeros(shape)
        u_true[:,10:500] = 1.0
    elif mode == 'dataset_image':
        
        fsource = 'results/data__dbounds_[0PKT01, 0PKT1, 10]__npoints_100__valid_True__make_invalid_False__nf_18__niter_100000'
        res_valid = mp.pload(fsource)
        data = select_working_subset(res_valid,tol=1e-05,max_len=5)
        u_true = data.imgs[0.04][3]

    elif (mode == 'data_noise_square') or (mode == 'image_noise_square'):
        u_true = np.zeros((120,120))
        u_true[:,10:60] = 1.0
    elif mode == 'data_noise_lowres':
        u_true = np.zeros((12,1000))
        u_true[:,100:500] = 1.0
    elif (mode == 'data_noise_highres') or (mode == 'highres_single_shift') or (mode == 'data_noise_highres_strong'):
        u_true = np.zeros((120,1000))
        u_true[:,10:500] = 1.0
    elif mode == 'square_single_shift':
        u_true = np.zeros((120,120))
        u_true[:,10:50] = 1.0
    elif mode == 'rectangle_single_shift':
        u_true = np.zeros((12,1000))
        u_true[:,10:500] = 1.0
    elif mode == 'double_shift':
        u_true = np.zeros((120,1000))
        u_true[:,10:500] = 1.0
    else:
        raise Warning("Unknown mode: " + mode)

    par.shape = u_true.shape
    N,M = par.shape

    # Create mask for data
    mask = np.zeros(par.shape)
    for i in range(-par.nf + 1,par.nf):
        for j in range(-par.nf + 1,par.nf):
            mask[i,j] = 1.0

    
    
    d0_true = np.fft.fft2(u_true)/np.sqrt(N*M)
    d0_true *= mask

    #Perturbed data
    us  = []
    variances = []

    for i in range(par.nshifts):
        
        u = np.zeros(par.shape)

        if mode == 'single_block_shift':
            u[:,100:5000+1+i] = 1.0

        elif mode == 'single_block_up':
            u[:,100:5000] = u_true[:,100:5000] + (i+1)*0.0008            

        elif mode == 'single_block_down':
            u[:,100:5000] = u_true[:,100:5000] - (i+1)*0.0008           

        elif (mode == 'image_noise') or (mode == 'image_noise_square'):
            u = u_true + np.random.normal(loc=0.0,scale=(i+1)*0.01,size=u_true.shape)
    
    
        elif mode == 'two_blocks_shift':
            u[:,2000:4000+1+i] = 1.0
            u[:,6000:8000+1+i] = 1.0

        elif mode == 'two_blocks_value':
            u[:,2000:4000] = 1.0 - (i+1)*0.001
            u[:,6000:8000] = 1.0 + (i+1)*0.001     
        elif mode == 'orig_example':
            u[:,10:500+1+i] = 1.0
        elif (mode == 'data_noise') or (mode == 'dataset_image') or (mode == 'data_noise_square') or (mode == 'data_noise_lowres') or (mode == 'data_noise_highres') or (mode == 'data_noise_highres_strong'):                
            # Create noisy image via noise in the data
            d0_tmp = np.fft.fft2(u_true)/np.sqrt(N*M)
            ncoeff = float((2*(nf-1) + 1)*(2*(nf-1) + 1))
            
            if not (mode=='data_noise_highres_strong'):
                var_tmp = np.power(10,-2 + 3*i/19)
            else:
                var_tmp = np.power(10,-1 + 3.5*i/19)
                
            d0_tmp.real += np.random.normal(loc=0.0,scale=np.sqrt(2*var_tmp/ncoeff),size=d0_tmp.shape)
            d0_tmp.imag += np.random.normal(loc=0.0,scale=np.sqrt(2*var_tmp/ncoeff),size=d0_tmp.shape)
            
            d0_tmp *= mask
            
            tmp = np.zeros(d0_tmp.shape,dtype=complex)
            #Symmetrize data
            for i in range(0,N):
                for j in range(0,M):
                    tmp[i,j] = 0.5*(d0_tmp[i,j] + d0_tmp[-i,-j].conj()) # variance

            d0_tmp = tmp #This should be data of a real image with noise level var_tmp
            
            # Get image from data
            u = np.fft.ifft2(d0_tmp)*np.sqrt(N*M).real
        elif mode == 'square_single_shift':
            u[:,10:50+1+i] = 1.0            
        elif (mode == 'highres_single_shift') or (mode == 'rectangle_single_shift'):
            u[:,10:500+1+i] = 1.0            
        elif mode == 'double_shift':
            u[:,10:500-1-i] = 1.0
            u[:,500-1-i:500+1+i] = 0.5
        else:
            raise Warning("Unknown mode: " + mode)
    
        us.append(np.copy(u))
        
        d0 = np.fft.fft2(u)/np.sqrt(N*M)
        d0 *= mask
        
        var = 0.5*np.square(np.abs(d0 - d0_true)).sum()
        variances.append(var)
        
        
    res = mp.output(par)
    
    
    res.u_true = u_true
    res.us = us
    res.variances = variances


    return res        
    
def get_counterexample_rate(data,C = 0.028,niter=100,nf=12,mode='img_input'):
# mode can be img_input or data_input

    l1error = np.zeros(len(data.variances))
    
    for vcount,var in enumerate(data.variances):
    
        if mode == 'img_input':
            tv_res = tvcl.tv_recon(niter=niter,u0=data.us[vcount],dmode='l2',nf=nf,ld=1.0/(np.sqrt(var)*C),variance=0,check=niter)
        elif mode == 'data_input':
            tv_res = tvcl.tv_recon(niter=niter,d0_input=data.d0s[vcount],dmode='l2',nf=nf,ld=1.0/(np.sqrt(var)*C),variance=0,check=niter)
        else:             
            raise Warning("Unknown mode: " + mode)
        
        l1error[vcount] = np.abs(data.u_true - tv_res.u).sum()/np.prod(data.u_true.shape)
        print('Finished: ' + str(vcount))

    return l1error





def create_counterexample_data(shape=(12,10000),nf=12,nshifts=20):

    par = mp.parameter({})
    
    par.shape = shape
    par.nf = nf
    par.nshifts = nshifts

    N,M = shape
    
    lshape = (shape[0],int(shape[1]/10))
    
    
    Nl,Ml = lshape

    # Create mask for data
    maskl = np.zeros(lshape)
    for i in range(-par.nf + 1,par.nf):
        for j in range(-par.nf + 1,par.nf):
            maskl[i,j] = 1.0

    #True low-res data
    u_true = np.zeros(lshape)
    u_true[:,10:500] = 1.0
    
    d0_true = np.fft.fft2(u_true)/np.sqrt(Nl*Ml)
    d0_true *= maskl

    #Perturbed data
    us  = []
    d0s  = []
    variances = []



    for i in range(par.nshifts):
        
        u = np.zeros(par.shape)
        u[:,100:5000+i+1] = 1.0
    
        us.append(np.copy(u))
        
        d0_tmp = np.fft.fft2(u)/np.sqrt(N*M)
        
        
        d0 = np.zeros(d0_true.shape).astype(d0_true.dtype)
        for i in range(-par.nf + 1,par.nf):
           for j in range(-par.nf + 1,par.nf):
                d0[i,j] = d0_tmp[i,j]
        

        d0 *= np.sqrt(Ml/M)
        
        d0s.append(d0)
          
        var = 0.5*np.square(np.abs(d0 - d0_true)).sum()
        variances.append(var)
        
        
    res = mp.output(par)
    
    
    res.u_true = u_true
    res.us = us
    
    res.d0_true = d0_true
    res.d0s = d0s
    res.variances = variances


    return res        


def create_counterexample_data_square(shape=(1200,1200),nf=12,nshifts=20):

    fac = 10

    par = mp.parameter({})
    
    par.shape = shape
    par.nf = nf
    par.nshifts = nshifts

    N,M = shape
    
    lshape = (int(shape[0]/fac),int(shape[1]/fac))
    
    
    Nl,Ml = lshape

    # Create mask for data
    maskl = np.zeros(lshape)
    for i in range(-par.nf + 1,par.nf):
        for j in range(-par.nf + 1,par.nf):
            maskl[i,j] = 1.0

    #True low-res data
    u_true = np.zeros(lshape)
    u_true[:,:60] = 1.0
    
    d0_true = np.fft.fft2(u_true)/np.sqrt(Nl*Ml)
    d0_true *= maskl

    #Perturbed data
    us  = []
    d0s  = []
    variances = []



    for shift in range(par.nshifts):
        
        u = np.zeros(par.shape)
        u[:,:(60*fac)+shift] = 1.0
    
        #us.append(np.copy(u))
        
        d0_tmp = np.fft.fft2(u)/np.sqrt(N*M)
        
        
        d0 = np.zeros(d0_true.shape).astype(d0_true.dtype)
        for i in range(-par.nf + 1,par.nf):
           for j in range(-par.nf + 1,par.nf):
                d0[i,j] = d0_tmp[i,j]
        

        d0 *= np.sqrt(Nl*Ml/(N*M))
        
        d0s.append(d0)
          
        var = 0.5*np.square(np.abs(d0 - d0_true)).sum()
        variances.append(var)
        
        print('Done with: ' + str(shift))
        
        
    res = mp.output(par)
    
    
    res.u_true = u_true
    res.us = us
    
    res.d0_true = d0_true
    res.d0s = d0s
    res.variances = variances


    return res   
    
    
