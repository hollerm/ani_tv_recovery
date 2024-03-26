# Imports
import matpy as mp
import tv_recon_ocl as tvcl
import numpy as np


niter = 2000 # Total number of iterations (set, e.g., to 20000 to get close to an optimal solution)
check=10 # Compute energy every "check" iteration (the higher check, the faster)


# Select sourc image
imname = 'imsource/cameraman.png'

# Compute tv reconstruction using:
#  dmode \in ['exact','l2'] data fidelity
#  ld as regularization parameter (only relevant for dmode='l2'
#  Fourier frequencys f with |f|<nf
res = tvcl.tv_recon(niter=niter,imname=imname,check=check,dmode='exact',ld=0.7,nf=10)

# Store result using 'nf' and 'niter' to generate the filename
res.save(folder='results',outpars = ['nf','niter'])

# Export zero-fill reconstruction rec0 and reconstructed image u
mp.imsave('images/cameraman_demo_rec0.png',res.rec0,rg=[res.u0.min(),res.u0.max()])
mp.imsave('images/cameraman_demo_rec.png',res.u,rg=[res.u0.min(),res.u0.max()])

# Difference to ground truth
print('Difference to ground truth: ' + str(np.abs(res.u- res.u0).sum()/np.prod(res.u.shape)))


