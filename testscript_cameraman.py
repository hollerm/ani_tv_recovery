## This script computes the exemplary results on the cameraman testimage

import numpy as np

import random
import itertools
import matpy as mp
import ani_tv_supp as tvsupp
import tv_recon_ocl as tvcl
import matplotlib.pyplot as plt


################
# Set parameter
imname = 'cameraman.png'

niter = int(1e06)
nfspace = [3,72,24]
nfs = [ int(i) for i in list(np.linspace(*nfspace))] 

# Load image
u0 = mp.imread('imsource/' + imname)

# Get result
res = tvsupp.test_cutoff_image(u0,niter=niter,nfs=nfs)


# Store parameter
res.nfs = nfs
res.u0 = u0
res.imname = imname
res.par.nfspace = nfspace

# Store result
res.fname = imname[:-4] + '_cutoff_test_'
res.save(folder='results',outpars = ['nfspace','niter'])
