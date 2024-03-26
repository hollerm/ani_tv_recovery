# This script evaluates L1-error and percentage of exact recovery for the recovery of images from Fourier data with different jump-point distances and number of Fourier measurements. It further evaluates the two cases of the assumption on consistent gradient directions being fulfilled or not.

# Imports
import numpy as np

import random
import itertools
import matpy as mp
import ani_tv_supp as tvsupp
import tv_recon_ocl as tvcl
import matplotlib.pyplot as plt




# Loop over two datasets, the first one with the assumption on consistend gradient directions being satisfied, the second one with this assumption not being satisfied
for fname in ['data__dbounds_[0DOT01, 0DOT1, 10]__npoints_100__valid_True__make_invalid_False','data__dbounds_[0DOT01, 0DOT1, 10]__npoints_100__valid_True__make_invalid_True__nflips_1']:
        # Loop over two different cutoff frequencies
        for nf in [12,18]:

                # Load pre-generated data
                data = mp.pload('data/' + fname)

                # Get reconstruction for frequency cutoff nf=12
                res = tvsupp.test_recon(data.img_data,niter=int(1e05),nf=nf)

                # Add data to result 
                res.point_data = data.point_data
                res.img_data = data.img_data

                # Store result
                res.fname = fname.replace('DOT','PKT')
                res.save(folder='results',outpars = ['nf','niter'])



