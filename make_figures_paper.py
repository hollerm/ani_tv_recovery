# Imports
import numpy as np

import random
import itertools
import matpy as mp
import ani_tv_supp as tvsupp
import tv_recon_ocl as tvcl
import matplotlib.pyplot as plt

import matpy as mp


## Exakt reconstruction with and without gradient condition

# Set tolerance and export flag
tol = 1e-4
export = True


# Gradient condition fulfilled, cutoff frequency 12
phi = 12
res_valid = mp.pload('results/data__dbounds_[0PKT01, 0PKT1, 10]__npoints_100__valid_True__make_invalid_False__nf_' + str(phi) + '__niter_100000')
title = 'Assumption 1 fulfilled, $\Phi = ' + str(phi) + '$'
error_valid = tvsupp.eval_result(res_valid,tol=tol,show=True,export=export,title=title)

# Gradient condition fulfilled, cutoff frequency 18
phi = 18
res_valid = mp.pload('results/data__dbounds_[0PKT01, 0PKT1, 10]__npoints_100__valid_True__make_invalid_False__nf_' + str(phi) + '__niter_100000')
title = 'Assumption 1 fulfilled, $\Phi = ' + str(phi) + '$'
error_valid = tvsupp.eval_result(res_valid,tol=tol,show=True,export=export,title=title)

# Gradient condition not fulfilled, cutoff frequency 12
phi = 12
res_invalid = mp.pload('results/data__dbounds_[0PKT01, 0PKT1, 10]__npoints_100__valid_True__make_invalid_True__nflips_1__nf_' + str(phi) + '__niter_100000')
title = 'Assumption 1 not fulfilled, $\Phi = ' + str(phi) + '$'
error_valid = tvsupp.eval_result(res_invalid,tol=tol,show=True,export=export,title=title)

# Gradient condition not fulfilled, cutoff frequency 118
phi = 18
res_invalid = mp.pload('results/data__dbounds_[0PKT01, 0PKT1, 10]__npoints_100__valid_True__make_invalid_True__nflips_1__nf_' + str(phi) + '__niter_100000')
title = 'Assumption 1 not fulfilled, $\Phi = ' + str(phi) + '$'
error_valid = tvsupp.eval_result(res_invalid,tol=tol,show=True,export=export,title=title)


## Convergence rate for vanishing noise

res_rate = mp.pload('results/rate_var_m3_3_20__max_len_5__niter_100000.pkl')
tvsupp.eval_results_rate(res_rate,export=True)


## Print cameramen images

#Load data
res_real = mp.pload('results/cameraman_cutoff_test___nfspace_[3, 72, 24]__niter_1000000.pkl')
# Save all images
for idx,nf in enumerate(list(res_real.nfs)):
    mp.imsave('images/cameraman_nf_' + str(nf) + '.png',res_real.imgs[idx])



## Create two synthetic example images
fname = 'data__dbounds_[0DOT01, 0DOT1, 10]__npoints_100__valid_True__make_invalid_False'

data = mp.pload('data/' + fname)


delta = 0.02
pos = np.argmin(np.abs(np.array(list(data.img_data.keys())) - delta)) #Numerically stable way to select correct key
mp.imsave('images/example_image_delta_' + str(delta) + '.png',data.img_data[list(data.img_data.keys())[pos]][0])

delta = 0.09
pos = np.argmin(np.abs(np.array(list(data.img_data.keys())) - delta)) #Numerically stable way to select correct key
mp.imsave('images/example_image_delta_' + str(delta) + '.png',data.img_data[list(data.img_data.keys())[pos]][0])

