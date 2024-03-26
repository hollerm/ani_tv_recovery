# This script evaluates numerically the convergence of vanishing noise.

import matpy as mp
import ani_tv_supp as tvsupp
import numpy as np


# Initialize Parameter class
par = mp.parameter({})

# Set data source
par.fsource = 'results/data__dbounds_[0PKT01, 0PKT1, 10]__npoints_100__valid_True__make_invalid_False__nf_18__niter_100000'


par.tol = 1e-05
par.max_len = 5
par.niter = int(1e05)
par.variances = np.power(10,np.linspace(-3,3,20))


par.C = 0.028

# Load result
res_valid = mp.pload(par.fsource)

# Select images as those where exact recovery was possible
data = tvsupp.select_working_subset(res_valid,tol=par.tol,max_len=par.max_len)

# Compute array of l1 errors
l1error = tvsupp.get_rate(data.imgs,variances=par.variances,C = par.C,niter=par.niter,nf=data.nf)

#Store output
res = mp.output()
res.par = par

res.fname = 'rate_var_m3_3_20'
res.l1error = l1error

res.save(folder='results',outpars=['max_len','niter'])

