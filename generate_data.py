# This script generates all synthetic data for the experiments of the paper


# Imports
import numpy as np
import matpy as mp
import ani_tv_supp as tvsupp


################
# Set parameter
par = mp.parameter({})

par.seed = 10
par.imgsz = 120

par.npoints = 100
par.maxjumps = 20
par.valid = True
# Set data bounds
par.dbounds = [0.01,0.1,10]
# Number of color flips to violate the assumption on consistent gradient directions
par.nflips = 1


## Create data

for make_invalid in [False,True]:

        par.make_invalid = make_invalid
        
        # Get data points
        deltas = np.linspace(*par.dbounds)
        point_data = tvsupp.get_jump_points(deltas,npoints=par.npoints,maxjumps=par.maxjumps,imgsz=par.imgsz,seed=par.seed)

        # Get image coloring
        img_data = tvsupp.get_image_database(point_data,imgsz=par.imgsz,seed=par.seed,valid=par.valid)

        # Make valid points invalid
        if par.make_invalid:
            img_data = tvsupp.make_invalid(img_data,point_data,nflips=par.nflips)


        # Save output
        res = mp.output({})

        res.point_data = point_data
        res.img_data = img_data

        res.par = par

        res.fname = 'data'
        if par.make_invalid:
                res.save(folder='data',outpars = ['dbounds','npoints','valid','make_invalid','nflips'])
        else:
                res.save(folder='data',outpars = ['dbounds','npoints','valid','make_invalid'])        





