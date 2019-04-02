#!/bin/env python

""" My first nowcast with pysteps

Pysteps 'hello world' script.

Compute your first deterministic radar extrapolation nowcast with pysteps.

More info: https://pysteps.github.io/
"""

import matplotlib.pylab as plt
import numpy as np
import pysteps as stp

# Get the two last observations

## read two consecutive radar fields 
filenames = ("sample_mch_radar_composite_00.gif","sample_mch_radar_composite_01.gif")
R = []
for fn in filenames:
    R_, _, metadata = stp.io.import_mch_gif(fn)
    R.append(R_)
    R_ = None
R = np.stack(R)

## convert to mm/h
R, metadata = stp.utils.to_rainrate(R, metadata)

## threshold the data
R[R<0.1] = 0.0

## copy the original data
R_ = R.copy()

## set NaN equal to zero
R_[~np.isfinite(R_)] = 0.0

## transform to dBR
R_, _ = stp.utils.dB_transform(R_)

# Compute motion field

oflow_method = stp.motion.get_method("lucaskanade")
UV = oflow_method(R_) 

# Perform the advection of the radar field

n_lead_times = 12 
adv_method = stp.extrapolation.get_method("semilagrangian") 
R_fct = adv_method(R_[-1,:,:], UV, n_lead_times, verbose=True)

## transform forecast values back to mm/h
R_fct, _ = stp.utils.dB_transform(R_fct, inverse=True)

# Plot the nowcast...

stp.plt.animate(R, R_fct=R_fct, UV=UV, nloops=5) 
