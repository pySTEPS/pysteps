#!/bin/env python

"""Generation of stochastic noise

This tutorial demonstrates the stochastic noise generators.

More info: https://pysteps.github.io/
"""

import datetime
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import os

import pysteps as stp
import config as cfg

# List of case studies that can be used in this tutorial

#+-------+--------------+-------------+----------------------------------------+
#| event |  start_time  | data_source | description                            |
#+=======+==============+=============+========================================+
#|  01   | 201701311030 |     mch     | orographic precipitation               |
#+-------+--------------+-------------+----------------------------------------+
#|  02   | 201505151630 |     mch     | non-stationary field, apparent rotation|
#+-------+--------------+------------------------------------------------------+
#|  03   | 201609281530 |     fmi     | stratiform rain band                   |
#+-------+--------------+-------------+----------------------------------------+
#|  04   | 201705091130 |     fmi     | widespread convective activity         |
#+-------+--------------+-------------+----------------------------------------+
#|  05   | 201806161100 |     bom     | bom example data                       |
#+-------+--------------+-------------+----------------------------------------+

# Set parameters for this tutorial

## input data (copy/paste values from table above)
startdate_str = "201701311030"
data_source   = "mch"

## parameters
r_threshold         = 0.1 # [mm/h]
noise_method        = "nonparametric" # parametric, nonparametric, ssft
num_realizations    = 7
unit                = "mm/h"    # mm/h or dBZ
transformation      = "dB"      # None or dB 
adjust_domain       = None      # None or "square"
seed                = 42        # for reproducibility

# Read-in the data
print('Read the data...')
startdate  = datetime.datetime.strptime(startdate_str, "%Y%m%d%H%M")

## import data specifications
ds = cfg.get_specifications(data_source)

## find radar field filenames
input_files = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern, 
                                  ds.fn_ext, ds.timestep, 0, 0)

## read radar field files
importer = stp.io.get_method(ds.importer)
R, _, metadata = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
R = R.squeeze() # since this contains just one frame
Rmask = np.isnan(R)

# Prepare input files
print("Prepare the data...")

## if necessary, convert to rain rates [mm/h]    
converter = stp.utils.get_method(unit)
R, metadata = converter(R, metadata)

## threshold the data
R[R<r_threshold] = 0.0
metadata["threshold"] = r_threshold

## if requested, make sure we work with a square domain
reshaper = stp.utils.get_method(adjust_domain)
R_, metadata_ = reshaper(R, metadata, method="pad")

## if requested, transform the data
transformer = stp.utils.get_method(transformation)
R_, metadata_ = transformer(R_, metadata_)

## set NaN equal to zero
R_[~np.isfinite(R_)] = metadata_["zerovalue"]

# Noise generation

# initialize the filter for generating the noise
# the Fourier spectrum of the input field in dBR is used as a filter (i.e. the 
# "nonparametric" method)
# this produces a noise field having spatial correlation structure similar to 
# the input field
init_noise, generate_noise = stp.noise.get_method(noise_method)
F = init_noise(R_)

# plot four realizations of the stochastic noise
nrows = int(np.ceil((1 + num_realizations)/4.))
plt.subplot(nrows,4,1)
for k in range(num_realizations+1):
    if k==0:
        plt.subplot(nrows,4,k+1)
        stp.plt.plot_precip_field(R, units=metadata["unit"], 
        title="Rainfall field", colorbar=False)
    else:
        ## generate the noise
        N = generate_noise(F, seed=seed+k)
        
        ## if necessary, reshape to orginal domain
        N, _ = reshaper(N, metadata_, inverse=True)
        
        plt.subplot(nrows,4,k+1)
        plt.imshow(N, cmap=cm.jet)
        plt.xticks([])
        plt.yticks([])
        plt.title("Noise field %d" % (k+1))

plt.show()
