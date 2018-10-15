#!/bin/env python

"""Generation of stochastic noise

This script shows the stochastic noise generators included in pysteps.

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
n_prvs_times        = 3
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
                                  ds.fn_ext, ds.timestep, n_prvs_times, 0)

## read radar field files
importer = stp.io.get_method(ds.importer, type="importer")
R, _, metadata = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
Rmask = np.isnan(R)

# Prepare input files
print("Prepare the data...")

## if necessary, convert to rain rates [mm/h]    
converter = stp.utils.get_method("mm/h")
R, metadata = converter(R, metadata)

## threshold the data
R[R<r_threshold] = 0.0
metadata["threshold"] = r_threshold

## convert the data
converter = stp.utils.get_method(unit)
R, metadata = converter(R, metadata)

## transform the data
transformer = stp.utils.get_method(transformation)
R, metadata = transformer(R, metadata)

## if requested, make sure we work with a square domain
reshaper = stp.utils.get_method(adjust_domain)
R, metadata = reshaper(R, metadata, method="pad")

## set NaN equal to zero
R[~np.isfinite(R)] = metadata["zerovalue"]

# Noise generation

# initialize the filter for generating the noise
# the Fourier spectrum of the input field in dBR is used as a filter (i.e. the 
# "nonparametric" method)
# this produces a noise field having spatial correlation structure similar to 
# the input field
print("Initialize the filter...")
init_noise, generate_noise = stp.noise.get_method(noise_method)
F = init_noise(R)

# generate the noise
print("Generate the noise fields...")
N=[]
for k in range(num_realizations):
    N.append(generate_noise(F, seed=seed+k))

# plot four realizations of the stochastic noise
print("Plot the results...")

## convert to rain rates [mm/h]    
converter = stp.utils.get_method("mm/h")
R, metadata = converter(R, metadata)

nrows = int(np.ceil((1 + num_realizations)/4.))
plt.subplot(nrows,4,1)
for k in range(num_realizations+1):
    if k==0:
        plt.subplot(nrows,4,k+1)
        stp.plt.plot_precip_field(R[-1,:,:], units=metadata["unit"], 
        title="Rainfall field", colorbar=False)
    else:
        
        N_ = N[k-1]
        
        plt.subplot(nrows,4,k+1)
        plt.imshow(N_, cmap=cm.jet)
        plt.xticks([])
        plt.yticks([])
        plt.title("Noise field %d" % (k))

plt.show()
