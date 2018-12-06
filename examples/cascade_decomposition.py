#!/bin/env python

"""Cascade decomposition

This script computes and plots the cascade decompositon of a single radar 
precipitation field.

More info: https://pysteps.github.io/
"""

import datetime
from matplotlib import cm, ticker
import matplotlib.pylab as plt
import numpy as np
import os

import pysteps as stp

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
num_cascade_levels  = 6
unit                = "mm/h"    # mm/h or dBZ
transformation      = "dB"      # None or dB 
adjust_domain       = None      # None or "square"

# Read-in the data
print('Read the data...')
startdate  = datetime.datetime.strptime(startdate_str, "%Y%m%d%H%M")

## import data specifications
ds = stp.rcparams.data_sources[data_source]

## find radar field filenames
input_files = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern, 
                                  ds.fn_ext, ds.timestep, 0, 0)

## read radar field files
importer = stp.io.get_method(ds.importer, type="importer")
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

# Plot the Fourier transform of the input field

F = abs(np.fft.fftshift(np.fft.fft2(R_)))
fig = plt.figure()
M,N = F.shape
im = plt.imshow(np.log(F**2), vmin=4, vmax=24, cmap=cm.jet, 
                extent=(-N/2, N/2, -M/2, M/2))
cb = fig.colorbar(im)
plt.xlabel("Wavenumber $k_x$")
plt.ylabel("Wavenumber $k_y$")
plt.title("Log-power spectrum of R")
plt.show()

# Cascade decomposition

## construct the Gaussian bandpass filter
bandapass_filter = stp.cascade.get_method("gaussian")
filter = bandapass_filter(R_.shape, num_cascade_levels, gauss_scale=0.5, 
                          gauss_scale_0=0.5)

## plot the bandpass filter weights
fig = plt.figure()
ax = fig.gca()
L = max(N, M)
for k in range(num_cascade_levels):
    ax.semilogx(np.linspace(0, L/2, len(filter["weights_1d"][k, :])), 
                filter["weights_1d"][k, :], "k-", 
                basex=pow(0.5*L/3, 1.0/(num_cascade_levels-2)))
                
ax.set_xlim(1, L/2)
ax.set_ylim(0, 1)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
xt = np.hstack([[1.0], filter["central_freqs"][1:]])
ax.set_xticks(xt)
ax.set_xticklabels(["%.2f" % cf for cf in filter["central_freqs"]])
ax.set_xlabel("Radial wavenumber $|\mathbf{k}|$")
ax.set_ylabel("Normalized weight")
ax.set_title("Bandpass filter weights")
plt.show()

## compute the cascade decomposition
decomposition = stp.cascade.get_method("fft")
cascade = decomposition(R_, filter)

## plot the normalized cascade levels (mean zero and standard deviation one)
grid_res_km = max(metadata["xpixelsize"], metadata["ypixelsize"])/1000.
mu,sigma = cascade["means"],cascade["stds"]
nrows = int(np.ceil((1+num_cascade_levels)/4.))
plt.subplot(nrows,4,1)
for k in range(num_cascade_levels+1):
    if k==0:
        plt.subplot(nrows,4,k+1)
        stp.plt.plot_precip_field(R, units=unit, title="Rainfall field", colorbar=False)
    else:
        R_k = cascade["cascade_levels"][k-1, :, :]
        R_k = (R_k - mu[k-1]) / sigma[k-1]
        plt.subplot(nrows,4,k+1)
        im = plt.imshow(R_k, cmap=cm.jet, vmin=-6, vmax=6)
        # cb = plt.colorbar(im)
        cb.set_label("Rainfall rate (dBR)")
        plt.xticks([])
        plt.yticks([])
        if filter["central_freqs"][k-1]==0:
            plt.title("Normalized cascade level %d (%i km)" % (k, L*grid_res_km))
        else:
            plt.title("Normalized cascade level %d (%i km)" % (k, L*1./filter["central_freqs"][k-1]*grid_res_km))
plt.show()
