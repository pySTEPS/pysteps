#!/bin/env python
"""
Precipitation downscaling with RainFARM
=======================================

This example script shows how to use the stochastic downscaling method RainFARM
available in pysteps.

RainFARM is a downscaling algorithm for rainfall developed by Rebora et al. (2006).
This method represents a realistic small-scale variability to the precipitation field 
using Gaussian random fields generated using a power-law spectral scaling.

"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint

from pysteps import io, rcparams
from pysteps.utils import aggregate_fields_space, square_domain, to_rainrate
from pysteps.downscaling import rainfarm
from pysteps.visualization import plot_precip_field

###############################################################################
# Read the input data
# -------------------
#
# As first step, we need to import the precipitation field that we are going
# to use in this example.

# Import the example radar composite
root_path = rcparams.data_sources["mch"]["root_path"]
filename = os.path.join(root_path, "20160711", "AQC161932100V_00005.801.gif")
precip, _, metadata = io.import_mch_gif(
    filename, product="AQC", unit="mm", accutime=5.0
)

# Convert to mm/h
precip, metadata = to_rainrate(precip, metadata)

# Reduce to a square domain
precip, metadata = square_domain(precip, metadata, "crop")

# Nicely print the metadata
pprint(metadata)

# Plot the original rainfall field
plt.figure()
plot_precip_field(precip, geodata=metadata)

# Assign the fill value to all the Nans
precip[~np.isfinite(precip)] = metadata["zerovalue"]

###############################################################################
# Upscale the field
# -----------------
#
# In order to test our downscaling method, we first need to
# upscale the original field to a lower resolution. We are going
# to use an upscaling factor of 16 x.

upscale_to = metadata["xpixelsize"] * 16  # upscaling factor : 16 x
precip_lr, metadata_lr = aggregate_fields_space(precip, metadata, upscale_to)

# Plot the upscaled rainfall field
plt.figure()
plot_precip_field(precip_lr, geodata=metadata_lr)

###############################################################################
# Downscale the field
# -------------------
#
# We can now use RainFARM to generate stochastic realizations of the downscaled
# precipitation field.

plt.figure()
# Set the number of stochastic realizations
num_realizations = 2
precip_hr = []
# Per realization, generate a stochastically downscaled precipitation field
# and plot it
for n in range(num_realizations):
    precip_hr = rainfarm.downscale(precip_lr, ds_factor=16)
    plt.subplot(1, num_realizations, n + 1)
    plot_precip_field(precip_hr, geodata=metadata, axis="off", colorbar=False)
plt.tight_layout()

###############################################################################
# Remarks
# -------
#
# The pysteps implementation of RainFARM currently covers spatial downscaling only,
# that is, it can improve spatial resolution of a rainfall field, but unlikely
# the original algorithm from Rebora et al. (2006), it cannot downscale the temporal
# dimension.

###############################################################################
# References
# ----------
#
# Rebora, N., L. Ferraris, J. von Hardenberg, and A. Provenzale, 2006: RainFARM:
# Rainfall downscaling by a filtered autoregressive model. J. Hydrometeor., 7,
# 724–738.
