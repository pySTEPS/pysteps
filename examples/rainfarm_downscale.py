#!/bin/env python
"""
Precipitation downscaling with RainFARM
=======================================

This example script shows how to use the stochastic downscaling method RainFARM
available in pysteps.

RainFARM is a downscaling algorithm for rainfall fields developed by Rebora et
al. (2006). The method can represent the realistic small-scale variability of the
downscaled precipitation field by means of Gaussian random fields.

Steps:
    1. Read the input precipitation data.
    2. Upscale the precipitation field.
    3. Downscale the field to its original resolution using RainFARM with defaults.
    4. Downscale with smoothing.
    5. Downscale with spectral fusion.
    6. Downscale with smoothing and spectral fusion.

References:

    Rebora, N., L. Ferraris, J. von Hardenberg, and A. Provenzale, 2006: RainFARM:
    Rainfall downscaling by a filtered autoregressive model. J. Hydrometeor., 7,
    724–738.

    D D'Onofrio, E Palazzi, J von Hardenberg, A Provenzale, and S Calmanti, 2014:
    Stochastic rainfall downscaling of climate models. J. Hydrometeorol., 15(2):830–843.
"""

import logging
import os
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from pysteps import io, rcparams
from pysteps.downscaling import rainfarm
from pysteps.utils import aggregate_fields_space, square_domain, to_rainrate
from pysteps.visualization import plot_precip_field
from pysteps.xarray_helpers import geodata_from_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
# Read the input data
# -------------------
#
# As first step, we need to import the precipitation field that we are going
# to use in this example.


def read_precipitation_data(file_path):
    """Read and process precipitation data from a file."""
    precip_dataset = io.import_mch_gif(
        file_path, product="AQC", unit="mm", accutime=5.0
    )
    # rainfarm.downscale() requires a time dimension, which a single imported
    # frame does not have.
    precip_dataset = precip_dataset.expand_dims(
        time=[np.datetime64(datetime(2016, 7, 11, 19, 32, 10))]
    )
    precip_dataset = to_rainrate(precip_dataset)
    precip_dataset = square_domain(precip_dataset, "crop")
    return precip_dataset


# Import the example radar composite
root_path = rcparams.data_sources["mch"]["root_path"]
filename = os.path.join(root_path, "20160711", "AQC161932100V_00005.801.gif")

# Read and process data
precip_dataset = read_precipitation_data(filename)
precip_var = precip_dataset.attrs["precip_var"]

# Nicely print the metadata
pprint(precip_dataset[precip_var].attrs)

# Plot the original rainfall field
plot_precip_field(
    precip_dataset[precip_var].isel(time=0),
    geodata=geodata_from_dataset(precip_dataset),
)
plt.title("Original Rainfall Field")
plt.show()

# Assign the fill value to all the Nans
precip = precip_dataset[precip_var].values
precip[~np.isfinite(precip)] = precip_dataset[precip_var].attrs["zerovalue"]

###############################################################################
# Upscale the field
# -----------------
#
# To test our downscaling method, we first need to upscale the original field to
# a lower resolution. This is only for demo purposes, as we need to artificially
# create a lower resolution field to apply our downscaling method.
# We are going to use a factor of 16 x.


def upscale_field(precip_dataset, scale_factor):
    """Upscale the precipitation field by a given scale factor."""
    upscaled_resolution = precip_dataset.x.attrs["stepsize"] * scale_factor
    return aggregate_fields_space(precip_dataset, upscaled_resolution)


scale_factor = 16
precip_dataset_lr = upscale_field(precip_dataset, scale_factor)

# Plot the upscaled rainfall field
plt.figure()
plot_precip_field(
    precip_dataset_lr[precip_var].isel(time=0),
    geodata=geodata_from_dataset(precip_dataset_lr),
)
plt.title("Upscaled Rainfall Field")
plt.show()

###############################################################################
# Downscale the field
# -------------------
#
# We can now use RainFARM to downscale the precipitation field.

# Fixed seed for reproducible stochastic noise fields
np.random.seed(42)

# Basic downscaling
precip_dataset_hr = rainfarm.downscale(precip_dataset_lr, ds_factor=scale_factor)

# Plot the downscaled rainfall field
plt.figure()
plot_precip_field(
    precip_dataset_hr[precip_var].isel(time=0),
    geodata=geodata_from_dataset(precip_dataset_hr),
)
plt.title("Downscaled Rainfall Field")
plt.show()

###############################################################################
# Downscale with smoothing
# ------------------------
#
# Add smoothing with a Gaussian kernel during the downscaling process.

precip_dataset_hr_smooth = rainfarm.downscale(
    precip_dataset_lr, ds_factor=scale_factor, kernel_type="gaussian"
)

# Plot the downscaled rainfall field with smoothing
plt.figure()
plot_precip_field(
    precip_dataset_hr_smooth[precip_var].isel(time=0),
    geodata=geodata_from_dataset(precip_dataset_hr_smooth),
)
plt.title("Downscaled Rainfall Field with Gaussian Smoothing")
plt.show()

###############################################################################
# Downscale with spectral fusion
# ------------------------------
#
# Apply spectral merging as described in D'Onofrio et al. (2014).

precip_dataset_hr_fusion = rainfarm.downscale(
    precip_dataset_lr, ds_factor=scale_factor, spectral_fusion=True
)

# Plot the downscaled rainfall field with spectral fusion
plt.figure()
plot_precip_field(
    precip_dataset_hr_fusion[precip_var].isel(time=0),
    geodata=geodata_from_dataset(precip_dataset_hr_fusion),
)
plt.title("Downscaled Rainfall Field with Spectral Fusion")
plt.show()

###############################################################################
# Combined Downscale with smoothing and spectral fusion
# -----------------------------------------------------
#
# Apply both smoothing with a Gaussian kernel and spectral fusion during the
# downscaling process to observe the combined effect.

precip_dataset_hr_combined = rainfarm.downscale(
    precip_dataset_lr,
    ds_factor=scale_factor,
    kernel_type="gaussian",
    spectral_fusion=True,
)

# Plot the downscaled rainfall field with smoothing and spectral fusion
plt.figure()
plot_precip_field(
    precip_dataset_hr_combined[precip_var].isel(time=0),
    geodata=geodata_from_dataset(precip_dataset_hr_combined),
)
plt.title("Downscaled Rainfall Field with Gaussian Smoothing and Spectral Fusion")
plt.show()

###############################################################################
# Remarks
# -------
#
# Currently, the pysteps implementation of RainFARM only covers spatial downscaling.
# That is, it can improve the spatial resolution of a rainfall field. However, unlike
# the original algorithm from Rebora et al. (2006), it cannot downscale the temporal
# dimension.

# sphinx_gallery_thumbnail_number = 2
