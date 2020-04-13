#!/usr/bin/env python
# coding: utf-8

"""
My first precipitation nowcast
=====================

Here we will use pysteps to compute and plot an extrapolation nowcast using
the NSSL's Multi-Radar/Multi-Sensor System
`MRMS <https://www.nssl.noaa.gov/projects/mrms/>`_ rain rate product.

The MRMS precipitation product is available every 2 minutes, over the
contiguous US. Each precipitation composite has 3500 x 7000 grid points,
separated 1 km from each other.
"""

###############################################################################
# Getting the example data
# ------------------------
# 
# First of all, install the example data and configure the pysteps's default
# parameters by following this tutorial:
# 
# https://pysteps.readthedocs.io/en/latest/user_guide/example_data.html
# 
# Load the MRMS example data
# --------------------------
# 
# Now that we have installed the example data let's load the example MRMS
# dataset using the `load_dataset()` helper function from the `pysteps.datasets`
# module.
# 
# First, let's see what the default parameters used to load the dataset
# (stored in the `pystepsrc parameters`_ file).
#
# .. _`pystepsrc parameters`: https://pysteps.readthedocs.io/en/latest/user_guide/set_pystepsrc.html

import pysteps
from pprint import pprint  # Nicely print the data

pprint(pysteps.rcparams.data_sources['mrms'])

###############################################################################
# The default 'timestep' parameter is 2 minutes, that corresponts to the time
# interval at which the MRMS product is made available.
# 
# Let's load 1 hour and 10 minutes of data, which corresponds to 36 frames
# (precipitation images).

from pysteps.datasets import load_dataset

precipitation, metadata, timestep = load_dataset('mrms', frames=36)  # precipitation in mm/h

###############################################################################
# Let's have a look at the values returned by the `load_dataset()` function.
# 
# - `precipitation`: A numpy array with (time, latitude, longitude) dimensions.
# - `metadata`: A dictionary with additional information (pixel sizes, map projections, etc.).
# - `timestep`: Time separation between each sample (in minutes)


print(precipitation.shape)

###############################################################################
# Note that the shape of the precipitation is 4 times smaller than the raw MRMS
# data (3500 x 7000). The `load_dataset()` function uses the default parameters
# for the `importers` used to read the data.
# By default, the MRMS importer upscale the data x4
# (from ~1k resolution to ~4km) and uses single precision to reduce the memory
# requirements. With that upscaling, the memory footprint of this example
# dataset is ~200Mb instead of the 3.1Gb of the raw (3500 x 7000) data.

print(timestep)  # In minutes
pprint(metadata)

###############################################################################
# So far, we have 1 hour and 10 minutes of precipitation images, separated 2
# minutes apart from each other. But, how do we use that data to run a
# precipitation forecast?
# 
# A simple way is by extrapolating the precipitation fields based, assuming that
# they maintain a uniform motion without changes in intensity.
# This is commonly known by Lagrangian persistence.
# 
# The first step to run our nowcast is the estimation of the motion field from
# a sequence of past precipitation observations. We use the Lucas-Kanade (LK)
# optical flow method implemented in pysteps.
# This method follows a local tracking approach that relies on the OpenCV
# package. Local features are tracked in a sequence of two or more radar images.
# The scheme includes a final interpolation step to produce a smooth field of
# motion vectors. Other optical flow methods are also available in pysteps.
# Check the full list here:
# https://pysteps.readthedocs.io/en/latest/pysteps_reference/motion.html
# 
# Now, let's use the first 5 precipitation images (10 min) to obtain the motion
# field of the radar pattern and the remaining 30 images (1h) to evaluate the
# quality of our forecast.

# precipitation[0:5] -> Used to find motion (past data).
# Let's refer to this data as "training data".
train_precip = precipitation[0:5]

# precipitation[5:] -> Used evaluate forecasts (future data, not available in
# "real" forecast situation). Let's call it observed precipitation because we
# will use it to compare our forecast with the actual observations.
observed_precip = precipitation[3:]

###############################################################################
# Let's see how this precipitation event looks like using the
# `pysteps.visualization.plot_precip_field <https://pysteps.readthedocs.io/en/latest/generated/pysteps.visualization.precipfields.plot_precip_field.html>`_
# function.
#


from matplotlib import pyplot as plt
from pysteps.visualization import plot_precip_field

# Set the figure size that looks nice ;)
plt.figure(figsize=(9, 5), dpi=200)

# Plot the last rainfall field in the "training" data.
# train_precip[-1] -> Last available composite for nowcasting.
plot_precip_field(train_precip[-1], geodata=metadata, axis="off", map="cartopy")

# The next line is not actually needed if you are using jupyter notebooks
plt.show()

###############################################################################
# Did you note the **shaded grey** regions? Those are the regions were no valid
# observations where available to estimate the precipitation (e.g., due to
# ground clutter, no radar coverage, or radar beam blockage). Those regions
# need to be handle with care when we run our nowcast.
# 
# Data exploration
# ~~~~~~~~~~~~~~~~
# 
# Before we produce a forecast, let's explore precipitation data.
# In particular, let's see how the distribution of the rain rate values looks.

import numpy as np

# Let's use the last available composite for nowcasting from the "training" data
# (train_precip[-1]). Also, we will discard any invalid value.
valid_precip_values = train_precip[-1][~np.isnan(train_precip[-1])]

bins = np.concatenate(([-0.01, 0.01], np.linspace(1, 40, 39)))
plt.figure(figsize=(4, 4))
plt.hist(valid_precip_values, bins=bins, log=True, edgecolor='black')
plt.autoscale(tight=True, axis='x')
plt.xlabel("Rainrate [mm/h]")
plt.ylabel("Counts")
plt.title('Precipitation rain rate histogram')
plt.tight_layout()
plt.show()

###############################################################################
# The previous histogram shows that rain rate values have a non-Gaussian and
# asymmetric distribution that is bounded at zero. Also, the probability of
# occurrence decays extremely fast with increasing rain rate values (note the
# logarithmic y-axis).
# 
# 
# For better performance of the motion estimation algorithms, we can convert
# the rain rate values (in mm/h) to a more log-normal distribution  of rain
# rates by applying the following logarithmic transformation:
#
# .. math::
#   \begin{equation}
#   R\rightarrow
#   \begin{cases}
#       10\log_{10}R, & \text{if } R\geq 0.1\text{ mm h$^{-1}$} \\
#       -15,          & \text{otherwise}
#   \end{cases}
#   \end{equation}
# 
# The transformed precipitation corresponds to logarithmic rain rates in units
# of dBR. The value of −15 dBR is equivalent to assigning a rain rate of
# approximately 0.03 mm h−1to the zeros.

from pysteps.utils import transformation

# Log-transform the data to dBR. 
# The threshold to 0.1 mm/h sets the fill value to -15 dBR.
train_precip_dbr, metadata_dbr = transformation.dB_transform(train_precip,
                                                             metadata,
                                                             threshold=0.1,
                                                             zerovalue=-15.0)

###############################################################################
# Let's see how the **transformed precipitation** distribution looks.

# Only use the valid data!
valid_precip_dbr = train_precip_dbr[-1][~np.isnan(train_precip_dbr[-1])]

plt.figure(figsize=(4, 4))

counts, bins, _ = plt.hist(valid_precip_dbr, bins=40,
                           log=True, edgecolor="black")

plt.autoscale(tight=True, axis="x")
plt.xlabel("Rainrate [dB]")
plt.ylabel("Counts")
plt.title("Precipitation rain rate histogram")

# Let's add to the plot a lognormal distribution that fits that data.
import scipy

bin_center = (bins[1:] + bins[:-1]) * 0.5
bin_width = np.diff(bins)

# We will use only one composite for to fit function to speed up things.
# Remove the no precip areas.
precip_to_fit = valid_precip_dbr[valid_precip_dbr > -15]

fit_params = scipy.stats.lognorm.fit(precip_to_fit)

fitted_pdf = scipy.stats.lognorm.pdf(bin_center, *fit_params)
# Multiply pdf by the bin width and the total number of grid points.
# pdf -> total counts per bin.
fitted_pdf = fitted_pdf * bin_width * precip_to_fit.size

plt.plot(bin_center, fitted_pdf, label="Fitted log-normal")
plt.legend()
plt.tight_layout()
plt.show()

###############################################################################
# That looks more like a log-normal distribution. Note the large peak at -15dB.
# That peak corresponds to the "zero" (below threshold) precipitation.
# 
# Compute the nowcast
# -------------------
# 
# These are the minimal steps to compute a short-term forecast using lagrangian
# extrapolation of the precipitation patterns:
#  
#  1. Estimate the precipitation motion field.
#  1. The motion field to advect the most recent radar rainfall field and
#     produce an extrapolation forecast.
# 
# But before, 
# 
# Estimate the motion field
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now we can estimate the motion field. Here we use a local feature-tracking
# approach (Lucas-Kanade). However, check the other methods available in the
# `pysteps.motion <https://pysteps.readthedocs.io/en/latest/pysteps_reference/motion.html>`_
# module.

# Estimate the motion field with Lucas-Kanade
from pysteps import motion
from pysteps.visualization import plot_precip_field, quiver

oflow_method = motion.get_method("LK")
motion_field = oflow_method(train_precip_dbr)

################################################################################
# Plot the motion field
# ~~~~~~~~~~~~~~~~~~~~~
# Set the figure size that looks nice ;)
plt.figure(figsize=(9, 5))

# Plot the last rainfall field in the "training" data.
# Remember to use the mm/h precipitation data since plot_precip_field assumes
# mm/h by default. You can change this behavior using the "units" keyword.
plot_precip_field(train_precip[-1], geodata=metadata, axis="off")
quiver(motion_field, geodata=metadata, step=50)
plt.show()

###############################################################################
# Extrapolate the observations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The final step is to advect the most recent radar rainfall field along the
# estimated motion field, producing an extrapolation forecast.

from pysteps import nowcasts

# Extrapolate the last radar observation
extrapolate = nowcasts.get_method("extrapolation")

# You can use the precipitation observations directly in mm/h for this step.
last_observation = train_precip[-1]

last_observation[~np.isfinite(last_observation)] = metadata["zerovalue"]

n_leadtimes = 30  # 1h
precip_forecast = extrapolate(train_precip[-1], motion_field, n_leadtimes)

precip_forecast.shape

###############################################################################
# Let's see how the last forecast time looks like.

# Plot precipitation at the end of the forecast period.
plt.figure(figsize=(9, 5))
plot_precip_field(precip_forecast[-1], geodata=metadata,
                  axis="off", map='cartopy')
plt.show()

###############################################################################
# Evaluate the forecast quality
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The Fractions Skill Score (FSS) provides an intuitive assessment of the
# dependency of skill on spatial scale and intensity. This makes the FSS an
# ideal skill score for high-resolution precipitation forecasts.
# 
# More rigorously, the FSS is a neighborhood spatial verification method that
# directly compares the fractional coverage of events in windows surrounding
# the observations and forecasts. The FSS varies from 0 (total mismatch)
# to 1 (perfect forecast). For most situations, an FSS value of >0.5 serves as
# a good indicator of a useful forecast
# (`Skok and Roberts, 2016 <https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.2849>`_).


from pysteps import verification

fss = verification.get_method("FSS")

# Compute fractions skill score (FSS) for all lead times for different scales
# using a 1 mm/h detection threshold.
scales = [
    2,
    4,
    8,
    16,
    32,
    64,
]  # In grid points.

scales_in_km = np.array(scales) * 4

thr = 1.0  # in mm/h
score = []
for i in range(n_leadtimes - 2):
    score_ = []
    for scale in scales:
        score_.append(
            fss(precip_forecast[i, :, :], observed_precip[i, :, :], thr, scale)
        )
    score.append(score_)

plt.figure(figsize=(6, 4))
x = np.arange(1, n_leadtimes - 1) * timestep
plt.plot(x, score, lw=2.0)
plt.xlabel("Lead time [min]")
plt.ylabel("FSS ( > 1.0 mm/h ) ")
plt.title("Fractions Skill Score")
plt.legend(
    scales_in_km,
    title="Scale [km]",
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    bbox_transform=plt.gca().transAxes,
)
xticks = np.linspace(0, 60, 7)
xticks[0] = 5
plt.xticks(xticks)
plt.autoscale(axis="x", tight=True)
plt.tight_layout()
plt.show()

# sphinx_gallery_thumbnail_number = 1