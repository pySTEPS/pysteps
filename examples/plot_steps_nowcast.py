#!/bin/env python
"""
STEPS nowcast
=============

This tutorial shows how to compute and plot an ensemble nowcast using Swiss
radar data.

"""

import matplotlib.pyplot as plt
import xarray as xr

from datetime import datetime
from pysteps import io, nowcasts, rcparams
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import conversion, dimension, transformation
from pysteps.visualization import plot_precip_field

# Set nowcast parameters
n_ens_members = 20
n_leadtimes = 6
seed = 24

###############################################################################
# Read precipitation field
# ------------------------
#
# First thing, the sequence of Swiss radar composites is imported, converted and
# transformed into units of dBR.


date = datetime.strptime("201701311200", "%Y%m%d%H%M")
data_source = "mch"

# Load data source config
root_path = rcparams.data_sources[data_source]["root_path"]
path_fmt = rcparams.data_sources[data_source]["path_fmt"]
fn_pattern = rcparams.data_sources[data_source]["fn_pattern"]
fn_ext = rcparams.data_sources[data_source]["fn_ext"]
importer_name = rcparams.data_sources[data_source]["importer"]
importer_kwargs = rcparams.data_sources[data_source]["importer_kwargs"]
timestep = rcparams.data_sources[data_source]["timestep"]

# Find the radar files in the archive
fns = io.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2
)

# Read the data from the archive
importer = io.get_method(importer_name, "importer")
precip_dataset = io.read_timeseries(fns, importer, **importer_kwargs)

# Convert to rain rate
precip_dataset = conversion.to_rainrate(precip_dataset)

# Upscale data to 2 km to limit memory usage
precip_dataset = dimension.aggregate_fields_space(precip_dataset, 2000)

# XR: change plot_precip_fields to take in an xarray and remove
# geodata, derive geodata from xarray?
geodata = {
    "projection": precip_dataset.attrs["projection"],
    "x1": precip_dataset.x.values[0],
    "x2": precip_dataset.x.values[-1],
    "y1": precip_dataset.y.values[0],
    "y2": precip_dataset.y.values[-1],
    "yorigin": "lower",  # is this always the case using xarray approach?
}

# Plot the rainfall field
plot_precip_field(precip_dataset["precip_intensity"][-1], geodata=geodata)
plt.show()

# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
# set the fill value to -15 dBR
precip_dataset = transformation.dB_transform(precip_dataset, threshold=0.1, zerovalue=-15.0)

# Set missing values with the fill value
precip_dataset["precip_intensity"] = precip_dataset["precip_intensity"].where(
    xr.ufuncs.isfinite(precip_dataset["precip_intensity"]), -15.0)

###############################################################################
# Deterministic nowcast with S-PROG
# ---------------------------------
#
# First, the motiong field is estimated using a local tracking approach based
# on the Lucas-Kanade optical flow.
# The motion field can then be used to generate a deterministic nowcast with
# the S-PROG model, which implements a scale filtering appraoch in order to
# progressively remove the unpredictable spatial scales during the forecast.

# Estimate the motion field
precip_dataset_w_motion = dense_lucaskanade(precip_dataset)

# BUG: sprog nowcast_method returns motion embedded in forecast dataset

# The S-PROG nowcast
nowcast_method = nowcasts.get_method("sprog")
precip_forecast = nowcast_method(
    precip_dataset_w_motion.isel(time=slice(-3, None)),
    n_leadtimes,
    n_cascade_levels=6,
    precip_thr=-10.0,
)

# XR:
# QUESTION: precip_forecast also contains velocity.
# What about the call to transformation.dB_transform ? 
# does it act only on precip? -> Yes 
# Will apply reverse transfomation if scale of transformation is dB
# Should make this clear in example
# Current IDEA: db_transform should take a dataArray instead of the whole dataset.

# Back-transform to rain rate
precip_forecast = transformation.dB_transform(precip_forecast, threshold=-10.0, inverse=True)

# XR: change plot_precip_fields to take in an xarray
# Plot the S-PROG forecast
plot_precip_field(
    precip_forecast['precip_intensity'][-1],
    geodata=geodata,
    title="S-PROG (+ %i min)" % (n_leadtimes * timestep),
)
plt.show()

###############################################################################
# As we can see from the figure above, the forecast produced by S-PROG is a
# smooth field. In other words, the forecast variance is lower than the
# variance of the original observed field.
# However, certain applications demand that the forecast retain the same
# statistical properties of the observations. In such cases, the S-PROG
# forecasts are of limited use and a stochatic approach might be of more
# interest.

###############################################################################
# Stochastic nowcast with STEPS
# -----------------------------
#
# The S-PROG approach is extended to include a stochastic term which represents
# the variance associated to the unpredictable development of precipitation. This
# approach is known as STEPS (short-term ensemble prediction system).

# The STEPS nowcast
nowcast_method = nowcasts.get_method("steps")
ensemble_precip_forecast = nowcast_method(
    precip_dataset_w_motion.isel(time=slice(-3, None)),
    n_leadtimes,
    n_ens_members,
    n_cascade_levels=6,
    precip_thr=-10.0,
    kmperpixel=2.0,
    timestep=timestep,
    noise_method="nonparametric",
    vel_pert_method="bps",
    mask_method="incremental",
    seed=seed,
)

# Back-transform to rain rates
ensemble_precip_forecast  = transformation.dB_transform(ensemble_precip_forecast , threshold=-10.0, inverse=True)

# Plot the ensemble mean
precip_forecast_mean = ensemble_precip_forecast["precip_intensity"].mean(dim="ens_number")
plot_precip_field(
    precip_forecast_mean[-1],
    geodata=geodata,
    title="Ensemble mean (+ %i min)" % (n_leadtimes * timestep),
)
plt.show()

###############################################################################
# The mean of the ensemble displays similar properties as the S-PROG
# forecast seen above, although the degree of smoothing also depends on
# the ensemble size. In this sense, the S-PROG forecast can be seen as
# the mean of an ensemble of infinite size.

# Plot some of the realizations
fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(221 + i)
    ax = plot_precip_field(
        ensemble_precip_forecast["precip_intensity"][i][-1], geodata=geodata, colorbar=False, axis="off"
    )
    ax.set_title("Member %02d" % i)
plt.tight_layout()
plt.show()

###############################################################################
# As we can see from these two members of the ensemble, the stochastic forecast
# mantains the same variance as in the observed rainfall field.
# STEPS also includes a stochatic perturbation of the motion field in order
# to quantify the its uncertainty.

###############################################################################
# Finally, it is possible to derive probabilities from our ensemble forecast.

# Compute exceedence probabilities for a 0.5 mm/h threshold
P = excprob(ensemble_precip_forecast["precip_intensity"][:, -1], 0.5)

# Plot the field of probabilities
plot_precip_field(
    P,
    geodata=geodata,
    ptype="prob",
    units="mm/h",
    probthr=0.5,
    title="Exceedence probability (+ %i min)" % (n_leadtimes * timestep),
)
plt.show()

# sphinx_gallery_thumbnail_number = 5
