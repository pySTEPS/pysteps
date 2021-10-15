# -*- coding: utf-8 -*-
"""
Blended forecast
====================

This tutorial shows how to construct a blended forecast from an ensemble nowcast
using the STEPS approach and a Numerical Weather Prediction (NWP) rainfall 
forecast. The used datasets are from the Royal Meteorological Insitute of Belgium.  

"""

from matplotlib import pyplot as plt
import numpy as np
import os
from datetime import datetime

import pysteps
from pysteps import io, rcparams, cascade, blending
from pysteps.utils import reprojection
from pysteps.visualization import plot_precip_field


################################################################################
# Read the radar images and the NWP forecast
# ------------------------------------------
#
# First, we import a sequence of 3 images of 5-minute radar composites
# and the corresponding NWP rainfall forecast that was available at that time.
#
# You need the pysteps-data archive downloaded and the pystepsrc file
# configured with the data_source paths pointing to data folders.

# Selected case
date_radar = datetime.strptime("202010310400", "%Y%m%d%H%M")
# The last NWP forecast was issued at 12:00 (and already gives the correct start time in pysteps data)
date_nwp = datetime.strptime("202010310000", "%Y%m%d%H%M")
radar_data_source = rcparams.data_sources["bom"]
nwp_data_source = rcparams.data_sources["bom_nwp"]


###############################################################################
# Load the data from the archive
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

root_path = radar_data_source["root_path"]
path_fmt = "prcp-c10/66/%Y/%m/%d"
fn_pattern = "66_%Y%m%d_%H%M00.prcp-c10"
fn_ext = radar_data_source["fn_ext"]
importer_name = radar_data_source["importer"]
importer_kwargs = radar_data_source["importer_kwargs"]
timestep = 10

# Find the radar files in the archive
fns = io.find_by_date(
    date_radar, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2
)

# Read the radar composites
importer = io.get_method(importer_name, "importer")
r_radar = io.read_timeseries(fns, importer, legacy=False, **importer_kwargs)
radar_data_xr = r_radar[-1, :, :]

# Get the metadata
radar_metadata = radar_data_xr.x.attrs.copy()
radar_metadata.update(**radar_data_xr.y.attrs)
radar_metadata.update(**radar_data_xr.attrs)

# Import the NWP data
filename = os.path.join(
    nwp_data_source["root_path"],
    datetime.strftime(date_nwp, nwp_data_source["path_fmt"]),
    datetime.strftime(date_nwp, nwp_data_source["fn_pattern"])
    + "."
    + nwp_data_source["fn_ext"],
)

nwp_data_xr = io.import_bom_nwp_xr(filename)
nwp_metadata = nwp_data_xr.x.attrs.copy()
nwp_metadata.update(**nwp_data_xr.y.attrs)
nwp_metadata.update(**nwp_data_xr.attrs)

# Only keep the NWP forecasts from the last radar observation time (2020-10-31 04:00)
# onwards
r_nwp = nwp_data_xr.sel(
    t=slice(np.datetime64("2020-10-31T04:00"), np.datetime64("2020-10-31T07:00"))
)


################################################################################
# Pre-processing steps
# --------------------

# Threshold the radar data and NWP forecast
r_radar.data[r_radar.data < radar_metadata["threshold"]] = 0.0
r_nwp.data[r_nwp.data < nwp_metadata["threshold"]] = 0.0

# Make sure the units are in mm/h
converter = pysteps.utils.get_method("mm/h")
r_radar, radar_metadata = converter(r_radar, radar_metadata)
r_nwp, nwp_metadata = converter(r_nwp, nwp_metadata)

# Plot the radar rainfall field and the first time step of the NWP forecast.
# For the initial time step (t=0), the NWP rainfall forecast is not that different
# from the observed radar rainfall, but it misses some of the locations and
# shapes of the observed rainfall fields. Therefore, the NWP rainfall forecast will
# initially get a low weight in the blending process.
date_str = datetime.strftime(date_radar, "%Y-%m-%d %H:%M")
plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_precip_field(
    r_radar[-1, :, :], geodata=radar_metadata, title=f"Radar observation at {date_str}"
)
plt.subplot(122)
plot_precip_field(
    r_nwp[0, :, :], geodata=nwp_metadata, title=f"NWP forecast at {date_str}"
)
plt.tight_layout()
plt.show()

# transform the data to dB
transformer = pysteps.utils.get_method("dB")
r_radar, radar_metadata = transformer(r_radar.values, radar_metadata, threshold=0.1)
transformer = pysteps.utils.get_method("dB")
r_nwp, nwp_metadata = transformer(r_nwp.values, nwp_metadata, threshold=0.1)

# Initial decomposition settings
decomp_method, recomp_method = cascade.get_method("fft")
bandpass_filter_method = "gaussian"
M, N = r_radar.shape[1:]
n_cascade_levels = 8
n_models = 1  # The number of NWP models to blend with the radar rainfall nowcast
filter_method = cascade.get_method(bandpass_filter_method)
filter = filter_method((M, N), n_cascade_levels)

# r_nwp has to be four dimentional (n_models, time, y, x).
# If we only use one model:
if r_nwp.ndim == 3:
    r_nwp = r_nwp[None, :]


################################################################################
# Decompose the NWP forecast
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

r_d_models = []
# Loop through the n_models
for i in range(r_nwp.shape[0]):
    r_d_models_ = []
    # Loop through the time steps
    for j in range(r_nwp.shape[1]):
        r_ = decomp_method(
            field=r_nwp[i, j, :, :],
            bp_filter=filter,
            normalize=True,
            compute_stats=True,
            compact_output=True,
        )
        r_d_models_.append(r_)
    r_d_models.append(r_d_models_)

r_d_models = np.array(r_d_models)


################################################################################
# Determine the velocity fields
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

oflow_method = pysteps.motion.get_method("lucaskanade")

# First for the radar images
v_radar = oflow_method(r_radar)

# Then for the NWP forecast
v_nwp = []
# Loop through the models
for n_model in range(r_nwp.shape[0]):
    # Loop through the timesteps. We need two images to construct a motion
    # field, so we can start from timestep 1. Timestep 0 will be the same
    # as timestep 1.
    _v_nwp_ = []
    for t in range(1, r_nwp.shape[1]):
        v_nwp_ = oflow_method(r_nwp[n_model, t - 1 : t + 1, :])
        _v_nwp_.append(v_nwp_)
        v_nwp_ = None
    # Add the velocity field at time step 1 to time step 0.
    _v_nwp_ = np.insert(_v_nwp_, 0, _v_nwp_[0], axis=0)
    v_nwp.append(_v_nwp_)
v_nwp = np.stack(v_nwp)


################################################################################
# The blended forecast
# --------------------

precip_forecast = blending.steps.forecast(
    R=r_radar,
    R_d_models=r_d_models,
    V=v_radar,
    V_models=v_nwp,
    timesteps=18,
    timestep=10.0,
    n_ens_members=1,
    n_cascade_levels=n_cascade_levels,
    blend_nwp_members=False,
    R_thr=radar_metadata["threshold"],
    kmperpixel=1.0,
    extrap_method="semilagrangian",
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    noise_method="nonparametric",
    noise_stddev_adj="auto",
    ar_order=2,
    vel_pert_method=None,
    conditional=False,
    probmatching_method="cdf",
    mask_method="incremental",
    callback=None,
    return_output=True,
    seed=None,
    num_workers=1,
    fft_method="numpy",
    domain="spatial",
    extrap_kwargs=None,
    filter_kwargs=None,
    noise_kwargs=None,
    vel_pert_kwargs=None,
    clim_kwargs=None,
    mask_kwargs=None,
    measure_time=False,
)

# Transform the data back into mm/h
precip_forecast, _ = converter(precip_forecast, radar_metadata)
r_radar, _ = converter(r_radar, radar_metadata)
r_nwp, _ = converter(r_nwp, nwp_metadata)


################################################################################
# Visualize the output
# ~~~~~~~~~~~~~~~~~~~~
#
# The NWP rainfall forecast has a lower weight than the radar-based extrapolation
# forecast at the issue time of the forecast (t=0). Therefore, the first time
# steps consist mostly of the extrapolation.
# However, near the end of the forecast (t=+3h), the NWP share in the blended
# forecast has become more important and the forecast starts to resemble the
# NWP forecast more.

# Plot the blended forecast
plt.figure(figsize=(10, 5))
plt.subplot(131)
plot_precip_field(
    precip_forecast[0, 2, :, :],
    geodata=radar_metadata,
    title="Blended forecast at t + 30 min",
)
plt.subplot(132)
plot_precip_field(
    precip_forecast[0, 5, :, :],
    geodata=radar_metadata,
    title="Blended forecast at t + 60 min",
)
plt.subplot(133)
plot_precip_field(
    precip_forecast[0, 17, :, :],
    geodata=radar_metadata,
    title="Blended forecast at t + 180 min",
)
plt.tight_layout()
plt.show()

# Plot the NWP forecast for comparison
plt.figure(figsize=(10, 5))
plt.subplot(131)
plot_precip_field(
    r_nwp[0, 3, :, :], geodata=nwp_metadata, title="NWP forecast at t + 30 min"
)
plt.subplot(132)
plot_precip_field(
    r_nwp[0, 6, :, :], geodata=nwp_metadata, title="NWP forecast at t + 60 min"
)
plt.subplot(133)
plot_precip_field(
    r_nwp[0, 18, :, :], geodata=nwp_metadata, title="NWP forecast at t + 180 min"
)
plt.tight_layout()
plt.show()


################################################################################
# References
# ~~~~~~~~~~
#
# Bowler, N. E., and C. E. Pierce, and A. W. Seed. 2004. "STEPS: A probabilistic
# precipitation forecasting scheme which merges an extrapolation nowcast with
# downscaled NWP." Forecasting Research Technical Report No. 433. Wallingford, UK.
#
# Bowler, N. E., and C. E. Pierce, and A. W. Seed. 2006. "STEPS: A probabilistic
# precipitation forecasting scheme which merges an extrapolation nowcast with
# downscaled NWP." Quarterly Journal of the Royal Meteorological Society 132(16):
# 2127-2155. https://doi.org/10.1256/qj.04.100
#
# Seed, A. W., and C. E. Pierce, and K. Norman. 2013. "Formulation and evaluation
# of a scale decomposition-based stochastic precipitation nowcast scheme." Water
# Resources Research 49(10): 6624-664. https://doi.org/10.1002/wrcr.20536
