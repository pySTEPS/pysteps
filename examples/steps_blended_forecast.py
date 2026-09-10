# -*- coding: utf-8 -*-
"""
Blended forecast
====================

This tutorial shows how to construct a blended forecast from an ensemble nowcast
using the STEPS approach and a Numerical Weather Prediction (NWP) rainfall
forecast. The used datasets are from the Bureau of Meteorology, Australia.

"""

import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

import pysteps
from pysteps import blending, io, nowcasts, rcparams
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field
from pysteps.xarray_helpers import (
    convert_input_to_xarray_dataset,
    convert_output_to_xarray_dataset,
    geodata_from_dataset,
)

################################################################################
# Read the radar images and the NWP forecast
# ------------------------------------------
#
# First, we import a sequence of 3 images of 10-minute radar composites
# and the corresponding NWP rainfall forecast that was available at that time.
#
# You need the pysteps-data archive downloaded and the pystepsrc file
# configured with the data_source paths pointing to data folders.
# Additionally, the pysteps-nwp-importers plugin needs to be installed, see
# https://github.com/pySTEPS/pysteps-nwp-importers.

# Selected case
date_radar = datetime.strptime("202010310400", "%Y%m%d%H%M")
# The last NWP forecast was issued at 00:00
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
timestep = 10.0

# Find the radar files in the archive
fns = io.find_by_date(
    date_radar, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2
)

# Read the radar composites
importer = io.get_method(importer_name, "importer")
radar_dataset = io.read_timeseries(fns, importer, **importer_kwargs)

# Import the NWP data. The pysteps-nwp-importers plugin has not been migrated
# to the new xarray-based data model yet, so it still returns the old-style
# (precip, quality, metadata) tuple. We wrap it into an xarray dataset by hand.
filename = os.path.join(
    nwp_data_source["root_path"],
    datetime.strftime(date_nwp, nwp_data_source["path_fmt"]),
    datetime.strftime(date_nwp, nwp_data_source["fn_pattern"])
    + "."
    + nwp_data_source["fn_ext"],
)

nwp_importer = io.get_method("bom_nwp", "importer")
nwp_precip, _, nwp_metadata = nwp_importer(filename)

# Only keep the NWP forecasts from the last radar observation time (2020-10-31 04:00)
# onwards

nwp_precip = nwp_precip[24:43, :, :]

# blending.steps.forecast requires model_dataset to carry an "ens_number"
# dimension (n_models/members). We only have one (deterministic) NWP model
# here, so we add a leading dimension of size 1.
if nwp_precip.ndim == 3:
    nwp_precip = nwp_precip[None, :]

# The importer sets metadata["transform"] = None to indicate "no transform",
# but convert_input_to_xarray_dataset/to_rainrate expect the key to be absent
# in that case.
nwp_metadata = dict(nwp_metadata)
if nwp_metadata.get("transform") is None:
    nwp_metadata.pop("transform", None)

# model_dataset's time grid must line up exactly with the radar dataset's
# timestep grid (blending.steps.forecast selects the model data with
# `.sel(time=...)` using the radar timestep), which is the case here since
# both the radar and NWP data have a native 10-minute cadence and index 0 of
# the sliced nwp_precip array corresponds to date_radar.
model_dataset = convert_input_to_xarray_dataset(
    nwp_precip,
    None,
    nwp_metadata,
    startdate=date_radar,
    timestep=int(timestep * 60),
)


################################################################################
# Pre-processing steps
# --------------------

# Make sure the units are in mm/h
radar_dataset = conversion.to_rainrate(radar_dataset)
model_dataset = conversion.to_rainrate(model_dataset)
radar_precip_var = radar_dataset.attrs["precip_var"]
model_precip_var = model_dataset.attrs["precip_var"]

# Threshold the data
radar_dataset[radar_precip_var] = radar_dataset[radar_precip_var].where(
    radar_dataset[radar_precip_var] >= 0.1, 0.0
)
model_dataset[model_precip_var] = model_dataset[model_precip_var].where(
    model_dataset[model_precip_var] >= 0.1, 0.0
)

# Plot the radar rainfall field and the first time step of the NWP forecast.
date_str = datetime.strftime(date_radar, "%Y-%m-%d %H:%M")
plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_precip_field(
    radar_dataset[radar_precip_var].isel(time=-1),
    geodata=geodata_from_dataset(radar_dataset),
    title=f"Radar observation at {date_str}",
    colorscale="STEPS-NL",
)
plt.subplot(122)
plot_precip_field(
    model_dataset[model_precip_var].isel(ens_number=0, time=0),
    geodata=geodata_from_dataset(model_dataset),
    title=f"NWP forecast at {date_str}",
    colorscale="STEPS-NL",
)
plt.tight_layout()
plt.show()

# transform the data to dB
radar_dataset = transformation.dB_transform(radar_dataset, threshold=0.1)
model_dataset = transformation.dB_transform(model_dataset, threshold=0.1)

###############################################################################
# For the initial time step (t=0), the NWP rainfall forecast is not that different
# from the observed radar rainfall, but it misses some of the locations and
# shapes of the observed rainfall fields. Therefore, the NWP rainfall forecast will
# initially get a low weight in the blending process.
#
# Determine the velocity fields
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

oflow_method = pysteps.motion.get_method("lucaskanade")

# First for the radar images. dense_lucaskanade returns the input dataset
# with velocity_x/velocity_y data variables added.
radar_dataset = oflow_method(radar_dataset)

# Then for the NWP forecast, per model and per lead time. We need two images
# to construct a motion field, so we start from time step 1; the velocity
# field at time step 0 is set equal to that of time step 1.
n_models = model_dataset.sizes["ens_number"]
n_nwp_times = model_dataset.sizes["time"]
h = model_dataset.sizes["y"]
w = model_dataset.sizes["x"]

velocity_x = np.zeros((n_models, n_nwp_times, h, w))
velocity_y = np.zeros((n_models, n_nwp_times, h, w))
for n_model in range(n_models):
    for t in range(1, n_nwp_times):
        pair_dataset = oflow_method(
            model_dataset.isel(ens_number=n_model, time=slice(t - 1, t + 1))
        )
        velocity_x[n_model, t] = pair_dataset["velocity_x"].values
        velocity_y[n_model, t] = pair_dataset["velocity_y"].values
    velocity_x[n_model, 0] = velocity_x[n_model, 1]
    velocity_y[n_model, 0] = velocity_y[n_model, 1]

model_dataset["velocity_x"] = (["ens_number", "time", "y", "x"], velocity_x)
model_dataset["velocity_y"] = (["ens_number", "time", "y", "x"], velocity_y)


################################################################################
# The blended forecast
# ~~~~~~~~~~~~~~~~~~~~

precip_thr = radar_dataset[radar_precip_var].attrs["threshold"]
kmperpixel = abs(float(radar_dataset.x.attrs["stepsize"])) / 1000.0

precip_forecast_dataset = blending.steps.forecast(
    radar_dataset,
    model_dataset,
    timesteps=18,
    timestep=timestep,
    issuetime=date_radar,
    n_ens_members=1,
    precip_thr=precip_thr,
    kmperpixel=kmperpixel,
    noise_stddev_adj="auto",
    vel_pert_method=None,
    seed=42,  # Fixed seed for reproducible ensemble members
)

# Transform the data back into mm/h
precip_forecast_dataset = conversion.to_rainrate(precip_forecast_dataset)
forecast_precip_var = precip_forecast_dataset.attrs["precip_var"]
precip_forecast = precip_forecast_dataset[forecast_precip_var].values

model_dataset_mmh = conversion.to_rainrate(model_dataset)
nwp_precip_mmh = model_dataset_mmh[model_precip_var].values


################################################################################
# Visualize the output
# ~~~~~~~~~~~~~~~~~~~~
#
# The NWP rainfall forecast has a lower weight than the radar-based extrapolation
# forecast at the issue time of the forecast (+0 min). Therefore, the first time
# steps consist mostly of the extrapolation.
# However, near the end of the forecast (+180 min), the NWP share in the blended
# forecast has become more important and the forecast starts to resemble the
# NWP forecast more.

fig = plt.figure(figsize=(5, 12))

leadtimes_min = [30, 60, 90, 120, 150, 180]
n_leadtimes = len(leadtimes_min)
for n, leadtime in enumerate(leadtimes_min):
    # Nowcast with blending into NWP
    ax1 = plt.subplot(n_leadtimes, 2, n * 2 + 1)
    plot_precip_field(
        precip_forecast[0, int(leadtime / timestep) - 1, :, :],
        geodata=geodata_from_dataset(radar_dataset),
        title=f"Nowcast +{leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )
    ax1.axis("off")

    # Raw NWP forecast
    plt.subplot(n_leadtimes, 2, n * 2 + 2)
    ax2 = plot_precip_field(
        nwp_precip_mmh[0, int(leadtime / timestep) - 1, :, :],
        geodata=geodata_from_dataset(model_dataset),
        title=f"NWP +{leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )
    ax2.axis("off")

plt.tight_layout()
plt.show()


###############################################################################
# It is also possible to blend a deterministic or probabilistic external nowcast
# (e.g. a pre-made nowcast or a deterministic AI-based nowcast) with NWP using
# the STEPS algorithm. In that case, we add a `precip_nowcast` to
# `blending.steps.forecast`. By providing an external nowcast, the STEPS blending
# method will omit the autoregression and advection step for the extrapolation
# cascade and use the existing external nowcast instead (which will thus be
# decomposed into multiplicative cascades!). The weights determination and
# possible post-processings steps will remain the same.
#
# Start with creating an external nowcast
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# We go for a simple advection-only nowcast for the example, but this setup can
# be replaced with any external deterministic or probabilistic nowcast.
extrapolate = nowcasts.get_method("extrapolation")

# Make sure the data has no nans
radar_zerovalue = radar_dataset[radar_precip_var].attrs["zerovalue"]
radar_dataset_to_advect = radar_dataset.copy(deep=True)
radar_dataset_to_advect[radar_precip_var] = radar_dataset_to_advect[
    radar_precip_var
].where(np.isfinite(radar_dataset_to_advect[radar_precip_var]), radar_zerovalue)

# Create the extrapolation
fc_lagrangian_extrapolation_dataset = extrapolate(
    radar_dataset_to_advect.isel(time=slice(-1, None)), 18
)
fc_precip_var = fc_lagrangian_extrapolation_dataset.attrs["precip_var"]
fc_lagrangian_extrapolation = fc_lagrangian_extrapolation_dataset[fc_precip_var].values

# Insert an additional timestep at the start, as t0, which is the same as the current first slice.
fc_lagrangian_extrapolation = np.insert(
    fc_lagrangian_extrapolation, 0, fc_lagrangian_extrapolation[0:1, :, :], axis=0
)
fc_lagrangian_extrapolation[~np.isfinite(fc_lagrangian_extrapolation)] = radar_zerovalue

# Wrap the resulting (t0 + 18 lead times) array back into a dataset (reusing
# the precip_var attrs of radar_dataset, e.g. transform="dB") so that it can
# be converted to mm/h for plotting below.
fc_lagrangian_extrapolation_dataset_full = convert_output_to_xarray_dataset(
    radar_dataset_to_advect.isel(time=slice(-1, None)),
    list(range(0, fc_lagrangian_extrapolation.shape[0])),
    fc_lagrangian_extrapolation,
)


################################################################################
# Blend the external nowcast with NWP - deterministic mode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

precip_forecast_dataset = blending.steps.forecast(
    radar_dataset,
    model_dataset,
    timesteps=18,
    timestep=timestep,
    issuetime=date_radar,
    n_ens_members=1,
    precip_nowcast=np.array(
        [fc_lagrangian_extrapolation]
    ),  # Add an extra dimension, becuase precip_nowcast has to be 4-dimensional
    precip_thr=precip_thr,
    kmperpixel=kmperpixel,
    noise_stddev_adj="auto",
    vel_pert_method=None,
    nowcasting_method="external_nowcast",
    noise_method=None,
    probmatching_method=None,
    mask_method=None,
    weights_method="bps",
)

# Transform the data back into mm/h
precip_forecast_dataset = conversion.to_rainrate(precip_forecast_dataset)
forecast_precip_var = precip_forecast_dataset.attrs["precip_var"]
precip_forecast = precip_forecast_dataset[forecast_precip_var].values

fc_lagrangian_extrapolation_mmh_dataset = conversion.to_rainrate(
    fc_lagrangian_extrapolation_dataset_full
)
fc_lagrangian_extrapolation_mmh = fc_lagrangian_extrapolation_mmh_dataset[
    fc_lagrangian_extrapolation_mmh_dataset.attrs["precip_var"]
].values


################################################################################
# Visualize the output
# ~~~~~~~~~~~~~~~~~~~~
#
# The NWP rainfall forecast has a lower weight than the radar-based extrapolation
# forecast at the issue time of the forecast (+0 min). Therefore, the first time
# steps consist mostly of the extrapolation.
# However, near the end of the forecast (+180 min), the NWP share in the blended
# forecast has become more important and the forecast starts to resemble the
# NWP forecast more.

fig = plt.figure(figsize=(6, 12))

leadtimes_min = [30, 60, 90, 120, 150, 180]
n_leadtimes = len(leadtimes_min)

for n, leadtime in enumerate(leadtimes_min):
    idx = int(leadtime / timestep) - 1

    # Blended nowcast
    ax1 = plt.subplot(n_leadtimes, 3, n * 3 + 1)
    plot_precip_field(
        precip_forecast[0, idx, :, :],
        geodata=geodata_from_dataset(radar_dataset),
        title=f"Blended +{leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )
    ax1.axis("off")

    # Raw extrapolated nowcast
    ax2 = plt.subplot(n_leadtimes, 3, n * 3 + 2)
    plot_precip_field(
        fc_lagrangian_extrapolation_mmh[idx, :, :],
        geodata=geodata_from_dataset(radar_dataset),
        title=f"NWC +{leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )
    ax2.axis("off")

    # Raw NWP forecast
    plt.subplot(n_leadtimes, 3, n * 3 + 3)
    ax3 = plot_precip_field(
        nwp_precip_mmh[0, idx, :, :],
        geodata=geodata_from_dataset(model_dataset),
        title=f"NWP +{leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )
    ax3.axis("off")

plt.show()

################################################################################
# Blend the external nowcast with NWP - ensemble mode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

precip_forecast_dataset = blending.steps.forecast(
    radar_dataset,
    model_dataset,
    timesteps=18,
    timestep=timestep,
    issuetime=date_radar,
    n_ens_members=5,
    precip_nowcast=np.array(
        [fc_lagrangian_extrapolation]
    ),  # Add an extra dimension, becuase precip_nowcast has to be 4-dimensional
    precip_thr=precip_thr,
    kmperpixel=kmperpixel,
    noise_stddev_adj="auto",
    vel_pert_method=None,
    nowcasting_method="external_nowcast",
    noise_method="nonparametric",
    probmatching_method="cdf",
    mask_method="incremental",
    weights_method="bps",
    seed=42,  # Fixed seed for reproducible ensemble members
)

# Transform the data back into mm/h
precip_forecast_dataset = conversion.to_rainrate(precip_forecast_dataset)
forecast_precip_var = precip_forecast_dataset.attrs["precip_var"]
precip_forecast = precip_forecast_dataset[forecast_precip_var].values


################################################################################
# Visualize the output
# ~~~~~~~~~~~~~~~~~~~~

fig = plt.figure(figsize=(8, 12))

leadtimes_min = [30, 60, 90, 120, 150, 180]
n_leadtimes = len(leadtimes_min)

for n, leadtime in enumerate(leadtimes_min):
    idx = int(leadtime / timestep) - 1

    # Blended nowcast member 1
    ax1 = plt.subplot(n_leadtimes, 4, n * 4 + 1)
    plot_precip_field(
        precip_forecast[0, idx, :, :],
        geodata=geodata_from_dataset(radar_dataset),
        title="Blend Mem. 1",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )
    ax1.axis("off")

    # Blended nowcast member 5
    ax2 = plt.subplot(n_leadtimes, 4, n * 4 + 2)
    plot_precip_field(
        precip_forecast[4, idx, :, :],
        geodata=geodata_from_dataset(radar_dataset),
        title="Blend Mem. 5",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )
    ax2.axis("off")

    # Raw extrapolated nowcast
    ax3 = plt.subplot(n_leadtimes, 4, n * 4 + 3)
    plot_precip_field(
        fc_lagrangian_extrapolation_mmh[idx, :, :],
        geodata=geodata_from_dataset(radar_dataset),
        title=f"NWC + {leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )
    ax3.axis("off")

    # Raw NWP forecast
    ax4 = plt.subplot(n_leadtimes, 4, n * 4 + 4)
    plot_precip_field(
        nwp_precip_mmh[0, idx, :, :],
        geodata=geodata_from_dataset(model_dataset),
        title=f"NWP + {leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )
    ax4.axis("off")

plt.show()

print("Done.")


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
#
# Imhoff, R.O., L. De Cruz, W. Dewettinck, C.C. Brauer, R. Uijlenhoet, K-J. van
# Heeringen, C. Velasco-Forero, D. Nerini, M. Van Ginderachter, and A.H. Weerts.
# 2023. "Scale-dependent blending of ensemble rainfall nowcasts and NWP in the
# open-source pysteps library". Quarterly Journal of the Royal Meteorological
# Society 149(753): 1-30. https://doi.org/10.1002/qj.4461
