# -*- coding: utf-8 -*-
"""
Ensemble-based Blending
=======================

This tutorial demonstrates how to construct a blended rainfall forecast by combining
an ensemble nowcast with an ensemble Numerical Weather Prediction (NWP) forecast.
The method follows the Reduced-Space Ensemble Kalman Filter approach described in
:cite:`Nerini2019MWR`.

The procedure starts from the most recent radar observations. In the **prediction step**,
a stochastic radar extrapolation technique generates short-term forecasts. In the
**correction step**, these forecasts are updated using information from the latest
ensemble NWP run. To make the matrix operations tractable, the Bayesian update is carried
out in the subspace defined by the leading principal components—hence the term *reduced
space*.

The datasets used in this tutorial are provided by the German Weather Service (DWD).
"""

import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from pysteps_nwp_importers.importer_dwd_nwp import unstructured2regular

import pysteps
from pysteps import blending, io, rcparams
from pysteps.utils import conversion, dimension, transformation
from pysteps.visualization import plot_precip_field
from pysteps.xarray_helpers import convert_input_to_xarray_dataset, geodata_from_dataset

################################################################################
# Read the radar images and the NWP forecast
# ------------------------------------------
#
# First, we import a sequence of 4 images of 5-minute radar composites
# and the corresponding NWP rainfall forecast that was available at that time.
#
# You need the pysteps-data archive downloaded and the pystepsrc file
# configured with the data_source paths pointing to data folders.
# Additionally, the pysteps-nwp-importers plugin needs to be installed, see
# https://github.com/pySTEPS/pysteps-nwp-importers.

# Selected case
date_radar = datetime.strptime("202506041645", "%Y%m%d%H%M")
# The last NWP forecast was issued at 16:00 - the blending tool will be able
# to find the correct lead times itself.
date_nwp = datetime.strptime("202506041600", "%Y%m%d%H%M")
radar_data_source = rcparams.data_sources["dwd"]
nwp_data_source = rcparams.data_sources["dwd_nwp"]


###############################################################################
# Load the data from the archive
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

root_path = radar_data_source["root_path"]
path_fmt = radar_data_source["path_fmt"]
fn_pattern = radar_data_source["fn_pattern"]
fn_ext = radar_data_source["fn_ext"]
importer_name = radar_data_source["importer"]
importer_kwargs = radar_data_source["importer_kwargs"]
timestep_radar = radar_data_source["timestep"]

# Find the radar files in the archive
fns = io.find_by_date(
    date_radar,
    root_path,
    path_fmt,
    fn_pattern,
    fn_ext,
    timestep_radar,
    num_prev_files=2,
)

# Read the radar composites (which are already in mm/h)
importer = io.get_method(importer_name, "importer")
radar_dataset = io.read_timeseries(fns, importer, **importer_kwargs)
# The DWD importer sets attrs["transform"] = None to indicate "no transform",
# but conversion.to_rainrate/dB_transform expect the key to be absent in that
# case.
_radar_precip_var = radar_dataset.attrs["precip_var"]
if radar_dataset[_radar_precip_var].attrs.get("transform", "unset") is None:
    del radar_dataset[_radar_precip_var].attrs["transform"]

# Import the NWP data. The pysteps-nwp-importers plugin has not been migrated
# to the new xarray-based data model yet: it returns an (unstructured-grid)
# precip DataArray together with an old-style (precip, quality, metadata)
# tuple, which we reproject and wrap into an xarray dataset by hand below.
filename = os.path.join(
    nwp_data_source["root_path"],
    datetime.strftime(date_nwp, nwp_data_source["path_fmt"]),
    datetime.strftime(date_nwp, nwp_data_source["fn_pattern"])
    + "."
    + nwp_data_source["fn_ext"],
)
nwp_importer = io.get_method("dwd_nwp", "importer")
# grid_file_path in importer_kwargs is already resolvable from the repo root,
# like every other data_sources path in pystepsrc.
kwargs = dict(nwp_data_source["importer_kwargs"])
nwp_precip, _, nwp_metadata = nwp_importer(filename, **kwargs)
# We lower the number of ens members to 10 to reduce the memory needs in the
# example here. However, it is advised to have a minimum of 20 members for the
# Reduced-Space Ensemble Kalman filter approach
nwp_precip = nwp_precip[:, 0:10, :].astype("single")


################################################################################
# Pre-processing steps
# --------------------

# Set the zerovalue and precipitation thresholds (these are fixed from DWD)
prec_thr = 0.049
zerovalue = 0.027

# Transform the zerovalue and precipitation thresholds to dBR
log_thr_prec = 10.0 * np.log10(prec_thr)
log_zerovalue = 10.0 * np.log10(zerovalue)

# Reproject the DWD ICON NWP data onto a regular grid
nwp_metadata["clon"] = nwp_precip["longitude"].values
nwp_metadata["clat"] = nwp_precip["latitude"].values
# We change the time step from the DWD NWP data to 15 min (it is actually 5 min)
# to have a longer forecast horizon available for this example, as pysteps_data
# only contains 1 hour of DWD forecast data (to minimize storage).
nwp_metadata["accutime"] = 15.0
nwp_precip = (
    nwp_precip.values.astype("single") * 3.0
)  # (to account for the change in time step from 5 to 15 min)

# Reproject ICON data onto a regular grid matching the (still full-resolution)
# radar grid. unstructured2regular is part of the old (non-xarray) plugin API,
# so it takes/returns plain arrays and old-style metadata dicts.
nwp_precip_rprj, nwp_metadata_rprj = unstructured2regular(
    nwp_precip, nwp_metadata, geodata_from_dataset(radar_dataset)
)
nwp_precip = None

# convert_input_to_xarray_dataset expects a 4-D precip array with dims
# (ens_number, time, y, x); unstructured2regular returns (time, ens, y, x).
nwp_precip_rprj = nwp_precip_rprj.transpose(1, 0, 2, 3)

# The importer sets metadata["transform"] = None to indicate "no transform",
# but convert_input_to_xarray_dataset/to_rainrate expect the key to be absent
# in that case.
if nwp_metadata_rprj.get("transform") is None:
    nwp_metadata_rprj.pop("transform", None)

model_dataset = convert_input_to_xarray_dataset(
    nwp_precip_rprj.astype("single"),
    None,
    nwp_metadata_rprj,
    startdate=date_nwp,
    timestep=int(nwp_metadata_rprj["accutime"] * 60),
)

# model_dataset's reprojected grid is a few pixels smaller than radar_dataset's
# native grid (a side-effect of the unstructured-to-regular reprojection
# above). Crop both datasets to a common, cleanly divisible-by-4 grid so that
# radar_dataset and model_dataset end up with identical shapes after
# upscaling below (blending.get_method("pca_enkf") requires this).
common_ny = 4 * (min(radar_dataset.sizes["y"], model_dataset.sizes["y"]) // 4)
common_nx = 4 * (min(radar_dataset.sizes["x"], model_dataset.sizes["x"]) // 4)
radar_dataset = radar_dataset.isel(y=slice(0, common_ny), x=slice(0, common_nx))
model_dataset = model_dataset.isel(y=slice(0, common_ny), x=slice(0, common_nx))

# Upscale both the radar and NWP data to a twice as coarse resolution to lower
# the memory needs (for this example). aggregate_fields_space's meter-based
# window computation is fragile here since radar_dataset's and
# model_dataset's pixel sizes differ at floating-point precision (a
# side-effect of the independent reprojection above), so aggregate directly
# by (common) pixel count instead.
radar_dataset = dimension.aggregate_fields(
    radar_dataset, 4, dim=["y", "x"], method="mean"
)
model_dataset = dimension.aggregate_fields(
    model_dataset, 4, dim=["y", "x"], method="mean"
)

# Make sure the units are in mm/h
radar_dataset = conversion.to_rainrate(
    radar_dataset
)  # The radar data should already be in mm/h
model_dataset = conversion.to_rainrate(model_dataset)
radar_precip_var = radar_dataset.attrs["precip_var"]
model_precip_var = model_dataset.attrs["precip_var"]

# Threshold the data
radar_dataset[radar_precip_var] = radar_dataset[radar_precip_var].where(
    np.logical_or(
        radar_dataset[radar_precip_var].values >= prec_thr,
        np.isnan(radar_dataset[radar_precip_var].values),
    ),
    0.0,
)
model_dataset[model_precip_var] = model_dataset[model_precip_var].where(
    np.logical_or(
        model_dataset[model_precip_var].values >= prec_thr,
        np.isnan(model_dataset[model_precip_var].values),
    ),
    0.0,
)

# Plot the radar rainfall field and the first time step and first ensemble member
# of the NWP forecast.
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
radar_dataset = transformation.dB_transform(
    radar_dataset, threshold=prec_thr, zerovalue=log_zerovalue
)
model_dataset = transformation.dB_transform(
    model_dataset, threshold=prec_thr, zerovalue=log_zerovalue
)


###############################################################################
# Determine the velocity fields
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In contrast to the STEPS blending method, no motion field for the NWP fields
# is needed in the ensemble kalman filter blending approach.

# Estimate the motion vector field. dense_lucaskanade returns the input
# dataset with velocity_x/velocity_y data variables added.
oflow_method = pysteps.motion.get_method("lucaskanade")
radar_dataset = oflow_method(radar_dataset)


################################################################################
# The blended forecast
# ~~~~~~~~~~~~~~~~~~~~

# Set the combination kwargs
combination_kwargs = dict(
    n_tapering=0,  # Tapering parameter: controls how many diagonals of the covariance matrix are kept (0 = no tapering)
    non_precip_mask=True,  # Specifies whether the computation should be truncated on grid boxes where at least a minimum number of ens. members forecast precipitation.
    n_ens_prec=1,  # Minimum number of ens. members that forecast precip for the above-mentioned mask.
    lien_criterion=True,  # Specifies wheter the Lien criterion should be applied.
    n_lien=5,  # Minimum number of ensemble members that forecast precipitation for the Lien criterion (equals half the ens. members here)
    prob_matching="iterative",  # The type of probability matching used.
    inflation_factor_bg=3.0,  # Inflation factor of the background (NWC) covariance matrix. (this value indicates a faster convergence towards the NWP ensemble)
    inflation_factor_obs=1.0,  # Inflation factor of the observation (NWP) covariance matrix.
    offset_bg=0.0,  # Offset of the background (NWC) covariance matrix.
    offset_obs=0.0,  # Offset of the observation (NWP) covariance matrix.
    nwp_hres_eff=14.0,  # Effective horizontal resolution of the utilized NWP model (in km here).
    sampling_prob_source="ensemble",  # Computation method of the sampling probability for the probability matching. 'ensemble' computes this probability as the ratio between the ensemble differences.
    use_accum_sampling_prob=False,  # Specifies whether the current sampling probability should be used for the probability matching or a probability integrated over the previous forecast time.
)


# Call the PCA EnKF method
blending_method = blending.get_method("pca_enkf")
precip_forecast_dataset = blending_method(
    radar_dataset,  # Radar dataset in dBR, with velocity_x/velocity_y embedded
    model_dataset,  # NWP ensemble dataset in dBR
    forecast_horizon=120,  # Forecast length (horizon) in minutes - only a short forecast horizon due to the limited dataset length stored here.
    issuetime=date_radar,  # Forecast issue time as datetime object
    n_ens_members=10,  # No. of ensemble members
    precip_mask_dilation=1,  # Dilation of precipitation mask in grid boxes
    n_cascade_levels=6,  # No. of cascade levels
    precip_thr=log_thr_prec,  # Precip threshold
    norain_thr=0.0005,  # Minimum of 0.5% precip needed, otherwise 'zero rainfall'
    # No. of parallel threads. Kept at 1 so that the forecast (and the plots
    # in this example) are exactly reproducible: multi-threaded FFT/dask
    # reductions are not guaranteed to be bit-for-bit deterministic.
    num_workers=1,
    noise_stddev_adj="auto",  # Standard deviation adjustment
    noise_method="ssft",  # SSFT as noise method
    enable_combination=True,  # Enable combination
    noise_kwargs={"win_size": (512, 512), "win_fun": "hann", "overlap": 0.5},
    extrap_kwargs={"interp_order": 3, "map_coordinates_mode": "nearest"},
    combination_kwargs=combination_kwargs,
    filter_kwargs={"include_mean": True},
    seed=42,  # Fixed seed for reproducible ensemble members
)

# Transform the data back into mm/h
precip_forecast_dataset = conversion.to_rainrate(precip_forecast_dataset)
forecast_precip_var = precip_forecast_dataset.attrs["precip_var"]
model_dataset_mmh = conversion.to_rainrate(model_dataset)


################################################################################
# Visualize the output
# ~~~~~~~~~~~~~~~~~~~~
#
# The NWP rainfall forecast has a much lower weight than the radar-based
# extrapolation # forecast at the issue time of the forecast (+0 min). Therefore,
# the first time steps consist mostly of the extrapolation. However, near the end
# of the forecast (+180 min), the NWP share in the blended forecast has become
# the more dominant contribution to the forecast and thus the forecast starts
# to resemble the NWP forecast.

fig = plt.figure(figsize=(5, 12))

leadtimes_min = [15, 30, 45, 60, 90, 120]
n_leadtimes = len(leadtimes_min)
for n, leadtime in enumerate(leadtimes_min):
    # Nowcast with blending into NWP
    plt.subplot(n_leadtimes, 2, n * 2 + 1)
    plot_precip_field(
        precip_forecast_dataset[forecast_precip_var].isel(
            ens_number=0, time=int(leadtime / timestep_radar) - 1
        ),
        geodata=geodata_from_dataset(radar_dataset),
        title=f"Blended +{leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )

    # Raw NWP forecast
    plt.subplot(n_leadtimes, 2, n * 2 + 2)
    plot_precip_field(
        model_dataset_mmh[model_precip_var].isel(
            ens_number=0,
            time=int(leadtime / int(model_dataset["time"].attrs["stepsize"] / 60)) - 1,
        ),
        geodata=geodata_from_dataset(model_dataset),
        title=f"NWP +{leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )

plt.show()

################################################################################
# References
# ~~~~~~~~~~
#

# Nerini, D., Foresti, L., Leuenberger, D., Robert, S., Germann, U. 2019. "A
# Reduced-Space Ensemble Kalman Filter Approach for Flow-Dependent Integration
# of Radar Extrapolation Nowcasts and NWP Precipitation Ensembles." Monthly
# Weather Review 147(3): 987-1006. https://doi.org/10.1175/MWR-D-18-0258.1.
