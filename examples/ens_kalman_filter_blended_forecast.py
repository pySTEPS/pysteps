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
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

import pysteps
from pysteps import io, rcparams, blending
from pysteps.utils import conversion, aggregate_fields_space
from pysteps.visualization import plot_precip_field
from pysteps.xarray_helpers import convert_input_to_xarray_dataset
import pysteps_nwp_importers

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
date_radar = datetime.strptime("202506041700", "%Y%m%d%H%M")
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
# radar_precip, _, radar_metadata = io.read_timeseries(fns, importer, **importer_kwargs)
radar_ds = io.read_timeseries(fns, importer, **importer_kwargs)
radar_ds = conversion.to_rainrate(radar_ds)

# Extract precipitation variable. In this case it is equal to "precip_intensity"
precip_var = radar_ds.attrs["precip_var"]
radar_precip = radar_ds[precip_var]

radar_metadata = {
    "projection": radar_ds.attrs["projection"],
    "x1": np.round(radar_ds.x.values[0]),
    "x2": np.round(radar_ds.x.values[-1] + 1000.0),
    "y1": np.round(radar_ds.y.values[0]),
    "y2": np.round(radar_ds.y.values[-1] - 1000.0),
    "xpixelsize": np.unique(np.round(np.diff(radar_ds.x.values)))[0],
    "ypixelsize": np.unique(np.round(np.diff(radar_ds.y.values)))[0],
    "cartesian_unit": "m",
    "yorigin": "lower",  # is this always the case using xarray approach?
}

# Import the NWP data
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
# nwp_precip = nwp_precip[:, 0:10, :].astype("single")
# nwp_precip = nwp_precip[:, :, :].astype("single")


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
nwp_metadata["accutime"] = 5.0
# nwp_precip = (
#    nwp_precip.values.astype("single") * 3.0
# )  # (to account for the change in time step from 5 to 15 min)
nwp_precip = nwp_precip.values.astype("single")

# Reproject ID2 data onto a regular grid
nwp_precip_rprj, nwp_metadata_rprj = (
    pysteps_nwp_importers.importer_dwd_nwp.unstructured2regular(
        nwp_precip, nwp_metadata, radar_metadata
    )
)

nwp_precip = None
nwp_precip_ds = convert_input_to_xarray_dataset(
    precip=nwp_precip_rprj,
    quality=None,
    metadata=nwp_metadata_rprj,
    startdate=datetime(2025, 6, 4, 17, 0),
    timestep=300,
)
nwp_precip_ds = conversion.to_rainrate(nwp_precip_ds)
precip_var_nwp = nwp_precip_ds.attrs["precip_var"]
nwp_precip_rprj = nwp_precip_ds[precip_var_nwp]

# Upscale both the radar and NWP data to a twice as coarse resolution to lower
# the memory needs (for this example)
# radar_precip, radar_metadata = aggregate_fields_space(
#    radar_precip, radar_metadata, radar_metadata["xpixelsize"] * 4
# )
# nwp_precip_rprj, nwp_metadata_rprj = aggregate_fields_space(
#    nwp_precip_rprj.astype("single"),
#    nwp_metadata_rprj,
#    nwp_metadata_rprj["xpixelsize"] * 4,
# )

# Make sure the units are in mm/h
# Convert to rain rate

converter = pysteps.utils.get_method("mm/h")
radar_ds = converter(radar_ds)  # The radar data should already be in mm/h
nwp_precip_ds = converter(nwp_precip_ds)

# Threshold the data
radar_precip.data[radar_precip.data < prec_thr] = 0.0
nwp_precip_rprj.data[nwp_precip_rprj.data < prec_thr] = 0.0


# Plot the radar rainfall field and the first time step and first ensemble member
# of the NWP forecast.
# date_str = datetime.strftime(date_radar, "%Y-%m-%d %H:%M")
# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plot_precip_field(
#     radar_precip[-1, :, :],
#     geodata=radar_metadata,
#     title=f"Radar observation at {date_str}",
#     colorscale="STEPS-NL",
# )
# plt.subplot(122)
# plot_precip_field(
#     nwp_precip_rprj[0, 0, :, :],
#     geodata=nwp_metadata_rprj,
#     title=f"NWP forecast at {date_str}",
#     colorscale="STEPS-NL",
# )
# plt.tight_layout()
# plt.show()

# transform the data to dB
transformer = pysteps.utils.get_method("dB")
radar_ds = transformer(radar_ds, threshold=prec_thr, zerovalue=log_zerovalue)
nwp_precip_ds = transformer(nwp_precip_ds, threshold=prec_thr, zerovalue=log_zerovalue)
# radar_precip, radar_metadata = transformer(
#    radar_precip, radar_metadata, threshold=prec_thr, zerovalue=log_zerovalue
# )
# nwp_precip_rprj, nwp_metadata_rprj = transformer(
#    nwp_precip_rprj, nwp_metadata_rprj, threshold=prec_thr, zerovalue=log_zerovalue
# )


###############################################################################
# Determine the velocity fields
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In contrast to the STEPS blending method, no motion field for the NWP fields
# is needed in the ensemble kalman filter blending approach.

# Estimate the motion vector field
oflow_method = pysteps.motion.get_method("lucaskanade")
radar_ds = oflow_method(radar_ds)


################################################################################
# The blended forecast
# ~~~~~~~~~~~~~~~~~~~~

# Set the combination kwargs for the masked enkf
combination_kwargs_masked = dict(
    # Tapering parameter: controls how many diagonals of the error covariance matrices
    # are kept (0 = no tapering)
    n_tapering=0,
    # Specifies whether the computation should be truncated on grid boxes where at
    # least a minimum number of ensemble members forecast precipitation.
    non_precip_mask=True,
    # Minimum number of ensemble members that forecast precipitation for the
    # above-mentioned mask.
    n_ens_prec=1,
    # Specifies whether the Lien criterion should be applied.
    lien_criterion=True,
    # Minimum number of ensemble members that forecast precipitation for the Lien
    # criterion (equals half the ensemble members here)
    n_lien=10,
    # The type of probability matching used.
    prob_matching="iterative",
    # Inflation factor of the background (NWC) covariance matrix. (This value indicates
    # a faster convergence towards the NWP ensemble.)
    inflation_factor_bg=3.0,
    # Inflation factor of the observation (NWP) covariance matrix.
    inflation_factor_obs=1.0,
    # Offset of the background (NWC) covariance matrix.
    offset_bg=0.0,
    # Offset of the observation (NWP) covariance matrix.
    offset_obs=0.0,
    # Effective horizontal resolution of the utilized NWP model (in km here).
    nwp_hres_eff=14.0,
    # Computation method of the sampling probability for the probability matching.
    # 'ensemble' computes this probability as the ratio between the ensemble
    # differences.
    sampling_prob_source="ensemble",
    # Specifies whether the current sampling probability should be used for the
    # probability matching or a probability integrated over the previous forecast time.
    use_accum_sampling_prob=False,
    # Specifies whether the end of the combination should represent the pure NWP
    # forecast.
    ensure_full_nwp_weight=True,
)

# Set the combination kwargs for the distances enkf
combination_kwargs_distances = dict(
    # Tapering parameter: controls how many diagonals of the error covariance matrices
    # are kept (0 = no tapering)
    n_tapering=0,
    # Specifies whether the computation should be truncated on grid boxes where at
    # least a minimum number of ensemble members forecast precipitation.
    non_precip_mask=True,
    # Minimum number of ensemble members that forecast precipitation for the
    # above-mentioned mask.
    n_ens_prec=1,
    # Specifies whether the Lien criterion should be applied.
    lien_criterion=False,
    # The type of probability matching used.
    prob_matching="none",
    # Inflation factor of the background (NWC) covariance matrix. (This value indicates
    # a faster convergence towards the NWP ensemble.)
    inflation_factor_bg=0.3,
    # Inflation factor of the observation (NWP) covariance matrix.
    inflation_factor_obs=1.0,
    # Offset of the background (NWC) covariance matrix.
    offset_bg=0.0,
    # Offset of the observation (NWP) covariance matrix.
    offset_obs=0.0,
    # Effective horizontal resolution of the utilized NWP model (in km here).
    nwp_hres_eff=0.0,
    # Computation method of the sampling probability for the probability matching.
    # 'ensemble' computes this probability as the ratio between the ensemble
    # differences.
    sampling_prob_source="explained_var",
    # Specifies whether the current sampling probability should be used for the
    # probability matching or a probability integrated over the previous forecast time.
    use_accum_sampling_prob=True,
    # Specifies whether the end of the combination should be represent the pure NWP
    # forecast.
    ensure_full_nwp_weight=True,
    # Specifies whether dry area transformation within distances enkf should be
    # additionally backed by transforming precipitation values back into linear space.
    use_linear_space=True,
    # Specifies whether the ensemble variance should be additionally adjusted in
    # regions of local precipitation maxima.
    local_maxima_adjustment=True,
    # Minimum ratio by what ensemble variance is adjusted if local maxima adjustment is
    # used.
    local_maxima_residual=0.02,
    # Specifies whether that adjustment should only take place in already existing
    # precipitation features or whether it should be used also in regions in that
    # there's only NWP forecast precipitation.
    use_only_exist_prec=False,
)

# Call the PCA EnKF method
blending_method = blending.get_method("pca_enkf")
precip_forecast = blending_method(
    radar_dataset=radar_ds,  # Radar data in dBR
    model_dataset=nwp_precip_ds,  # NWP in dBR
    forecast_horizon=60,  # Forecast length (horizon) in minutes - only a short forecast horizon due to the limited dataset length stored here.
    issuetime=date_radar,  # Forecast issue time as datetime object
    n_ens_members=20,  # No. of ensemble members
    precip_mask_dilation=1,  # Dilation of precipitation mask in grid boxes
    n_cascade_levels=12,  # No. of cascade levels
    precip_thr=log_thr_prec,  # Precip threshold
    norain_thr=0.005,  # Minimum of 0.5% precip needed, otherwise 'zero rainfall'
    # No. of parallel threads. Kept at 1 so that the forecast (and the plots
    # in this example) are exactly reproducible: multi-threaded FFT/dask
    # reductions are not guaranteed to be bit-for-bit deterministic.
    num_workers=30,
    noise_stddev_adj="auto",  # Standard deviation adjustment
    noise_method="ssft",  # SSFT as noise method
    enkf_method="distances_enkf",
    enable_combination=True,  # Enable combination
    noise_kwargs={"win_size": (512, 512), "win_fun": "hann", "overlap": 0.5},
    extrap_kwargs={"interp_order": 3, "map_coordinates_mode": "nearest"},
    combination_kwargs=combination_kwargs_distances,
    filter_kwargs={"include_mean": True},
    seed=42,  # Fixed seed for reproducible ensemble members
)

# Transform the data back into mm/h
precip_forecast = converter(precip_forecast, radar_metadata)
radar_ds = converter(radar_ds, radar_metadata)
nwp_precip_ds = converter(nwp_precip_ds, nwp_metadata_rprj)

pv = precip_forecast.attrs["precip_var"]
pf = precip_forecast[pv]

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

fig = plt.figure(figsize=(16, 28))

# leadtimes_min = [15, 30, 45, 60, 90, 120]
leadtimes_min = [0, 10, 20, 30, 45, 60]
# leadtimes_min = [0, 5, 10, 15]
n_leadtimes = len(leadtimes_min)
for n, leadtime in enumerate(leadtimes_min):
    idx_obs = 0 if leadtime == 0 else int(leadtime / timestep_radar) - 1
    idx_nwp = (
        0 if leadtime == 0 else int(leadtime / int(nwp_metadata_rprj["accutime"])) - 1
    )
    # Nowcast with blending into NWP
    plt.subplot(n_leadtimes, 2, n * 2 + 1)
    plot_precip_field(
        pf[0, idx_obs, :, :],
        geodata=radar_metadata,
        title=f"Blended +{leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )

    # Raw NWP forecast
    plt.subplot(n_leadtimes, 2, n * 2 + 2)
    plot_precip_field(
        nwp_precip_rprj[0, idx_nwp, :, :],
        geodata=nwp_metadata_rprj,
        title=f"NWP +{leadtime} min",
        axis="off",
        colorscale="STEPS-NL",
        colorbar=False,
    )
plt.savefig("intense_test.png", dpi=120)

################################################################################
# References
# ~~~~~~~~~~
#

# Nerini, D., Foresti, L., Leuenberger, D., Robert, S., Germann, U. 2019. "A
# Reduced-Space Ensemble Kalman Filter Approach for Flow-Dependent Integration
# of Radar Extrapolation Nowcasts and NWP Precipitation Ensembles." Monthly
# Weather Review 147(3): 987-1006. https://doi.org/10.1175/MWR-D-18-0258.1.
