# -*- coding: utf-8 -*-

"""
Linear blending
===============

This tutorial shows how to construct a simple linear blending between a STEPS
ensemble nowcast and a Numerical Weather Prediction (NWP) rainfall forecast. The
used datasets are from the Bureau of Meteorology, Australia.
"""

import os
from datetime import datetime

from matplotlib import pyplot as plt

import pysteps
from pysteps import io, rcparams, nowcasts, blending
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field
from pysteps.xarray_helpers import convert_input_to_xarray_dataset

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


def geodata_from_dataset(dataset):
    """Build a plot_precip_field/quiver-style geodata dict from a dataset."""
    x = dataset.x.values
    y = dataset.y.values
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    y1 = y[0] - dy / 2.0
    y2 = y[-1] + dy / 2.0
    return {
        "projection": dataset.attrs["projection"],
        "x1": x[0] - dx / 2.0,
        "x2": x[-1] + dx / 2.0,
        "y1": min(y1, y2),
        "y2": max(y1, y2),
        "yorigin": "lower" if dy > 0 else "upper",
    }


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
# End of the forecast is 18 time steps (+3 hours) in advance.
nwp_precip = nwp_precip[24:43, :, :]

# The importer sets metadata["transform"] = None to indicate "no transform",
# but convert_input_to_xarray_dataset/to_rainrate expect the key to be absent
# in that case.
nwp_metadata = dict(nwp_metadata)
if nwp_metadata.get("transform") is None:
    nwp_metadata.pop("transform", None)

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
# For the initial time step (t=0), the NWP rainfall forecast is not that different
# from the observed radar rainfall, but it misses some of the locations and
# shapes of the observed rainfall fields. Therefore, the NWP rainfall forecast will
# initially get a low weight in the blending process.
date_str = datetime.strftime(date_radar, "%Y-%m-%d %H:%M")
plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_precip_field(
    radar_dataset[radar_precip_var].isel(time=-1),
    geodata=geodata_from_dataset(radar_dataset),
    title=f"Radar observation at {date_str}",
)
plt.subplot(122)
plot_precip_field(
    model_dataset[model_precip_var].isel(time=0),
    geodata=geodata_from_dataset(model_dataset),
    title=f"NWP forecast at {date_str}",
)
plt.tight_layout()
plt.show()

# Only keep the NWP forecasts from 2020-10-31 04:05 onwards, because the first
# forecast lead time starts at 04:05.
model_dataset = model_dataset.isel(time=slice(1, None))

# Transform the radar data to dB - this transformation is useful for the motion
# field estimation and the subsequent nowcasts. The NWP forecast is not
# transformed, because the linear blending code sets everything back in mm/h
# after the nowcast.
radar_dataset = transformation.dB_transform(radar_dataset, threshold=0.1)


################################################################################
# Determine the velocity field for the radar rainfall nowcast
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

oflow_method = pysteps.motion.get_method("lucaskanade")
radar_dataset = oflow_method(radar_dataset)


################################################################################
# The linear blending of nowcast and NWP rainfall forecast
# --------------------------------------------------------

# Calculate the blended precipitation field
precip_blended_dataset = blending.linear_blending.forecast(
    radar_dataset,
    timesteps=18,
    timestep=10,
    nowcast_method="extrapolation",  # simple advection nowcast
    model_dataset=model_dataset,
    start_blending=60,  # in minutes (this is an arbritrary choice)
    end_blending=120,  # in minutes (this is an arbritrary choice)
)


################################################################################
# The salient blending of nowcast and NWP rainfall forecast
# ---------------------------------------------------------
#
# This method follows the saliency-based blending procedure described in :cite:`Hwang2015`. The
# blending is based on intensities and forecast times. The blended product preserves pixel
# intensities with time if they are strong enough based on their ranked salience. Saliency is
# the property of an object to be outstanding with respect to its surroundings. The ranked salience
# is calculated by first determining the difference in the normalized intensity of the nowcasts
# and NWP. Next, the pixel intensities are ranked, in which equally comparable values receive
# the same ranking number.

# Calculate the salient blended precipitation field
precip_salient_blended_dataset = blending.linear_blending.forecast(
    radar_dataset,
    timesteps=18,
    timestep=10,
    nowcast_method="extrapolation",  # simple advection nowcast
    model_dataset=model_dataset,
    start_blending=60,  # in minutes (this is an arbritrary choice)
    end_blending=120,  # in minutes (this is an arbritrary choice)
    saliency=True,
)


################################################################################
# Visualize the output
# --------------------

################################################################################
# Calculate the radar rainfall nowcasts for visualization

nowcast_method_func = nowcasts.get_method("extrapolation")
precip_nowcast_dataset = nowcast_method_func(radar_dataset, timesteps=18)

# Make sure that precip_nowcast are in mm/h
precip_nowcast_dataset = conversion.to_rainrate(precip_nowcast_dataset)
nowcast_precip_var = precip_nowcast_dataset.attrs["precip_var"]

precip_nowcast = precip_nowcast_dataset[nowcast_precip_var].values
precip_blended = precip_blended_dataset[
    precip_blended_dataset.attrs["precip_var"]
].values
precip_salient_blended = precip_salient_blended_dataset[
    precip_salient_blended_dataset.attrs["precip_var"]
].values
precip_nwp = model_dataset[model_precip_var].values
nwp_geodata = geodata_from_dataset(model_dataset)
radar_geodata = geodata_from_dataset(radar_dataset)

################################################################################
# The linear blending starts at 60 min, so during the first 60 minutes the
# blended forecast only consists of the extrapolation forecast (consisting of an
# extrapolation nowcast). Between 60 and 120 min, the NWP forecast gradually gets more
# weight, whereas the extrapolation forecasts gradually gets less weight. In addition,
# the saliency-based blending takes also the difference in pixel intensities into account,
# which are preserved over time if they are strong enough based on their ranked salience.
# Furthermore, pixels with relative low intensities get a lower weight and stay smaller in
# the saliency-based blending compared to linear blending. After 120 min, the blended
# forecast entirely consists of the NWP rainfall forecast.

fig = plt.figure(figsize=(8, 12))

leadtimes_min = [30, 60, 80, 100, 120]
n_leadtimes = len(leadtimes_min)
for n, leadtime in enumerate(leadtimes_min):
    # Extrapolation
    plt.subplot(n_leadtimes, 4, n * 4 + 1)
    plot_precip_field(
        precip_nowcast[int(leadtime / timestep) - 1, :, :],
        geodata=radar_geodata,
        title=f"Nowcast + {leadtime} min",
        axis="off",
        colorbar=False,
    )

    # Nowcast with blending into NWP
    plt.subplot(n_leadtimes, 4, n * 4 + 2)
    plot_precip_field(
        precip_blended[int(leadtime / timestep) - 1, :, :],
        geodata=radar_geodata,
        title=f"Linear + {leadtime} min",
        axis="off",
        colorbar=False,
    )

    # Nowcast with salient blending into NWP
    plt.subplot(n_leadtimes, 4, n * 4 + 3)
    plot_precip_field(
        precip_salient_blended[int(leadtime / timestep) - 1, :, :],
        geodata=radar_geodata,
        title=f"Salient + {leadtime} min",
        axis="off",
        colorbar=False,
    )

    # Raw NWP forecast
    plt.subplot(n_leadtimes, 4, n * 4 + 4)
    plot_precip_field(
        precip_nwp[int(leadtime / timestep) - 1, :, :],
        geodata=nwp_geodata,
        title=f"NWP + {leadtime} min",
        axis="off",
        colorbar=False,
    )

plt.tight_layout()
plt.show()

################################################################################
# Note that the NaN values of the extrapolation forecast are replaced with NWP data
# in the blended forecast, even before the blending starts.
