"""
MLCast datasets for nowcasting with pysteps
===========================================

This tutorial demonstrates how to use MLCast datasets for rainfall
nowcasting with the pysteps methods.

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from pysteps import motion, nowcasts
from pysteps.utils import transformation
from pysteps.visualization import get_colormap, plot_precip_field
from pysteps.xarray_helpers import geodata_from_dataset

################################################################################
# Load the RMI dataset from S3
# ----------------------------------------
#
# We will use the RMI dataset for a specific event day and extract a
# subset of the data for nowcasting.

# Load dataset with xarray
ds = xr.open_dataset(
    "s3://mlcast-source-datasets/radar_precipitation/BE-RMI-radclim-mfb/v0.1.1/5_minutes.zarr/",
    engine="zarr",
    storage_options={
        "anon": True,
        "client_kwargs": {"endpoint_url": "https://object-store.os-api.cci2.ecmwf.int"},
    },
)

# Pick a day
EVENT_DAY = "2021-06-19"
T0 = "2021-06-19T20:15"
N_INPUT, N_LEADTIMES = 12, 12

# Slice an 256 x 256 km box from the full field
ymin, ymax, xmin, xmax = 243, 499, 257, 513

# Work on a 256 x 256 km box centred on the site around T0
window = (
    ds.isel(y=slice(ymin, ymax), x=slice(xmin, xmax))
    .sel(time=slice(f"{EVENT_DAY}T17:00", f"{EVENT_DAY}T21:30"))
    .load()
)
T0_INDEX = int(np.argwhere(window.time.values == np.datetime64(T0))[0, 0])
window

################################################################################
# Add metadata required by pysteps
# --------------------------------
#
# Name of the data_var containing the precipitation
precip_var = "rain_rate"
window.attrs["precip_var"] = precip_var
# Data timestep in seconds
window["time"].attrs["stepsize"] = 300

# x, y intervals and units
window["x"].attrs["stepsize"] = window.x.values[1] - window.x.values[0]
window["y"].attrs["stepsize"] = window.y.values[1] - window.y.values[0]
window["x"].attrs["units"] = "m"
window["y"].attrs["units"] = "m"

# Grid projection
window.attrs["projection"] = window[window[precip_var].grid_mapping].proj4

window


################################################################################
# Pick intervals for input dataset
# --------------------------------
#
# Create input and observation datasets
input_ds = window.isel(time=slice(T0_INDEX - N_INPUT + 1, T0_INDEX + 1))
obs_ds = window.isel(time=slice(T0_INDEX + 1, T0_INDEX + 1 + N_LEADTIMES))

# Transform rain rate to dBR
input_ds_dBR = transformation.dB_transform(input_ds, threshold=-10.0)
input_ds_dBR = input_ds_dBR.where(np.isfinite(input_ds_dBR), -10.0)
input_ds_dBR


################################################################################
# Estimate motion field
# ---------------------
#
oflow_method = motion.get_method("LK")
input_ds_dBR = oflow_method(input_ds_dBR.isel(time=slice(-3, -1)))
input_ds_dBR


################################################################################
# Create extrapolation nowcast
# ----------------------------
#
# Extrapolate the last radar observation
extrapolate = nowcasts.get_method("extrapolation")

ds_forecast_dBR = extrapolate(input_ds_dBR, N_LEADTIMES)
ds_forecast_dBR

################################################################################
# Transform forecast back to rain rate
# ------------------------------------
#
ds_forecast = transformation.dB_transform(
    ds_forecast_dBR, threshold=-10.0, inverse=True
)
ds_forecast = ds_forecast.where(ds_forecast[precip_var] > 0.0)

################################################################################
# Plot the nowcast
# ----------------
#
fig = plt.figure(figsize=(20, 6))
cmap, norm, clevs, clevs_str = get_colormap("intensity", colorscale="STEPS-NL")

# Get geographical metadata required by the plotting functions
geodata = geodata_from_dataset(ds_forecast)

for i in range(N_LEADTIMES):
    ax = plt.subplot(2, 6, i + 1)

    # Plot each field
    plot_precip_field(
        ds_forecast[precip_var].isel(time=i),
        ax=ax,
        geodata=geodata,
        # Actual units in data is kg m^-2 h^-1, but
        # pysteps doesn't accept this
        units="mm/h",
        colorscale="STEPS-NL",
        colorbar=False,
    )

    ax.set_title(f"Lead time {(i+1)*5} min")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.label_outer()
    ax.set_yticklabels([])
    ax.set_xticklabels([])

# Add a colorbar for the entire figure
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ticks=clevs,
    ax=fig.axes,
    orientation="vertical",
    extend="max",
    fraction=0.02,
    pad=0.01,
)
cbar.set_label("Rain rate (mm/h)")
fig.savefig("nowcast.png", dpi=300, bbox_inches="tight")
# plt.show()
