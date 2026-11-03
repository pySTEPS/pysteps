"""
MLCast datasets for nowcasting with pysteps
===========================================

This tutorial demonstrates how to use MLCast datasets for rainfall
nowcasting with the pysteps methods.

"""

import numpy as np
import matplotlib.pyplot as plt

from pysteps import motion, nowcasts
from pysteps.utils import transformation
from pysteps.visualization import get_colormap

import mlcast_datasets

################################################################################
# Load the RMI dataset from MLCast catalog
# ----------------------------------------
#
# We will use the RMI dataset for a specific event day and extract a
# subset of the data for nowcasting.

# Load dataset from catalog
cat = mlcast_datasets.open_catalog()
ds = cat.precipitation.be_rmi_radclim_mfb_5min.to_dask()

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
window.attrs["precip_var"] = "rain_rate"
# Data timestep in seconds
window["time"].attrs["stepsize"] = 300


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
ds_forecast = ds_forecast.where(ds_forecast["rain_rate"] > 0.0)

################################################################################
# Plot the nowcast
# ----------------
#

fig, axs = plt.subplots(2, 6, figsize=(20, 6))

cmap, norm, clevs, clevs_str = get_colormap("intensity", colorscale="STEPS-NL")

for i in range(N_LEADTIMES):
    ax = axs.flatten()[i]
    ds_forecast.isel(time=i)["rain_rate"].plot(
        ax=ax, cmap=cmap, norm=norm, add_colorbar=False
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
    ax=axs,
    orientation="vertical",
    extend="max",
    fraction=0.02,
    pad=0.01,
)
cbar.set_label("Rain rate (mm/h)")
plt.show()
