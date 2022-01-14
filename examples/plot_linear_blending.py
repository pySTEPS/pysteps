# -*- coding: utf-8 -*-

"""
Plot linear blending
====================

This tutorial shows how to construct a simple linear blending between an ensemble 
nowcast and a dummy Numerical Weather Prediction (NWP) rainfall forecast.
"""

from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint
from datetime import datetime
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps import io, rcparams
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field
from pysteps.utils import dimension
from pysteps.blending.linear_blending import forecast


def gaussian(x, max_value, mean, sigma):
    return max_value * np.exp(-(x - mean) * (x - mean) / sigma / sigma / 2)


def dummy_nwp(precip, n_leadtimes, max_value=20, mean=0, sigma=0.25, speed=100):
    """Generates dummy NWP data with the same dimension as the input
    precipitation field precip. The NWP data is a vertical line with a Gaussian
    profile moving to the left"""

    # precip is original radar image
    rows = precip.shape[0]
    cols = precip.shape[1]

    # Initialise the dummy NWP data
    precip_nwp = np.zeros((n_leadtimes, rows, cols))
    x = np.linspace(-5, 5, cols)

    for n in range(n_leadtimes):
        for i in range(rows):
            precip_nwp[n, i, :] = gaussian(x, max_value, mean, sigma)
        mean -= speed / rows

    return precip_nwp


###############################################################################
# Set nowcast parameters
n_ens_members = 10
n_leadtimes = 18
start_blending = 30  # in minutes
end_blending = 60  # in minutes
seed = 24

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
precip = io.read_timeseries(fns, importer, **importer_kwargs)

# Get the metadata
metadata = precip.x.attrs.copy()
metadata.update(**precip.y.attrs)
metadata.update(**precip.attrs)

# Convert to rain rate
precip, metadata = conversion.to_rainrate(precip, metadata)

# Upscale data to 2 km to limit memory usage
precip, metadata = dimension.aggregate_fields_space(precip, metadata, 2000)

# Import the dummy NWP data (vertical line moving to the left)
precip_nwp = dummy_nwp(
    precip[-1, :, :], n_leadtimes + 1, max_value=7, mean=4, speed=0.2 * 350
)
metadata_nwp = metadata.copy()

# Plot the radar rainfall field and the first time step of the dummy NWP forecast.
date_str = datetime.strftime(date, "%Y-%m-%d %H:%M")
plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_precip_field(
    precip[-1, :, :], geodata=metadata, title=f"Radar observation at {date_str}"
)
plt.subplot(122)
plot_precip_field(
    precip_nwp[0, :, :], geodata=metadata_nwp, title=f"Dummy NWP forecast at {date_str}"
)
plt.tight_layout()
plt.show()

# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
# set the fill value to -15 dBR
precip, metadata = transformation.dB_transform(
    precip, metadata, threshold=0.1, zerovalue=-15.0
)

# Nicely print the metadata
pprint(metadata)

###############################################################################
# Linear blending of nowcast and NWP data

# Estimate the motion field
velocity = dense_lucaskanade(precip)

# Define nowcast keyword arguments
nowcast_kwargs = {
    "n_ens_members": n_ens_members,
    "n_cascade_levels": 6,
    "R_thr": -10.0,
    "kmperpixel": 2.0,
    "timestep": timestep,
    "noise_method": "nonparametric",
    "vel_pert_method": "bps",
    "mask_method": "incremental",
}

# Calculate the blended precipitation field
precip_blended = forecast(
    precip=precip[-3:, :, :],
    precip_metadata=metadata,
    velocity=velocity,
    timesteps=n_leadtimes,
    timestep=timestep,
    nowcast_method="steps",
    precip_nwp=precip_nwp[1:, :, :],
    precip_nwp_metadata=metadata_nwp,
    start_blending=start_blending,
    end_blending=end_blending,
    nowcast_kwargs=nowcast_kwargs,
)

"""
extrap_kwargs = {"allow_nonfinite_values": True}
nowcast_kwargs = {"extrap_kwargs": extrap_kwargs}

# Calculate the blended precipitation field
precip_blended = forecast(
    precip[-1, :, :],
    V,
    n_leadtimes,
    timestep,
    "extrapolation",
    precip_nwp=precip_nwp[1: :, :],
    start_blending=start_blending,
    end_blending=end_blending,
    nowcast_kwargs=nowcast_kwargs,
)
"""

# Calculate the ensemble average
if len(precip_blended.shape) == 4:
    precip_blended_mean = np.mean(precip_blended[:, :, :, :], axis=0)
else:
    precip_blended_mean = np.copy(precip_blended)

# Plot the blended field
for i in range(0, n_leadtimes, 3):
    plot_precip_field(
        precip_blended_mean[i, :, :],
        geodata=metadata,
        title="Blended field (+ %i min)" % ((i + 1) * timestep),
    )
    plt.show()
    plt.close()
