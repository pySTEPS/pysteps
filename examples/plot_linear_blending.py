# -*- coding: utf-8 -*-

"""
Plot linear blending
====================
"""

from matplotlib import cm, pyplot as plt
import numpy as np
from pprint import pprint
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps import io, rcparams, nowcasts
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field
from datetime import datetime
from pysteps.utils import dimension
from pysteps.nowcasts.linear_blending import forecast


def gaussian(x, max, mean, sigma):
    return max * np.exp(-(x - mean) * (x - mean) / sigma / sigma / 2)


def dummy_nwp(R, n_leadtimes, max=20, mean=0, sigma=0.25, speed=100):
    """Generates dummy NWP data with the same dimension as the input
    precipitation field R. The NWP data is a vertical line with a Gaussian
    profile moving to the left"""

    # R is original radar image
    rows = R.shape[0]
    cols = R.shape[1]

    # Initialise the dummy NWP data
    R_nwp = np.zeros((n_leadtimes, rows, cols))
    x = np.linspace(-5, 5, cols)

    for n in range(n_leadtimes):
        for i in range(rows):
            R_nwp[n, i, :] = gaussian(x, max, mean, sigma)
        mean -= speed / rows

    return R_nwp


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
R, _, metadata = io.read_timeseries(fns, importer, legacy=True, **importer_kwargs)

# Convert to rain rate
R, metadata = conversion.to_rainrate(R, metadata)

# Upscale data to 2 km to limit memory usage
R, metadata = dimension.aggregate_fields_space(R, metadata, 2000)

# Import the dummy NWP data (vertical line moving to the left)
R_nwp = dummy_nwp(R[-1, :, :], n_leadtimes + 1, max=7, mean=4, speed=0.2 * 350)

# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
# set the fill value to -15 dBR
R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)

# Set missing values with the fill value
# R[~np.isfinite(R)] = -15.0

# Nicely print the metadata
pprint(metadata)

###############################################################################
# Linear blending of nowcast and NWP data

# Estimate the motion field
V = dense_lucaskanade(R)

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
R_blended = forecast(
    R[-3:, :, :],
    V,
    n_leadtimes,
    timestep,
    "steps",
    R_nwp=R_nwp[1:, :, :],
    start_blending=start_blending,
    end_blending=end_blending,
    nowcast_kwargs=nowcast_kwargs,
)

"""
extrap_kwargs = {"allow_nonfinite_values": True}
nowcast_kwargs = {"extrap_kwargs": extrap_kwargs}

# Calculate the blended precipitation field
R_blended = forecast(
    R[-1, :, :],
    V,
    n_leadtimes,
    timestep,
    "extrapolation",
    R_nwp=R_nwp[1: :, :],
    start_blending=start_blending,
    end_blending=end_blending,
    nowcast_kwargs=nowcast_kwargs,
)
"""

# Calculate the ensemble average
if len(R_blended.shape) == 4:
    R_blended_mean = np.mean(R_blended[:, :, :, :], axis=0)
else:
    R_blended_mean = np.copy(R_blended)

# Plot the blended field
for i in range(n_leadtimes):
    plot_precip_field(
        R_blended_mean[i, :, :],
        geodata=metadata,
        title="Blended field (+ %i min)" % ((i + 1) * timestep),
    )
    plt.show()
    plt.close()
