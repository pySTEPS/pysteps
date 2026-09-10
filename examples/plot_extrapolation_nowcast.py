#!/bin/env python
"""
Extrapolation nowcast
=====================

This tutorial shows how to compute and plot an extrapolation nowcast using
Finnish radar data.

"""

from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from pysteps import io, motion, nowcasts, rcparams, verification
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver
from pysteps.xarray_helpers import geodata_from_dataset

###############################################################################
# Read the radar input images
# ---------------------------
#
# First, we will import the sequence of radar composites.
# You need the pysteps-data archive downloaded and the pystepsrc file
# configured with the data_source paths pointing to data folders.

# Selected case
date = datetime.strptime("201609281600", "%Y%m%d%H%M")
data_source = rcparams.data_sources["fmi"]
n_leadtimes = 12

###############################################################################
# Load the data from the archive
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]
timestep = data_source["timestep"]

# Find the input files from the archive
fns = io.archive.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2
)

# Read the radar composites
importer = io.get_method(importer_name, "importer")
precip_dataset = io.read_timeseries(fns, importer, **importer_kwargs)

# Convert to rain rate
precip_dataset = conversion.to_rainrate(precip_dataset)
precip_var = precip_dataset.attrs["precip_var"]

# Derive the geodata dict expected by the plotting routines from the dataset
geodata = geodata_from_dataset(precip_dataset)

# Plot the rainfall field
plot_precip_field(precip_dataset[precip_var][-1], geodata=geodata)
plt.show()

# Store the last frame for plotting it later later
R_ = precip_dataset[precip_var][-1].copy()

# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
# set the fill value to -15 dBR
precip_dataset = transformation.dB_transform(
    precip_dataset, threshold=0.1, zerovalue=-15.0
)

# Nicely print the metadata
pprint(dict(precip_dataset[precip_var].attrs))

###############################################################################
# Compute the nowcast
# -------------------
#
# The extrapolation nowcast is based on the estimation of the motion field,
# which is here performed using a local tracking approach (Lucas-Kanade).
# The most recent radar rainfall field is then simply advected along this motion
# field in oder to produce an extrapolation forecast.

# Estimate the motion field with Lucas-Kanade
oflow_method = motion.get_method("LK")
precip_dataset_w_motion = oflow_method(precip_dataset)

# Extrapolate the last radar observation
extrapolate = nowcasts.get_method("extrapolation")
precip_dataset_w_motion[precip_var] = precip_dataset_w_motion[precip_var].fillna(
    precip_dataset_w_motion[precip_var].attrs["zerovalue"]
)
precip_forecast = extrapolate(
    precip_dataset_w_motion.isel(time=slice(-1, None)), n_leadtimes
)

# Back-transform to rain rate
precip_forecast = transformation.dB_transform(
    precip_forecast, threshold=-10.0, inverse=True
)

# Plot the motion field
plot_precip_field(R_, geodata=geodata)
velocity = np.stack(
    [
        precip_dataset_w_motion["velocity_x"].values,
        precip_dataset_w_motion["velocity_y"].values,
    ]
)
quiver(velocity, geodata=geodata, step=50)
plt.show()

###############################################################################
# Verify with FSS
# ---------------
#
# The fractions skill score (FSS) provides an intuitive assessment of the
# dependency of skill on spatial scale and intensity, which makes it an ideal
# skill score for high-resolution precipitation forecasts.

# Find observations in the data archive
fns = io.archive.find_by_date(
    date,
    root_path,
    path_fmt,
    fn_pattern,
    fn_ext,
    timestep,
    num_prev_files=0,
    num_next_files=n_leadtimes,
)
# Read the radar composites
obs_dataset = io.read_timeseries(fns, importer, **importer_kwargs)
obs_dataset = conversion.to_rainrate(obs_dataset, 223.0, 1.53)

# Compute fractions skill score (FSS) for all lead times, a set of scales and 1 mm/h
fss = verification.get_method("FSS")
scales = [2, 4, 8, 16, 32, 64, 128, 256, 512]
thr = 1.0
score = []
for i in range(n_leadtimes):
    score_ = []
    for scale in scales:
        score_.append(
            fss(
                precip_forecast[precip_var][i].values,
                obs_dataset[precip_var][i + 1].values,
                thr,
                scale,
            )
        )
    score.append(score_)

plt.figure()
x = np.arange(1, n_leadtimes + 1) * timestep
plt.plot(x, score)
plt.legend(scales, title="Scale [km]")
plt.xlabel("Lead time [min]")
plt.ylabel("FSS ( > 1.0 mm/h ) ")
plt.title("Fractions skill score")
plt.show()

# sphinx_gallery_thumbnail_number = 3
