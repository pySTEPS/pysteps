#!/bin/env python
"""
Extrapolation nowcast
=====================

This tutorial shows how to compute and plot an extrapolation nowcast using 
Finnish radar data.

"""

from pylab import *
from datetime import datetime
import numpy as np
from pprint import pprint
from pysteps import io, motion, nowcasts, rcparams, verification
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver

###############################################################################
# Read the radar input images
# ---------------------------
#
# First thing, the sequence of radar composites is imported, converted and
# transformed into units of dBR.

date = datetime.strptime("201609281600", "%Y%m%d%H%M")
data_source = "fmi"
n_leadtimes = 12

# Load data source config
root_path = rcparams.data_sources[data_source]["root_path"]
path_fmt = rcparams.data_sources[data_source]["path_fmt"]
fn_pattern = rcparams.data_sources[data_source]["fn_pattern"]
fn_ext = rcparams.data_sources[data_source]["fn_ext"]
importer_name = rcparams.data_sources[data_source]["importer"]
importer_kwargs = rcparams.data_sources[data_source]["importer_kwargs"]
timestep = rcparams.data_sources[data_source]["timestep"]

# Find the input files from the archive
fns = io.archive.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2
)

# Read the radar composites
importer = io.get_method(importer_name, "importer")
Z, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)

# Convert to rain rate using the finnish Z-R relationship
R, metadata = conversion.to_rainrate(Z, metadata, 223.0, 1.53)

# Plot the rainfall field
plot_precip_field(R[-1, :, :], geodata=metadata)

# Store the last frame for plotting it later later
R_ = R[-1, :, :].copy()

# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h,
# set the fill value to -15 dBR
R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)

# Nicely print the metadata
pprint(metadata)

###############################################################################
# Compute the nowcast
# -------------------
#
# The extrapolation nowcast is based on the estimation of the motion field,
# which is here performed using a local tacking approach (Lucas-Kanade).
# The most recent radar rainfall field is then simply advected along this motion
# field in oder to produce an extrapolation forecast.

# Estimate the motion field with Lucas-Kanade
oflow_method = motion.get_method("LK")
V = oflow_method(R[-3:, :, :])

# Extrapolate the last radar observation
extrapolate = nowcasts.get_method("extrapolation")
R[~np.isfinite(R)] = metadata["zerovalue"]
R_f = extrapolate(R[-1, :, :], V, n_leadtimes)

# Back-transform to rain rate
R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]

# Plot the motion field
plot_precip_field(R_, geodata=metadata)
quiver(V, geodata=metadata, step=50)

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
R_o, _, metadata_o = io.read_timeseries(fns, importer, **importer_kwargs)
R_o, metadata_o = conversion.to_rainrate(R_o, metadata_o, 223.0, 1.53)

# Compute fractions skill score (FSS) for all lead times, a set of scales and 1 mm/h
fss = verification.get_method("FSS")
scales = [2, 4, 8, 16, 32, 64, 128, 256, 512]
thr = 1.0
score = []
for i in range(n_leadtimes):
    score_ = []
    for scale in scales:
        score_.append(fss(R_f[i, :, :], R_o[i + 1, :, :], thr, scale))
    score.append(score_)

figure()
x = np.arange(1, n_leadtimes + 1) * timestep
plot(x, score)
legend(scales, title="Scale [km]")
xlabel("Lead time [min]")
ylabel("FSS ( > 1.0 mm/h ) ")
title("Fractions skill score")

# sphinx_gallery_thumbnail_number = 3
