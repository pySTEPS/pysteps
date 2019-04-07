#!/bin/env python
"""
Optical flow
============

This tutorial offers a short overview to the optical flow routines available in 
pysteps and it will cover how to compute and plot the motion field from a 
sequence of radar images.

"""

from datetime import datetime
from matplotlib import pyplot
import numpy as np
from pprint import pprint
from pysteps import io, motion, rcparams
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver

###############################################################################
# Read the radar input images
# ---------------------------
#
# First thing, the sequence of radar composites is imported, converted and 
# transformed into units of dBR.

date = datetime.strptime("201505151630", "%Y%m%d%H%M")
data_source = "mch"
root_path = rcparams.data_sources[data_source]["root_path"]
path_fmt = rcparams.data_sources[data_source]["path_fmt"]
fn_pattern = rcparams.data_sources[data_source]["fn_pattern"]
fn_ext = rcparams.data_sources[data_source]["fn_ext"]
importer_name = rcparams.data_sources[data_source]["importer"]
importer_kwargs = rcparams.data_sources[data_source]["importer_kwargs"]

# Find the input files from the archive
fns = io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=9)

# Read the radar composites
importer = io.get_method(importer_name, "importer")
R, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)

# Convert to mm/h
R, metadata = conversion.to_rainrate(R, metadata)

# Store the last frame for polotting it later later
R_ = R[-1, : , :].copy()

# Log-transform the data
R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)

# Nicely print the metadata
pprint(metadata)

###############################################################################
# Lucas-Kanade (LK)
# -----------------
#
# The Lucas-Kanade optical flow method implemented in pysteps is a local
# tracking apporach that relies on the OpenCV package.
# Local features are tracked in a sequence of two or more radar images. The
# scheme includes a final interpolation step in order to produce a smooth 
# field of motion vectors.

oflow_method = motion.get_method("LK")
V1 = oflow_method(R[-3:, :, :])

# Plot the motion field
plot_precip_field(R_, geodata=metadata, title="Lucas-Kanade")
quiver(V1, geodata=metadata, step=25)

###############################################################################
# Variational echo tracking (VET)
# -------------------------------
#
# This module implements the VET algorithm presented 
# by Laroche and Zawadzki (1995) and used in the McGill Algorithm for 
# Prediction by Lagrangian Extrapolation (MAPLE) described in 
# Germann and Zawadzki (2002).
# The approach essentially consists of a global optimization routine that seeks
# at minimizing a cost function between the displaced and the reference image. 

oflow_method = motion.get_method("VET")
V2 = oflow_method(R[-3:, :, :])

# Plot the motion field
plot_precip_field(R_, geodata=metadata, title="Variational echo tracking")
quiver(V2, geodata=metadata, step=25)

###############################################################################
# Dynamic and adaptive radar tracking of storms (DARTS)
# -----------------------------------------------------
#
# DARTS uses a spectral approach to optical flow that is based on the discrete
# Fourier transform (DFT) of a temporal sequence of radar fields. 
# The level of truncation of the DFT coefficients controls the degree of 
# smoothness of the estimated motion field, allowing for an efficient 
# motion estimation. DARTS requires a longer sequence of radar fields for 
# estimating the motion, here we are going to use all the available 10 fields.

oflow_method = motion.get_method("DARTS")
R[~np.isfinite(R)] = metadata["zerovalue"]
V3 = oflow_method(R) # needs longer training sequence

# Plot the motion field
plot_precip_field(R_, geodata=metadata, title="DARTS")
quiver(V3, geodata=metadata, step=25)

# sphinx_gallery_thumbnail_number = 1