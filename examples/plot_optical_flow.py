#!/bin/env python
"""
Optical flow
============

This tutorial offers a short overview to the optical flow routines available in 
pysteps and it will cover how to compute and plot the motion field from a 
sequence of radar images.

"""

from matplotlib import pyplot
import numpy as np
from pprint import pprint
from pysteps import io, motion
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver

###############################################################################
# Read the radar input images
# ---------------------------
#
# First thing, the sequence of radar composites is imported.

# Import the example radar composites
fns = (
    "data/sample_mch_radar_composite_00.gif", 
    "data/sample_mch_radar_composite_01.gif",
    )

R = []
for fn in fns:
    R_, _, metadata = io.import_mch_gif(fn)
    R.append(R_)
    R_ = None
R = np.stack(R)

# Convert to mm/h
R, metadata = conversion.to_rainrate(R, metadata)


# Log-transform the data
R_, metadata_ = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)


# Nicely print the metadata
pprint(metadata_)

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
V1 = oflow_method(R_)

# Plot the motion field
plot_precip_field(R[0, :, :], geodata=metadata, title="Lucas-Kanade")
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
V2 = oflow_method(R_)

# Plot the motion field
plot_precip_field(R[0, :, :], geodata=metadata, title="Variational echo tracking")
quiver(V2, geodata=metadata, step=25)

###############################################################################
# Dynamic and adaptive radar tracking of storms (DARTS)
# -----------------------------------------------------
#
# **Under development**
