"""
Optical flow
============

This tutorial offers a short overview of the optical flow routines available in
pysteps and it will cover how to compute and plot the motion field from a
sequence of radar images.
"""

from datetime import datetime
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

from pysteps import io, motion, rcparams
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver

################################################################################
# Read the radar input images
# ---------------------------
#
# First, we will import the sequence of radar composites.
# You need the pysteps-data archive downloaded and the pystepsrc file
# configured with the data_source paths pointing to data folders.

# Selected case
date = datetime.strptime("201505151630", "%Y%m%d%H%M")
data_source = rcparams.data_sources["mch"]

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
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=9
)

# Read the radar composites
importer = io.get_method(importer_name, "importer")
precip_dataset = io.read_timeseries(fns, importer, **importer_kwargs)

###############################################################################
# Preprocess the data
# ~~~~~~~~~~~~~~~~~~~

# Convert to mm/h
precip_dataset = conversion.to_rainrate(precip_dataset)
precip_var = precip_dataset.attrs["precip_var"]

# Derive the geodata needed by the plotting functions from the dataset
geodata = {
    "projection": precip_dataset.attrs["projection"],
    "x1": precip_dataset.x.values[0],
    "x2": precip_dataset.x.values[-1],
    "y1": precip_dataset.y.values[0],
    "y2": precip_dataset.y.values[-1],
    "yorigin": "lower",
}

# Store the reference frame (in mm/h, for plotting)
R_ = precip_dataset[precip_var][-1].copy()

# Log-transform the data [dBR]
precip_dataset = transformation.dB_transform(
    precip_dataset, threshold=0.1, zerovalue=-15.0
)

# Nicely print the attributes of the precipitation variable
pprint(precip_dataset[precip_var].attrs)

################################################################################
# Lucas-Kanade (LK)
# -----------------
#
# The Lucas-Kanade optical flow method implemented in pysteps is a local
# tracking approach that relies on the OpenCV package.
# Local features are tracked in a sequence of two or more radar images. The
# scheme includes a final interpolation step in order to produce a smooth
# field of motion vectors.

oflow_method = motion.get_method("LK")
dataset_lk = oflow_method(precip_dataset.isel(time=slice(-3, None)))
V1 = np.stack([dataset_lk["velocity_x"].values, dataset_lk["velocity_y"].values])

# Plot the motion field on top of the reference frame
plot_precip_field(R_, geodata=geodata, title="LK")
quiver(V1, geodata=geodata, step=25)
plt.show()

################################################################################
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
dataset_vet = oflow_method(precip_dataset.isel(time=slice(-3, None)))
V2 = np.stack([dataset_vet["velocity_x"].values, dataset_vet["velocity_y"].values])

# Plot the motion field
plot_precip_field(R_, geodata=geodata, title="VET")
quiver(V2, geodata=geodata, step=25)
plt.show()

################################################################################
# Dynamic and adaptive radar tracking of storms (DARTS)
# -----------------------------------------------------
#
# DARTS uses a spectral approach to optical flow that is based on the discrete
# Fourier transform (DFT) of a temporal sequence of radar fields.
# The level of truncation of the DFT coefficients controls the degree of
# smoothness of the estimated motion field, allowing for an efficient
# motion estimation. DARTS requires a longer sequence of radar fields for
# estimating the motion, here we are going to use all the available 10 fields.

# Fill missing values with the fill value before running DARTS
zerovalue = precip_dataset[precip_var].attrs["zerovalue"]
precip_dataset[precip_var] = precip_dataset[precip_var].where(
    np.isfinite(precip_dataset[precip_var]), zerovalue
)

oflow_method = motion.get_method("DARTS")
dataset_darts = oflow_method(precip_dataset)  # needs longer training sequence
V3 = np.stack([dataset_darts["velocity_x"].values, dataset_darts["velocity_y"].values])

# Plot the motion field
plot_precip_field(R_, geodata=geodata, title="DARTS")
quiver(V3, geodata=geodata, step=25)
plt.show()

################################################################################
# Anisotropic diffusion method (Proesmans et al 1994)
# ---------------------------------------------------
#
# This module implements the anisotropic diffusion method presented in Proesmans
# et al. (1994), a robust optical flow technique which employs the notion of
# inconsistency during the solution of the optical flow equations.

oflow_method = motion.get_method("proesmans")
dataset_proesmans = oflow_method(precip_dataset.isel(time=slice(-2, None)))
V4 = np.stack(
    [dataset_proesmans["velocity_x"].values, dataset_proesmans["velocity_y"].values]
)

# Plot the motion field
plot_precip_field(R_, geodata=geodata, title="Proesmans")
quiver(V4, geodata=geodata, step=25)
plt.show()

################################################################################
# Farnebäck smoothed method
# -------------------------
#
# This module implements the pyramidal decomposition method for motion estimation
# of Farnebäck as implemented in OpenCV, with an option for smoothing and
# renormalization of the motion fields proposed by Driedger et al.:
# https://cmosarchives.ca/Congress_P_A/program_abstracts2022.pdf (p. 392).

oflow_method = motion.get_method("farneback")
dataset_farneback = oflow_method(
    precip_dataset.isel(time=slice(-2, None)), verbose=True
)
V5 = np.stack(
    [dataset_farneback["velocity_x"].values, dataset_farneback["velocity_y"].values]
)

# Plot the motion field
plot_precip_field(R_, geodata=geodata, title="Farneback")
quiver(V5, geodata=geodata, step=25)
plt.show()

# sphinx_gallery_thumbnail_number = 1
