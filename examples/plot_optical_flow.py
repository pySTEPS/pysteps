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
import xarray as xr

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

# Read the radar composites as an xarray Dataset
importer = io.get_method(importer_name, "importer")
ds = io.read_timeseries(fns, importer, **importer_kwargs)

###############################################################################
# Preprocess the data
# ~~~~~~~~~~~~~~~~~~~

# Convert to mm/h
ds = conversion.to_rainrate(ds)


# Figure out the precip variable name (ensure it's present)
ds.attrs.setdefault("precip_var", "precip_intensity")
precip_var = ds.attrs["precip_var"]

# Store the reference frame (last frame for background plotting)
R = ds[precip_var]
R_ = R.isel(time=-1).copy()

# Log-transform the data [dBR] (sets zerovalue in attrs if your transform does so)
ds = transformation.dB_transform(ds, threshold=0.1, zerovalue=-15.0)

# Fill missing with the dBR zerovalue for methods that require it
zerovalue_db = ds.attrs.get("zerovalue", -15.0)
ds[precip_var] = ds[precip_var].where(xr.ufuncs.isfinite(ds[precip_var]), zerovalue_db)

# Nicely print the dataset attrs (formerly 'metadata')
pprint(ds.attrs)

# Build geodata (plot_precip_field expects these keys)
geodata = {
    "projection": ds.attrs.get("projection", None),
    "x1": float(R.x.values[0]),
    "x2": float(R.x.values[-1]),
    "y1": float(R.y.values[0]),
    "y2": float(R.y.values[-1]),
    "yorigin": "lower",
}

# Small helper to convert motion dataset -> quiver-compatible (2, m, n) array
def _stack_uv(motion_ds: xr.Dataset) -> np.ndarray:
    return np.stack(
        [motion_ds["velocity_x"].values, motion_ds["velocity_y"].values], axis=0
    )

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
# pass a short time slice (last 3 frames) as a Dataset
V1_ds = oflow_method(ds.isel(time=slice(-3, None)))
V1 = _stack_uv(V1_ds)

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
V2_ds = oflow_method(ds.isel(time=slice(-3, None)))
V2 = _stack_uv(V2_ds)

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

oflow_method = motion.get_method("DARTS")
V3_ds = oflow_method(ds)  # pass the whole sequence
V3 = _stack_uv(V3_ds)

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
# inconsitency during the solution of the optical flow equations.

oflow_method = motion.get_method("proesmans")
V4_ds = oflow_method(ds.isel(time=slice(-2, None)))
V4 = _stack_uv(V4_ds)

# Plot the motion field
plot_precip_field(R_, geodata=geodata, title="Proesmans")
quiver(V4, geodata=geodata, step=25)
plt.show()

# sphinx_gallery_thumbnail_number = 1
