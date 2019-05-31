"""
Handling of no-data in Lucas-Kanade
===================================

Areas of missing data in radar images are typically caused by visibility limits
such as beam blockage and the radar coverage itself. These artifacts can mislead 
the echo tracking algorithms. For instance, precipitation leaving the domain
might be erroneously detected as nearly stationary velocities.

This example shows how the Lucas-Kanade algorithm can be tuned to avoid the 
erroneous interpretation of velocities near the maximum range of the radars by 
buffering the no-data mask in the radar image in order to exclude all vectors 
detected nearby nodata areas.
"""

from datetime import datetime
from pprint import pprint
from matplotlib import cm, colors

import matplotlib.pyplot as plt
import numpy as np

from pysteps import io, motion, nowcasts, rcparams, verification
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
date = datetime.strptime("201607112100", "%Y%m%d%H%M")
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
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=1
)

# Read the radar composites
importer = io.get_method(importer_name, "importer")
R, quality, metadata = io.read_timeseries(fns, importer, **importer_kwargs)

del quality  # Not used

###############################################################################
# Preprocess the data
# ~~~~~~~~~~~~~~~~~~~

# Convert to mm/h
R, metadata = conversion.to_rainrate(R, metadata)

# Keep the reference frame in mm/h and its mask (for plotting purposes)
ref_mm = R[0, :, :].copy()
mask = np.ones(ref_mm.shape)
mask[~np.isnan(ref_mm)] = np.nan

# Log-transform the data [dBR]
R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)

# Keep the reference frame in dBR (for plotting purposes)
ref_dbr = R[0].copy()
ref_dbr[ref_dbr < -10] = np.nan

# Plot the reference field
plot_precip_field(ref_mm, title="Reference field")
circle = plt.Circle((620, 400), 100, color="b", clip_on=False, fill=False)
plt.gca().add_artist(circle)
plt.show()

###############################################################################
# Notice the "half-in, half-out" precipitation area within the blue circle.
# As we are going to show next, the tracking algorithm can erroneously interpret
# precipitation leaving the domain as stationary motion.
#
# Also note that the radar image includes NaNs in areas of missing data.
# These are used by the optical flow algorithm to define the radar mask.
#
# Sparse Lucas-Kanade
# -------------------
#
# By setting the dense=False, the LK algorithm returns the motion vectors detected
# by the Lucas-Kanade scheme without interpolating them on the grid. This allows
# us to better identify the presence of wrongly detected stationary motion in
# ares where precipitation is leaving the domain (look for the red dots within
# the blue circle in the left panel of the figure below).

# get Lucas-Kanade optical flow method
LK_optflow = motion.get_method("LK")

# Mask invalid values
R = np.ma.masked_invalid(R)
R.data[R.mask] = np.nan

# Use default settings (i.e., no buffering of the radar mask)
x, y, u, v = LK_optflow(R, dense=False, buffer_mask=0, quality_level_ST=0.1)
plt.subplot(121)
plt.imshow(ref_dbr, cmap=plt.get_cmap("Greys"))
plt.imshow(mask, cmap=colors.ListedColormap(["black"]), alpha=0.5)
plt.quiver(x, y, u, v, color="red", angles="xy", scale_units="xy", scale=0.2)
circle = plt.Circle((620, 245), 100, color="b", clip_on=False, fill=False)
plt.gca().add_artist(circle)
plt.title("buffer_mask = 0")

# with buffer
x, y, u, v = LK_optflow(R, dense=False, buffer_mask=20, quality_level_ST=0.2)
plt.subplot(122)
plt.imshow(ref_dbr, cmap=plt.get_cmap("Greys"))
plt.imshow(mask, cmap=colors.ListedColormap(["black"]), alpha=0.5)
plt.quiver(x, y, u, v, color="red", angles="xy", scale_units="xy", scale=0.2)
circle = plt.Circle((620, 245), 100, color="b", clip_on=False, fill=False)
plt.gca().add_artist(circle)
plt.title("buffer_mask = 20")

plt.tight_layout()
plt.show()

################################################################################
# Dense Lucas-Kanade
# ------------------
#
# The above displacement vectors produced by the Lucas-Kanade method are
# interpolated to produce a full field of motion (i.e., dense=True).
# By comparing the x- and y-components of the motion field, we can easily notice
# the negative bias that is introduced by the the erroneous interpretation of
# velocities near the maximum range of the radars.

UV1 = LK_optflow(R, dense=True, buffer_mask=0, quality_level_ST=0.1)
UV2 = LK_optflow(R, dense=True, buffer_mask=20, quality_level_ST=0.2)

V1 = np.sqrt(UV1[0] ** 2 + UV1[1] ** 2)
V2 = np.sqrt(UV2[0] ** 2 + UV2[1] ** 2)

plt.imshow((V1 - V2) / V2, cmap=cm.RdBu_r, vmin=-0.1, vmax=0.1)
plt.colorbar(fraction=0.04, pad=0.04)
plt.title("Relative difference in motion speed")
plt.show()

################################################################################
# Notice that the original motion field is significantly slower (more than 10%
# slower) because of the presence of wrong stationary motion vectors at the edges.
#
# Forecast skill
# --------------
#
# We are now going to evaluate the benefit of buffering the radar mask by computing
# the forecast skill in terms of the Spearman correlation coefficient.

extrapolate = nowcasts.get_method("extrapolation")
R[~np.isfinite(R)] = metadata["zerovalue"]
R_f1 = extrapolate(R[-1], UV1, 12)
R_f2 = extrapolate(R[-1], UV2, 12)

# Back-transform to rain rate
R_f1 = transformation.dB_transform(R_f1, threshold=-10.0, inverse=True)[0]
R_f2 = transformation.dB_transform(R_f2, threshold=-10.0, inverse=True)[0]

# Find the input files from the archive
fns = io.archive.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_next_files=12
)

# Read and convert the radar composites
R_o, _, metadata_o = io.read_timeseries(fns, importer, **importer_kwargs)
R_o, metadata_o = conversion.to_rainrate(R_o, metadata_o)

# Compute Spearman correlation
skill = verification.get_method("corr_s")
score_1 = []
score_2 = []
for i in range(12):
    score_1.append(skill(R_f1[i, :, :], R_o[i + 1, :, :])["corr_s"])
    score_2.append(skill(R_f2[i, :, :], R_o[i + 1, :, :])["corr_s"])

x = (np.arange(12) + 1) * 5  # [min]
plt.plot(x, score_1, label="no mask buffer")
plt.plot(x, score_2, label="with mask buffer")
plt.legend()
plt.xlabel("Lead time [min]")
plt.ylabel("Corr. coeff. []")
plt.title("Spearman correlation")

plt.tight_layout()
plt.show()

# sphinx_gallery_thumbnail_number = 2
