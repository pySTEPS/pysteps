"""
Lucas-Kanade
============

This example shows how the Lucas-Kande algorithm can be tuned to avoid the 
erroneous interpretation of velocities near the maximum range of the radars.

The Lucas-Kanade optical flow method identifies features on the input precipitation
frame and then tries to track them in the successive frame. 

With radar images, the limited extent of the domain means that often
precipitation leaves or enters the image. 
The tracking algorithm is particularly challenged by precipitation leaving 
the domain, which may result in nearly stationary velocities.

In order to address the boundary effects, the buffer_mask argument in the 
Lucas-Kanade method allows to exclude all vectors detected nearby nodata areas, 
meaning all those regions affected by blockage, limits of the radar coverage, 
or other radar artifacts that may mislead the tracking.
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

# Keep the reference frame in mm/h
ref_mm = R[0, :, :].copy()

# Extract radar mask
mask = np.ones(ref_mm.shape)
mask[~np.isnan(ref_mm)] = np.nan

# Log-transform the data [dBR]
R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)

# Keep the reference frame in dBR
ref_dbr = R[0].copy()
ref_dbr[ref_dbr < -10] = np.nan

################################################################################
# sparse Lucas-Kanade
# -------------------
#
# By setting the dense=False, the LK algorithm returns the sparse vectors only, 
# meaning that the interpolation is not performed. This allows us to better 
# identify the presence of nearly stationary motion in ares where precipitation 
# is leaving the domain (see blue circle in the figure below).


LK_optflow = motion.get_method("LK")

# no buffer (default settings)
x, y, u, v = LK_optflow(R, dense=False, buffer_mask=0, quality_level_ST=0.1)
plt.subplot(121)
plt.imshow(ref_dbr, cmap=plt.get_cmap("Greys"))
plt.imshow(mask, cmap=colors.ListedColormap(['black']), alpha=0.5)
plt.quiver(x, y, u, v, color="red", angles="xy", scale_units="xy", scale=.2)
circle = plt.Circle((620, 245), 100, color='b', clip_on=False, fill=False)
plt.gca().add_artist(circle)
plt.title("buffer_mask = 0")

# with buffer
x, y, u, v = LK_optflow(R, dense=False, buffer_mask=20, quality_level_ST=0.2)
plt.subplot(122)
plt.imshow(ref_dbr, cmap=plt.get_cmap("Greys"))
plt.imshow(mask, cmap=colors.ListedColormap(['black']), alpha=0.5)
plt.quiver(x, y, u, v, color="red", angles="xy", scale_units="xy", scale=.2)
circle = plt.Circle((620, 245), 100, color='b', clip_on=False, fill=False)
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
# negative bias that is introduced by the the erroneous interpretation of 
# velocities near the maximum range of the radars.

UV1 = LK_optflow(R, dense=True, buffer_mask=0, quality_level_ST=0.1)
UV2 = LK_optflow(R, dense=True, buffer_mask=20, quality_level_ST=0.2)

plt.subplot(121)
plt.imshow((UV1[0] - UV2[0])/UV2[0], cmap=cm.RdBu_r, vmin=-.2, vmax=.2)
plt.colorbar(fraction=0.04, pad=0.04)
plt.title(r"(u - u$_{buffer}$)/u$_{buffer}$")

plt.subplot(122)
plt.imshow((UV1[1] - UV2[1])/UV2[1], cmap=cm.RdBu_r, vmin=-.2, vmax=.2)
plt.colorbar(fraction=0.04, pad=0.04)
plt.title(r"(v - v$_{buffer}$)/v$_{buffer}$")

plt.tight_layout()
plt.show()

################################################################################
# Forecast skill
# --------------
#
# We are going to evaluate the benefit of buffering the radar mask by computing 
# the forecast error as the correlation coefficient (Spearman).

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

# Compute fractions skill score (FSS) for all lead times, a set of scales and 1 mm/h
skill = verification.get_method("corr_s")
score_1 = []
score_2 = []
for i in range(12):
    score_1.append(skill(R_f1[i, :, :], R_o[i + 1, :, :])["corr_s"])
    score_2.append(skill(R_f2[i, :, :], R_o[i + 1, :, :])["corr_s"])

x = (np.arange(12) + 1) * 5 # [min]
plt.plot(x, score_1, label="no mask buffer")
plt.plot(x, score_2, label="with mask buffer")
plt.legend()
plt.xlabel("Lead time [min]")
plt.ylabel("Corr. coeff. []")
plt.title("Spearman correlation")

plt.tight_layout()
plt.show()

# sphinx_gallery_thumbnail_number = 1
