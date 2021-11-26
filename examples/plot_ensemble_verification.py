#!/bin/env python
"""
Ensemble verification
=====================

In this tutorial we perform a verification of a probabilistic extrapolation nowcast
using MeteoSwiss radar data.

"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from pysteps import io, nowcasts, rcparams, verification
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.postprocessing import ensemblestats
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field


###############################################################################
# Read precipitation field
# ------------------------
#
# First, we will import the sequence of MeteoSwiss ("mch") radar composites.
# You need the pysteps-data archive downloaded and the pystepsrc file
# configured with the data_source paths pointing to data folders.

# Selected case
date = datetime.strptime("201607112100", "%Y%m%d%H%M")
data_source = rcparams.data_sources["mch"]
n_ens_members = 20
n_leadtimes = 6
seed = 24

###############################################################################
# Load the data from the archive
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The data are upscaled to 2 km resolution to limit the memory usage and thus
# be able to afford a larger number of ensemble members.

root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]
timestep = data_source["timestep"]

# Find the radar files in the archive
fns = io.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2
)

# Read the data from the archive
importer = io.get_method(importer_name, "importer")
precip, _, metadata = io.read_timeseries(fns, importer, legacy=True, **importer_kwargs)

# Convert to rain rate
precip = precip.pysteps.to_rainrate()

# Upscale data to 2 km
precip = precip.coarsen(x=2, y=2).mean()

# Plot the rainfall field
plot_precip_field(precip[-1, :, :], geodata=metadata)
plt.show()

# Log-transform the data to unit of dBR
precip = precip.pysteps.db_transform()

# Set missing values with the fill value
precip[~np.isfinite(precip)] = -15.0

# Nicely print the metadata
pprint(metadata)

###############################################################################
# Forecast
# --------
#
# We use the STEPS approach to produce a ensemble nowcast of precipitation fields.

# Estimate the motion field
V = dense_lucaskanade(precip)

# Perform the ensemble nowcast with STEPS
nowcast_method = nowcasts.get_method("steps")
R_f = nowcast_method(
    precip[-3:, :, :],
    V,
    n_leadtimes,
    n_ens_members,
    n_cascade_levels=6,
    R_thr=-10.0,
    kmperpixel=2.0,
    timestep=timestep,
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    noise_method="nonparametric",
    vel_pert_method="bps",
    mask_method="incremental",
    seed=seed,
)

# Back-transform to rain rates
R_f = R_f.pysteps.db_transform(inverse=True)

# Plot some of the realizations
fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(221 + i)
    ax.set_title("Member %02d" % i)
    plot_precip_field(R_f[i, -1, :, :], geodata=metadata, colorbar=False, axis="off")
plt.tight_layout()
plt.show()

###############################################################################
# Verification
# ------------
#
# Pysteps includes a number of verification metrics to help users to analyze
# the general characteristics of the nowcasts in terms of consistency and
# quality (or goodness).
# Here, we will verify our probabilistic forecasts using the ROC curve,
# reliability diagrams, and rank histograms, as implemented in the verification
# module of pysteps.

# Find the files containing the verifying observations
fns = io.archive.find_by_date(
    date,
    root_path,
    path_fmt,
    fn_pattern,
    fn_ext,
    timestep,
    0,
    num_next_files=n_leadtimes,
)

# Read the observations
R_o, _, metadata_o = io.read_timeseries(fns, importer, legacy=True, **importer_kwargs)

# Convert to mm/h
R_o = R_o.pysteps.to_rainrate()

# Upscale data to 2 km
R_o = R_o.coarsen(x=2, y=2).mean()

# Compute the verification for the last lead time

# compute the exceedance probability of 0.1 mm/h from the ensemble
P_f = ensemblestats.excprob(R_f[:, -1, :, :], 0.1, ignore_nan=True)

###############################################################################
# ROC curve
# ~~~~~~~~~

roc = verification.ROC_curve_init(0.1, n_prob_thrs=10)
verification.ROC_curve_accum(roc, P_f, R_o[-1, :, :])
fig, ax = plt.subplots()
verification.plot_ROC(roc, ax, opt_prob_thr=True)
ax.set_title("ROC curve (+%i min)" % (n_leadtimes * timestep))
plt.show()

###############################################################################
# Reliability diagram
# ~~~~~~~~~~~~~~~~~~~

reldiag = verification.reldiag_init(0.1)
verification.reldiag_accum(reldiag, P_f, R_o[-1, :, :])
fig, ax = plt.subplots()
verification.plot_reldiag(reldiag, ax)
ax.set_title("Reliability diagram (+%i min)" % (n_leadtimes * timestep))
plt.show()

###############################################################################
# Rank histogram
# ~~~~~~~~~~~~~~

rankhist = verification.rankhist_init(R_f.shape[0], 0.1)
verification.rankhist_accum(rankhist, R_f[:, -1, :, :], R_o[-1, :, :])
fig, ax = plt.subplots()
verification.plot_rankhist(rankhist, ax)
ax.set_title("Rank histogram (+%i min)" % (n_leadtimes * timestep))
plt.show()

# sphinx_gallery_thumbnail_number = 5
