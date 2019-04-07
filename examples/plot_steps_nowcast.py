#!/bin/env python
"""
STEPS nowcast
=============

This tutorial shows how to compute and plot an ensemble nowcast using Swiss
radar data.

"""

from pylab import *
from datetime import datetime
from pysteps import io, nowcasts, rcparams
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field

# Set nowcast parameters
n_ens_members = 12
n_leadtimes = 12
seed = 24

###############################################################################
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
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2,
)

# Read the data from the archive
importer = io.get_method(importer_name, "importer")
R, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)

# Convert to rain rate
R, metadata = conversion.to_rainrate(R, metadata)

# Log-transform the data to unit of dBR, set the threshold to 0.1 mm/h
R = transformation.dB_transform(R, threshold=0.1, zerovalue=-15.0)[0]

# Set missing values with the fill value
R[~np.isfinite(R)] = -15.0

###############################################################################
# Deterministic nowcast with S-PROG
# ---------------------------------
#
# First, the motiong field is estimated using a local tracking approach based
# on the Lucas-Kanade optical flow.
# The motion field can then be used to generate a deterministic nowcast with
# the S-PROG model, which implements a scale filtering appraoch in order to
# progressively remove the unpredictable spatial scales during the forecast.

# Estimate the motion field
V = dense_lucaskanade(R)

# The S-PROG nowcast
nowcast_method = nowcasts.get_method("sprog")
R_f = nowcast_method(
    R[-3:, :, :],
    V,
    n_leadtimes,
    n_cascade_levels=8,
    R_thr=-10.0,
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    probmatching_method="mean",
)

# Back-transform to rain rate
R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]

# Plot the S-PROG forecast
figure()
bm = plot_precip_field(R_f[-1, :, :], geodata=metadata, title="S-PROG")

###############################################################################
# As we can see from the figure above, the forecast produced by S-PROG is a
# smooth field. In other words, the forecast variance is lower than the
# variance of the original observed field.
# However, given applications demand that the forecast retain the same
# statistical properties of the observations. In such cases, the S-PROG
# forecsats are of limited use and a stochatic approahc might be of more
# interest.

###############################################################################
# Stochastic nowcast with STEPS
# -----------------------------
#
#
# The S-PROG approach is extended to include a stochastic term which represents
# the variance linked to the unpredictable development of precipitation. This
# approach is known as STEPS (short-term ensemble prediction system).

# The STEPES nowcast
nowcast_method = nowcasts.get_method("steps")
R_f = nowcast_method(
    R[-3:, :, :],
    V,
    12,
    n_ens_members=n_ens_members,
    n_cascade_levels=8,
    R_thr=-10.0,
    kmperpixel=1.0,
    timestep=5,
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    noise_method="nonparametric",
    vel_pert_method="bps",
    mask_method="incremental",
    seed=seed,
)

# Back-transform to rain rates
R_f = transformation.dB_transform(R_f, threshold=-10.0, inverse=True)[0]


# Plot the ensemble mean
R_f_mean = np.mean(R_f[:, -1, :, :], axis=0)
figure()
bm = plot_precip_field(R_f_mean, geodata=metadata, title="Ensemble mean")

###############################################################################
# The mean of the ensemble displays similar properties as the S-PROG
# forecast seen above, although the degree of smoothing strongly depends on
# the ensemble size. In this sense, the S-PROG forecast can be seen as
# the mean forecast from an ensemble of infinite size.

# Plot the first two realizations
fig = figure()
for i in range(2):
    ax = fig.add_subplot(121 + i)
    ax.set_title("Member %02d" % i)
    bm = plot_precip_field(R_f[i, -1, :, :], geodata=metadata)
tight_layout()

###############################################################################
# As we can see from these two members of the ensemble, the stochastic forecast
# mantains the same variance as in the observed rainfall field.
# Finally, it is possible to derive probabilities from our ensemble forecast.

# Compute exceedence probabilities for a 0.5 mm/h threshold
P = excprob(R_f[:, -1, :, :], 0.5)

# Plot the field of probabilities
figure()
bm = plot_precip_field(
    P,
    geodata=metadata,
    drawlonlatlines=False,
    type="prob",
    units="mm/h",
    probthr=0.5,
    title="Exceedence probability",
)

# sphinx_gallery_thumbnail_number = 3
