#!/bin/env python
"""
STEPS nowcast
=============

This tutorial shows how to compute and plot an ensemble nowcast using Finnish
radar data.

"""

from pylab import *
from datetime import datetime
from pysteps.io.archive import find_by_date
from pysteps.io.importers import import_fmi_pgm
from pysteps.io.readers import read_timeseries
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps import nowcasts, rcparams
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field

# Set nowcast parameters
date = datetime.strptime("201609281600", "%Y%m%d%H%M")
n_ens_members = 12
seed = 24

###############################################################################
# Read precipitation field
# ------------------------
#
# First thing, the sequence of Finnish radar composites is imported, converted and 
# transformed into units of dBR.

# Load data source config
root_path = rcparams.data_sources["fmi"]["root_path"]
path_fmt = rcparams.data_sources["fmi"]["path_fmt"]
fn_pattern = rcparams.data_sources["fmi"]["fn_pattern"]
fn_ext = rcparams.data_sources["fmi"]["fn_ext"]
timestep = rcparams.data_sources["fmi"]["timestep"]

# Find the radar files in the archive
inputfns = find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=9
)

# Read the data from the archive
Z, _, metadata = read_timeseries(inputfns, import_fmi_pgm, gzipped=True)

# Convert to rain rate using the finnish Z-R relationship
R = conversion.to_rainrate(Z, metadata, 223.0, 1.53)[0]

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
    12,
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
bm = plot_precip_field(
    R_f[-1, :, :],
    map="basemap",
    geodata=metadata,
    drawlonlatlines=False,
    basemap_resolution="h",
    basemap_scale_args=[30.0, 58.5, 30.2, 58.5, 120],
    title="S-PROG",
)

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
bm = plot_precip_field(
    R_f_mean,
    map="basemap",
    geodata=metadata,
    drawlonlatlines=False,
    basemap_resolution="h",
    basemap_scale_args=[30.0, 58.5, 30.2, 58.5, 120],
    title="Ensemble mean",
)

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
    bm = plot_precip_field(
        R_f[i, -1, :, :],
        map="basemap",
        geodata=metadata,
        drawlonlatlines=False,
        basemap_resolution="h",
        basemap_scale_args=[30.0, 58.5, 30.2, 58.5, 120],
    )
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
    map="basemap",
    geodata=metadata,
    drawlonlatlines=False,
    basemap_resolution="h",
    basemap_scale_args=[30.0, 58.5, 30.2, 58.5, 120],
    type="prob",
    units="mm/h",
    probthr=0.5,
    title="Exceedence probability",
)

# sphinx_gallery_thumbnail_number = 3
