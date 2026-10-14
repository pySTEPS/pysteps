"""
Parametric nowcast with Swiss radar data
========================================

This tutorial uses the same Swiss radar sequence as the STEPS nowcast example.
Three rain-rate observations are passed directly to the parametric engine,
which performs transformation, motion estimation, parameter estimation, and
ensemble generation internally.

Download the `pysteps example data <https://github.com/pySTEPS/pysteps-data>`_
and configure the ``mch`` data source in ``pystepsrc`` before running this example.
The radar observations are provided by MeteoSwiss.
"""

from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from pysteps import io, nowcasts, rcparams
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import conversion, dimension
from pysteps.visualization import basemaps, plot_precip_field

n_ens_members = 20
n_leadtimes = 6
seed = 24
rain_threshold = 1.0  # mm/h: reference occurrence threshold of the engine
date = datetime(2017, 1, 31, 12, 0)

###############################################################################
# Read three observations in rain-rate units
# ------------------------------------------
#
# Load the observations at 11:50, 11:55, and 12:00 UTC using the usual pySTEPS
# data-source configuration. The imported accumulation is converted to mm/h.

source = rcparams.data_sources["mch"]
timestep = source["timestep"]  # minutes between observations
filenames = io.find_by_date(
    date,
    source["root_path"],
    source["path_fmt"],
    source["fn_pattern"],
    source["fn_ext"],
    timestep,
    num_prev_files=2,
)
importer = io.get_method(source["importer"], "importer")
rain, _, metadata = io.read_timeseries(filenames, importer, **source["importer_kwargs"])
rain, metadata = conversion.to_rainrate(rain, metadata)

###############################################################################
# Prepare a complete spatial domain
# ---------------------------------
#
# The full Swiss composite has missing coverage outside the radar footprint.
# This first parametric engine requires complete finite grids. Select a 300 km
# square subdomain with complete coverage at all three times, in Swiss LV03
# coordinates (metres). Clip at the native resolution, then aggregate to 2 km
# to limit computation and memory. Missing pixels are not replaced with zero.

extent = (450000, 750000, 0, 300000)  # left, right, bottom, top in metres
rain, metadata = dimension.clip_domain(rain, metadata, extent)
rain, metadata = dimension.aggregate_fields_space(rain, metadata, 2000)
if rain.shape[0] != 3 or not np.isfinite(rain).all() or np.any(rain < 0):
    raise ValueError("The tutorial requires three complete nonnegative rain fields")
kmperpixel = metadata["xpixelsize"] / 1000.0


# Use the same geographic metadata for observations and forecasts. A coarse
# basemap limits the auxiliary Natural Earth map data needed by the example.
def make_map(subplot=(1, 1, 1)):
    """Create the geographic axes before plotting the precipitation raster."""
    return basemaps.plot_geography(
        metadata["projection"], extent, scale="110m", subplot=subplot
    )


plt.figure()
plot_precip_field(
    rain[-1],
    ax=make_map(),
    geodata=metadata,
    axis="off",
    title="Swiss radar observation, 31 January 2017, 12:00 UTC",
)
plt.show()

###############################################################################
# Generate the parametric ensemble
# --------------------------------
#
# Pass linear rain rates directly: no external dBR transform or motion estimate
# is needed. The returned ensemble is also in mm/h, with shape
# (member, lead, y, x). Its first lead is +5 minutes and its last is +30 minutes.
# Fixed motion and estimated statistical parameters are shared across leads;
# each member has its own reproducible random stream.
#
# This example uses the existing regional reference defaults, including the
# 1 mm/h occurrence threshold and 20 km spectral break. Applying them to this
# Swiss case illustrates the interface; it does not establish Swiss calibration
# or forecast skill. The STEPS tutorial uses a different occurrence threshold.

forecast = nowcasts.get_method("parametric")
ensemble, diagnostics = forecast(
    rain,
    timesteps=n_leadtimes,
    n_ens_members=n_ens_members,
    kmperpixel=kmperpixel,
    timestep=timestep,
    rain_threshold=rain_threshold,
    seed=seed,
    return_diagnostics=True,
)
print("Ensemble shape:", ensemble.shape)
print("Generation parameters:")
pprint(diagnostics["parameters"])
print("Marginal moment source:", diagnostics["safeguards"]["moment_source"])
print("Persistence definition:", diagnostics["temporal_model"]["definition"])

###############################################################################
# Plot the ensemble mean
# ----------------------
#
# Averaging realizations smooths rainfall features whose position or intensity
# differs between members. Individual members retain more spatial variability.

lead_minutes = n_leadtimes * timestep
ensemble_mean = ensemble[:, -1].mean(axis=0)
plt.figure()
plot_precip_field(
    ensemble_mean,
    ax=make_map(),
    geodata=metadata,
    axis="off",
    title=f"Parametric ensemble mean (+{lead_minutes:g} min)",
)
plt.show()

###############################################################################
# Plot four ensemble members
# --------------------------

fig = plt.figure(figsize=(10, 8))
for member in range(4):
    ax = make_map((2, 2, member + 1))
    ax = plot_precip_field(
        ensemble[member, -1],
        ax=ax,
        geodata=metadata,
        colorbar=False,
        axis="off",
    )
    ax.set_title(f"Member {member + 1} (+{lead_minutes:g} min)")
plt.tight_layout()
plt.show()

###############################################################################
# Derive exceedance probabilities
# -------------------------------
#
# Use a probability threshold consistent with the engine's occurrence threshold.
# The ensemble fraction exceeding 1 mm/h is an estimated exceedance probability,
# not an independently calibrated probability product.

probability_threshold = 1.0  # mm/h
probabilities = excprob(ensemble[:, -1], probability_threshold)
plt.figure()
plot_precip_field(
    probabilities,
    ax=make_map(),
    geodata=metadata,
    axis="off",
    ptype="prob",
    units="mm/h",
    probthr=probability_threshold,
    title=f"Probability of rain >= {probability_threshold:g} mm/h (+{lead_minutes:g} min)",
)
plt.show()

# sphinx_gallery_thumbnail_number = 3
