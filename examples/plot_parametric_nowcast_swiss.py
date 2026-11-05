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
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import xarray as xr

from pysteps import io, nowcasts, rcparams
from pysteps.utils import conversion, dimension
from pysteps.visualization import basemaps, plot_precip_field
from pysteps.xarray_helpers import geodata_from_dataset

# Set ensemble size, forecast length, and a seed for repeatable results.
n_ens_members = 20
n_leadtimes = 6
seed = 24
rain_threshold = 1.0  # mm/h
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
observations = io.read_timeseries(filenames, importer, **source["importer_kwargs"])
observations = conversion.to_rainrate(observations)
precip_var = observations.attrs["precip_var"]

###############################################################################
# Prepare a complete spatial domain
# ---------------------------------
#
# The full Swiss composite has missing coverage outside the radar footprint.
# The parametric method requires complete finite grids. Select a 300 km
# square subdomain with complete coverage at all three times, in Swiss LV03
# coordinates (metres). Clip at the native resolution, then aggregate to 2 km
# to limit computation and memory. Missing pixels are not replaced with zero.

extent = (450000, 750000, 0, 300000)  # left, right, bottom, top in metres
observations = dimension.clip_domain(observations, extent)
observations = dimension.aggregate_fields_space(observations, 2000)
rain = observations[precip_var]
if (
    observations.sizes["time"] != 3
    or not np.isfinite(rain.values).all()
    or (rain.values < 0).any()
):
    raise ValueError("The tutorial requires three complete nonnegative rain fields")
geodata = geodata_from_dataset(observations)


# Use the same geographic metadata for observations and forecasts. A coarse
# basemap limits the auxiliary Natural Earth map data needed by the example.
def make_map(subplot=(1, 1, 1)):
    """Create the geographic axes before plotting the precipitation raster."""
    return basemaps.plot_geography(
        observations.attrs["projection"], extent, scale="110m", subplot=subplot
    )


# Inspect the newest observation.
plt.figure()
plot_precip_field(
    rain.isel(time=-1).values,
    ax=make_map(),
    geodata=geodata,
    axis="off",
    title="Swiss radar observation, 31 January 2017, 12:00 UTC",
)
plt.show()

###############################################################################
# Generate the parametric ensemble
# --------------------------------
#
# Pass linear rain rates directly: no external dBR transform or motion estimate
# is needed. The returned dataset contains rain rates in mm/h with dimensions
# (ens_number, time, y, x). Its first valid time is 12:05 and its last is 12:30.
# Grid cell size and time step are derived from the observation coordinates.
# Motion and rainfall statistics are estimated once and held fixed. Each member's
# rainfall pattern evolves separately using its own random perturbations.
# Setting the seed makes these results reproducible.
#
# The defaults include a 1 mm/h rain threshold and a 20 km spectral break.
# This example demonstrates the workflow; regional calibration and forecast
# skill require a separate assessment.

forecast = nowcasts.get_method("parametric")
ensemble, diagnostics = forecast(
    observations,
    timesteps=n_leadtimes,
    n_ens_members=n_ens_members,
    rain_threshold=rain_threshold,
    seed=seed,
    return_diagnostics=True,
)
forecast_rain = ensemble[ensemble.attrs["precip_var"]]
print("Ensemble dimensions:", dict(forecast_rain.sizes))
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

lead_minutes = float(
    (ensemble.time.values[-1] - ensemble.forecast_reference_time.values)
    / np.timedelta64(1, "m")
)
ensemble_mean = forecast_rain.isel(time=-1).mean("ens_number", skipna=False)
plt.figure()
plot_precip_field(
    ensemble_mean.values,
    ax=make_map(),
    geodata=geodata,
    axis="off",
    title=f"Ensemble mean parametric nowcast (+{lead_minutes:g} min)",
)
plt.show()

###############################################################################
# Plot four ensemble members
# --------------------------

fig = plt.figure(figsize=(10, 8))
for member in range(4):
    ax = make_map((2, 2, member + 1))
    ax = plot_precip_field(
        forecast_rain.isel(ens_number=member, time=-1).values,
        ax=ax,
        geodata=geodata,
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
# Estimate the probability of rain at or above 1 mm/h as the fraction of members
# meeting that threshold. Use the same threshold as the forecast.

probability_threshold = 1.0  # mm/h
probabilities = (forecast_rain.isel(time=-1) >= probability_threshold).mean(
    "ens_number"
)
plt.figure()
plot_precip_field(
    probabilities.values,
    ax=make_map(),
    geodata=geodata,
    axis="off",
    ptype="prob",
    units="mm/h",
    probthr=probability_threshold,
    title=f"Probability of rain >= {probability_threshold:g} mm/h (+{lead_minutes:g} min)",
)
plt.show()

###############################################################################
# Animate observations followed by the first ensemble member
# ----------------------------------------------------------
#
# Show the three radar observations followed by six forecasts on the same map
# and colour scale. Array index zero selects ensemble member number one.
# Display the valid times and pause at the last observation and forecast.

animation_member_index = 0
animation_fields = np.concatenate(
    [rain.values, forecast_rain.isel(ens_number=animation_member_index).values]
)
animation_times = np.concatenate([observations.time.values, ensemble.time.values])
animation_observations = observations.sizes["time"]
animation_member_number = int(ensemble.ens_number.values[animation_member_index])
animation_figure = plt.figure(figsize=(7, 7))
animation_axis = make_map()
plot_precip_field(animation_fields[0], ax=animation_axis, geodata=geodata, axis="off")
animation_raster = animation_axis.images[-1]
animation_figure.text(
    0.5,
    0.025,
    f"Forecast occurrence threshold: {rain_threshold:g} mm/h. QPE retains measured rates.",
    ha="center",
    fontsize=9,
)


def update_animation(frame):
    """Update rainfall and its timestamp while retaining geography and colours."""
    animation_raster.set_data(animation_fields[frame])
    valid_time = np.datetime_as_string(animation_times[frame], unit="m").replace(
        "T", " "
    )
    if frame < animation_observations:
        label = "QPE observation"
    else:
        lead = float(
            (animation_times[frame] - ensemble.forecast_reference_time.values)
            / np.timedelta64(1, "m")
        )
        label = (
            f"Forecast member index {animation_member_index} "
            f"(ens_number={animation_member_number}), +{lead:g} min"
        )
    animation_axis.set_title(f"{label}\n{valid_time} UTC")
    return (animation_raster, animation_axis.title)


animation_frames = (
    list(range(animation_observations))
    + [animation_observations - 1]
    + list(range(animation_observations, len(animation_times)))
    + [len(animation_times) - 1] * 2
)
animation = FuncAnimation(
    animation_figure,
    update_animation,
    frames=animation_frames,
    interval=800,
    repeat=True,
    cache_frame_data=False,
)
animation_path = (
    Path(rcparams.outputs["path_outputs"]).expanduser()
    / "parametric_nowcast_swiss_member0.gif"
)
animation_path.parent.mkdir(parents=True, exist_ok=True)
animation.save(animation_path, writer=PillowWriter(fps=1.25), dpi=110)
plt.close(animation_figure)
print("QPE and first-member animation:", animation_path.resolve())

###############################################################################
# Export the complete forecast dataset
# -------------------------------------
#
# Save the dataset with lossless compression and check that reopening preserves
# its values, coordinates, and attributes. Choose a persistent output path to
# keep the file. Projection metadata are retained as supplied; export does not
# add missing CF projection mappings.

with TemporaryDirectory(prefix="pysteps-parametric-swiss-") as output_directory:
    output_path = Path(output_directory) / "parametric_nowcast_swiss.nc"
    ensemble.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding={precip_var: {"zlib": True, "complevel": 1}},
    )
    with xr.open_dataset(output_path, engine="netcdf4") as reopened:
        xr.testing.assert_identical(ensemble, reopened.load())
    print(
        "NetCDF round trip preserved all forecast values, coordinates, and attributes."
    )

# sphinx_gallery_thumbnail_number = 3
