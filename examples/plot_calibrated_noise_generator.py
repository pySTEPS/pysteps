"""
Calibrated stochastic rainfall fields
=====================================

This example creates random rainfall patterns with statistics similar to a
Swiss radar observation. It estimates the rainy area, rain-intensity
distribution, and spatial structure, then uses them to generate three fields.
The figures compare the generated fields with the observation.

It uses the same radar image as
:ref:`sphx_glr_auto_examples_plot_noise_generators.py`. The final section shows
how to separate a generated field into spatial scales for use in a nowcast.

Download the `pysteps example data <https://github.com/pySTEPS/pysteps-data>`_
and configure the ``mch`` data source in ``pystepsrc``. MeteoSwiss provides the
radar observation used here. This example generates independent spatial fields;
it does not perform temporal nowcasting.
"""

from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from pysteps import io, rcparams
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.noise.parametric import (
    generate_calibrated_field,
    generate_stochastic_field,
)
from pysteps.utils import conversion, dimension, transformation
from pysteps.utils.spectral import fit_power_law_spectrum, rapsd_logarithmic
from pysteps.utils.statistics import compute_field_statistics
from pysteps.visualization import basemaps, plot_precip_field
from pysteps.xarray_helpers import geodata_from_dataset

rain_threshold = 1.0  # mm/h
scale_break = 20.0  # km, prescribed rather than estimated
lower_truncation = 1.25  # dBR, separate from the occurrence threshold
seed = 42
num_realizations = 3

###############################################################################
# Read the Swiss radar composite
# ------------------------------
#
# Use the 11 July 2016, 21:00 UTC image from ``plot_noise_generators.py``.
# Select a complete 300 km subdomain in Swiss LV03 coordinates, then aggregate
# to 2 km. Missing radar coverage is not interpreted as dry rainfall.

filename = (
    Path(rcparams.data_sources["mch"]["root_path"])
    / "20160711"
    / "AQC161932100V_00005.801.gif"
)
dataset = io.import_mch_gif(str(filename), product="AQC", unit="mm", accutime=5.0)

# Convert to mm/h, select the domain, and average to a 2 km grid.
dataset = conversion.to_rainrate(dataset)
extent = (450000, 750000, 0, 300000)
dataset = dimension.clip_domain(dataset, extent)
dataset = dimension.aggregate_fields_space(dataset, 2000)
precip_var = dataset.attrs["precip_var"]
rain = dataset[precip_var].values
geodata = geodata_from_dataset(dataset)
if not np.isfinite(rain).all() or np.any(rain < 0):
    raise ValueError("The example requires complete nonnegative rain-rate data")
kmperpixel = abs(float(dataset.x.attrs["stepsize"])) / 1000.0

# Work in dBR = 10*log10(rain rate in mm/h). Extract arrays for the estimators.
transformed_dataset = transformation.dB_transform(
    dataset,
    threshold=rain_threshold,
    zerovalue=10 * np.log10(rain_threshold) - 0.1,
)
observed = transformed_dataset[precip_var].values
threshold = transformed_dataset[precip_var].attrs["threshold"]
zerovalue = transformed_dataset[precip_var].attrs["zerovalue"]

###############################################################################
# Estimate rainfall statistics
# ----------------------------
#
# Calculate the fraction of rainy pixels and their mean and standard deviation.
# Also fit a Normal distribution that accounts for the lower fitting cutoff.
# Use its fitted parameters if they are close to the sample estimates;
# otherwise, use the sample estimates. This is the selection used by the nowcast.
#
# Fit spectral slopes on either side of the prescribed 20 km break to describe
# spatial structure. Convert the break to pixels for this estimator.

statistics = compute_field_statistics(observed, threshold, lower_truncation)
frequency, power = rapsd_logarithmic(observed)
spectrum = fit_power_law_spectrum(frequency, power, scale_break / kmperpixel)
if spectrum["status"] != "ok":
    raise ValueError(f"Unavailable spectral estimate: {spectrum['status']}")
ordinary = np.array([statistics["ordinary_mean"], statistics["ordinary_stdev"]])
latent = np.array([statistics["latent_mean"], statistics["latent_stdev"]])
if not np.isfinite(ordinary).all() or ordinary[1] <= 0:
    raise ValueError("Raining-pixel moments are unavailable")
# Measure the difference between parameter pairs in units of the sample
# standard deviation. The selection limit is configurable in the nowcast.
valid_latent = np.isfinite(latent).all() and latent[1] > 0
moment_distance = (
    np.linalg.norm(latent - ordinary) / ordinary[1] if valid_latent else np.inf
)
use_latent = valid_latent and moment_distance <= 0.5
location, scale = latent if use_latent else ordinary
parameters = dict(
    beta_1=spectrum["beta_1"],
    beta_2=spectrum["beta_2"],
    rain_fraction=statistics["rain_fraction"],
    mean_db=float(location),
    stdev_db=float(scale),
)
print("Ordinary wet-pixel mean/stdev (dBR):", ordinary)
print("Fitted parent-Normal mean/stdev (dBR):", latent)
print("Moment source:", "latent" if use_latent else "ordinary")
print("Generation targets:")
pprint(parameters)

###############################################################################
# Generate calibrated rainfall fields
# -----------------------------------
#
# Generate three fields in dBR using separate seeds for repeatable results.
# Matching rain intensities changes spatial structure, so the generator adjusts
# the input spectral slopes and checks the resulting spectrum. Diagnostics
# report whether each field is close enough to its target slopes.
#
# Also generate a field without the correction to compare the spectra later.
# It uses the first calibrated run's initial seed, although retries may make
# that run select a different random field.

fields, diagnostics = [], []
for member in range(num_realizations):
    field, info = generate_calibrated_field(
        rain.shape,
        parameters,
        kmperpixel,
        scale_break,
        threshold,
        seed=seed + member,
        zerovalue=zerovalue,
        return_diagnostics=True,
    )
    fields.append(field)
    diagnostics.append(info)
    print(
        f"Realization {member + 1}: {info['status']}, "
        f"distance={info['distance']:.3f}, attempts={info['attempts']}, "
        f"selected={info['selected_attempt']}"
    )
uncorrected = generate_stochastic_field(
    rain.shape,
    parameters,
    kmperpixel,
    scale_break,
    threshold,
    seed=seed,
    zerovalue=zerovalue,
)

# Convert to mm/h for plotting. Keep the original dBR arrays for decomposition.
rain_fields = []
for field in fields:
    generated = transformed_dataset.copy(deep=True)
    generated[precip_var].data = field.copy()
    generated = transformation.dB_transform(generated, inverse=True)
    rain_fields.append(generated[precip_var].values)
observed_rain = transformation.dB_transform(transformed_dataset, inverse=True)[
    precip_var
].values

# Plot the observation and generated rainfall fields.
fig = plt.figure(figsize=(10, 8))
for index, (field, title) in enumerate(
    zip(
        [observed_rain, *rain_fields],
        ["Observation", "Realization 1", "Realization 2", "Realization 3"],
    )
):
    ax = basemaps.plot_geography(
        dataset.attrs["projection"], extent, scale="110m", subplot=(2, 2, index + 1)
    )
    plot_precip_field(field, ax=ax, geodata=geodata, axis="off", title=title)
plt.tight_layout()
plt.show()

###############################################################################
# Compare rainfall occurrence and intensity distributions
# -------------------------------------------------------
#
# For each rain-rate threshold, plot the fraction of all pixels at or above it.
# At 1 mm/h, this is the rainy fraction. Distribution matching is approximate,
# so generated fractions and intensities can differ from their targets.

colors = ["tab:blue", "tab:orange", "tab:green"]
plotted_fields = [observed_rain, *rain_fields]
levels = np.geomspace(rain_threshold, max(f.max() for f in plotted_fields), 100)
fig, ax = plt.subplots()
for index, (field, label) in enumerate(
    zip(plotted_fields, ["Observed", "Calibrated 1", "Calibrated 2", "Calibrated 3"])
):
    ordered = np.sort(field.ravel())
    probability = (
        ordered.size - np.searchsorted(ordered, levels, side="left")
    ) / ordered.size
    ax.semilogx(
        levels,
        probability,
        color="black" if index == 0 else colors[index - 1],
        label=label,
    )
ax.set_xlabel("Rain-rate threshold (mm/h)")
ax.set_ylabel("Fraction of pixels at or above threshold")
ax.set_title("Rainfall occurrence and intensity")
ax.legend()
ax.grid(True, alpha=0.25)
plt.show()

###############################################################################
# Compare the realized radial spectra
# -----------------------------------
#
# Compare how variability is distributed across spatial scales. As in the
# calibration check, replace values at or below the rain threshold on a copy.
# The estimator subtracts the mean and divides by the standard deviation before
# averaging FFT power over directions; it does not describe rainfall orientation.

fig, ax = plt.subplots()
ax.loglog(kmperpixel / frequency, power, color="black", label="Observed")
for index, (field, label) in enumerate(
    zip(
        [uncorrected, *fields],
        ["Without correction", "Calibrated 1", "Calibrated 2", "Calibrated 3"],
    )
):
    assessment = field.copy()
    assessment[assessment <= threshold] = zerovalue
    frequencies, powers = rapsd_logarithmic(assessment)
    ax.loglog(
        kmperpixel / frequencies,
        powers,
        color="gray" if index == 0 else colors[index - 1],
        linestyle="--" if index == 0 else "-",
        label=label,
    )
ax.axvline(scale_break, color="black", linestyle=":", label="Prescribed break")
ax.invert_xaxis()
ax.set_xlabel("Wavelength (km)")
ax.set_ylabel("Radial mean FFT power of standardized dBR")
ax.set_title("Spectral assessment after marginal mapping")
ax.legend()
ax.grid(True, alpha=0.25)
plt.show()

###############################################################################
# Separate the generated field into spatial scales
# ------------------------------------------------
#
# A cascade splits the field into bands of spatial scales. Each band is then
# normalized to mean zero and standard deviation one. The nowcast uses these
# normalized bands as random innovations in its autoregressive update.
# Decompose the dBR field directly, retaining its values below the rain threshold.

bandpass = filter_gaussian(fields[0].shape, 6)
cascade = decomposition_fft(fields[0], bandpass, compute_stats=True, normalize=True)
innovations = cascade["cascade_levels"]
print("Innovation means:", innovations.mean(axis=(1, 2)))
print("Innovation standard deviations:", innovations.std(axis=(1, 2)))
fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
for ax, level in zip(axes, [0, 5]):
    image = ax.imshow(innovations[level], cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_title(f"Normalized cascade level {level + 1}")
    ax.set_axis_off()
fig.colorbar(image, ax=axes, label="Standardized anomaly", shrink=0.8)
plt.show()

# sphinx_gallery_thumbnail_number = 1
