"""
Calibrated stochastic rainfall fields
=====================================

This companion to :ref:`sphx_glr_auto_examples_plot_noise_generators.py`
uses the new marginal and
spectral estimators and calibrated generator on the same Swiss radar composite.
It compares generated rainfall distributions and radial spectra, then shows
how the raw fields become normalized cascade innovations.

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
rain, _, metadata = io.import_mch_gif(
    str(filename), product="AQC", unit="mm", accutime=5.0
)
rain, metadata = conversion.to_rainrate(rain, metadata)
extent = (450000, 750000, 0, 300000)
rain, metadata = dimension.clip_domain(rain, metadata, extent)
rain, metadata = dimension.aggregate_fields_space(rain, metadata, 2000)
if not np.isfinite(rain).all() or np.any(rain < 0):
    raise ValueError("The example requires complete nonnegative rain-rate data")
kmperpixel = metadata["xpixelsize"] / 1000.0
observed, transformed_metadata = transformation.dB_transform(
    rain,
    metadata,
    threshold=rain_threshold,
    zerovalue=10 * np.log10(rain_threshold) - 0.1,
)
threshold = transformed_metadata["threshold"]
zerovalue = transformed_metadata["zerovalue"]

###############################################################################
# Estimate marginal parameters and prescribed-break spectral slopes
# -----------------------------------------------------------------
#
# Ordinary moments describe retained wet pixels; latent moments describe the
# fitted parent Normal. The reference engine selects the latent pair when its
# normalized distance from the ordinary pair is at most 0.5, otherwise the
# ordinary pair. Make that caller decision explicit here: the generator itself
# does not select moments or apply regional parameter bounds.
#
# Radial frequencies are in cycles per pixel, so convert the 20 km break to
# pixels for fitting. Pass physical km units to the generator later.

statistics = compute_field_statistics(observed, threshold, lower_truncation)
frequency, power = rapsd_logarithmic(observed)
spectrum = fit_power_law_spectrum(frequency, power, scale_break / kmperpixel)
if spectrum["status"] != "ok":
    raise ValueError(f"Unavailable spectral estimate: {spectrum['status']}")
ordinary = np.array([statistics["ordinary_mean"], statistics["ordinary_stdev"]])
latent = np.array([statistics["latent_mean"], statistics["latent_stdev"]])
if not np.isfinite(ordinary).all() or ordinary[1] <= 0:
    raise ValueError("Raining-pixel moments are unavailable")
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
# The outputs are float64 fields in dBR retaining a rainfall marginal, including
# continuous dry anomalies. They are not globally standardized noise. Calibration
# applies the reference -0.5 correction to both slopes, assesses realized slopes
# on a thresholded copy, and reports whether its distance tolerance was met.
# A best-available field can be returned without meeting that tolerance.
#
# Use three reproducible local seeds and retain their diagnostics. Also generate
# an uncorrected field for the spectrum comparison. Its seed matches the first
# calibrated run's initial draw; retries can select a different candidate.

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

# Convert copies for rainfall plots. Keep the raw dBR arrays for decomposition.
rain_fields = [
    transformation.dB_transform(f, metadata=transformed_metadata, inverse=True)[0]
    for f in fields
]
observed_rain = transformation.dB_transform(
    observed, metadata=transformed_metadata, inverse=True
)[0]
fig = plt.figure(figsize=(10, 8))
for index, (field, title) in enumerate(
    zip(
        [observed_rain, *rain_fields],
        ["Observation", "Realization 1", "Realization 2", "Realization 3"],
    )
):
    ax = basemaps.plot_geography(
        metadata["projection"], extent, scale="110m", subplot=(2, 2, index + 1)
    )
    plot_precip_field(field, ax=ax, geodata=metadata, axis="off", title=title)
plt.tight_layout()
plt.show()

###############################################################################
# Compare rainfall occurrence and intensity distributions
# -------------------------------------------------------
#
# Unconditional exceedance curves include all pixels in their denominator.
# Their value at 1 mm/h measures realized wet fraction. Histogram mapping and
# threshold crossings mean that occurrence and marginal moments are approximate;
# fitted parent-Normal moments are not the observed conditional moments.
# The generator's Normal-quantile clipping also limits its extreme tails.

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
# Follow the calibration assessment: replace values <= the dBR threshold by
# the dry value on a copy before estimating each generated spectrum. The radial
# estimator standardizes its input internally; the plotted quantity is mean
# squared unscaled FFT amplitude per mode, not a physical spectral density.
# A radial spectrum does not preserve the observation's directional structure.

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
# Decompose the raw field into normalized cascade innovations
# -----------------------------------------------------------
#
# The nowcast engine decomposes the raw field, retaining its continuous dry
# anomalies. Normalization happens at each cascade level, rather than treating
# the rainfall-marginal field as globally standardized perturbation noise.

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
