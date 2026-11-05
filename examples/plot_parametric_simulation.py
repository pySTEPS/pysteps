"""
Simulate a time series of rainfall fields
=========================================

Generate rainfall patterns that move and change with time using prescribed
rainfall statistics. This example sets the rain fraction, intensity distribution,
spatial structure, and temporal persistence explicitly. It runs without radar
observations.

The correlation time is 400 minutes at a wavelength of 256 km. Smaller rainfall
features have shorter correlation times and change more quickly. A uniform
motion field moves the rain one pixel east every five minutes.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.noise.parametric import generate_calibrated_field
from pysteps.nowcasts.parametric import simulate
from pysteps.timeseries.parametric import (
    ar2_correlation_time,
    estimate_reference_correlation_time,
    lagrangian_autocorrelation,
    model_cascade_correlations,
)
from pysteps.utils.spectral import fit_power_law_spectrum, rapsd_logarithmic
from pysteps.utils.statistics import compute_field_statistics
from pysteps.utils.transformation import dB_transform

###############################################################################
# Choose rainfall and temporal parameters
# ---------------------------------------
#
# Describe the rainy fraction and intensity distribution in dBR, where
# dBR = 10*log10(rain rate in mm/h). The two spectral slopes describe spatial
# variability on either side of the 20 km break. These parameters stay fixed;
# the positions and intensities of individual rainfall features evolve.

shape = (128, 128)
kmperpixel = 1.0
timestep = 5.0  # minutes
n_cascade_levels = 6
rain_threshold = 1.0  # mm/h
scale_break = 20.0  # km
parameters = dict(
    beta_1=-2.4,
    beta_2=-3.1,
    rain_fraction=0.2,
    mean_db=6.0,
    stdev_db=3.0,
)
reference_time = 400.0  # minutes
reference_wavelength = 256.0  # km
temporal_exponent = 0.81
alpha = 2.72

###############################################################################
# Calculate lag correlations at each spatial scale
# ------------------------------------------------
#
# Separate the field into six bands of spatial scales, called cascade levels.
# Assign a correlation time to each level using
# T = 400 * (wavelength / 256)**0.81, with wavelengths in km and times in minutes.
# The temporal helper finds lag-1 and lag-2 correlations for each time using
# rho2 = rho1**alpha, adjusted where necessary for a nonperiodic AR(2) process.
# Correlation time here is a discrete autocorrelation sum including lag zero.

bandpass = filter_gaussian(shape, n_cascade_levels)
wavelengths = kmperpixel / bandpass["central_freqs"]
lag_correlations, temporal_diagnostics = model_cascade_correlations(
    reference_time,
    wavelengths,
    temporal_exponent,
    alpha,
    timestep,
    reference_wavelength=reference_wavelength,
    return_diagnostics=True,
)
print("Prescribed temporal model (wavelength in km, time in minutes):")
print(f"{'Wavelength':>12} {'Time':>12} {'Lag-1':>12} {'Lag-2':>12}")
for row in np.column_stack(
    [wavelengths, temporal_diagnostics["requested_times"], lag_correlations]
):
    print(" ".join(f"{value:12.4f}" for value in row))

###############################################################################
# Construct an initial history and uniform motion
# -----------------------------------------------
#
# Generate one random rainfall field, then create two earlier fields by shifting
# it west. np.roll wraps the array edges to construct this artificial history.
# The simulation itself uses open boundaries, with random variability added at
# inflow. Shifted copies describe pure translation; the initial history is not
# a stationary realization of the prescribed temporal model.

transformed = generate_calibrated_field(
    shape,
    parameters,
    kmperpixel,
    scale_break,
    threshold=10 * np.log10(rain_threshold),
    seed=9,
)
rain = 10 ** (transformed / 10)
rain[rain < rain_threshold] = 0
initial_fields = np.stack([np.roll(rain, -2, axis=1), np.roll(rain, -1, axis=1), rain])

# Attach physical coordinates and the supplied velocity. Rainfall is in mm/h;
# velocity is in pixels per interval, and the time stepsize attribute is seconds.
dataset = xr.Dataset(
    {
        "rain": (("time", "y", "x"), initial_fields, {"units": "mm/h"}),
        "velocity_x": (("y", "x"), np.ones(shape), {"units": "pixels/timestep"}),
        "velocity_y": (("y", "x"), np.zeros(shape), {"units": "pixels/timestep"}),
    },
    coords={
        "time": (
            "time",
            np.array(
                ["2026-01-01T00:00", "2026-01-01T00:05", "2026-01-01T00:10"],
                dtype="datetime64[s]",
            ),
            {"stepsize": timestep * 60},
        ),
        "x": (
            "x",
            (np.arange(shape[1]) + 0.5) * kmperpixel,
            {"units": "km", "stepsize": kmperpixel},
        ),
        "y": (
            "y",
            (np.arange(shape[0]) + 0.5) * kmperpixel,
            {"units": "km", "stepsize": kmperpixel},
        ),
    },
    attrs={"precip_var": "rain"},
)

###############################################################################
# Simulate one hour of rainfall
# -----------------------------
#
# simulate() computes the same lag correlations shown above, then converts them
# to AR coefficients and innovation weights. At each step, it advects the two
# previous cascade states and combines them with a new random field. Rainfall
# distribution matching and boundary tapering feed into the next state.
# The supplied parameters and velocity are used throughout, without estimation.

ensemble, diagnostics = simulate(
    dataset,
    timesteps=12,
    parameters=parameters,
    reference_time=reference_time,
    reference_wavelength=reference_wavelength,
    temporal_exponent=temporal_exponent,
    alpha=alpha,
    n_cascade_levels=n_cascade_levels,
    rain_threshold=rain_threshold,
    scale_break=scale_break,
    n_ens_members=1,
    seed=42,
    return_diagnostics=True,
)
simulated_rain = ensemble[ensemble.attrs["precip_var"]].isel(ens_number=0)

###############################################################################
# Recover parameters from the simulated rainfall
# ----------------------------------------------
#
# Estimate statistics from all twelve generated fields of the first realization.
# Exclude the initial field and its shifted copies. Use the returned rain rates,
# transformed back to dBR, rather than the internal cascade states or generation
# diagnostics. Dry output pixels receive the same transformed dry value as the
# initial data.
#
# Estimate the fraction of rainy pixels and fit the parent Normal distribution
# above 1.25 dBR. Its fitted location and scale can be compared with mean_db and
# stdev_db. Ordinary wet-pixel moments describe the truncated sample and are
# different quantities. Fit both spectral slopes with the prescribed break.

threshold_db = 10 * np.log10(rain_threshold)
lower_truncation = 1.25  # dBR, fitting cutoff distinct from the rain threshold
transformed_ensemble = dB_transform(
    ensemble, threshold=rain_threshold, zerovalue=threshold_db - 0.1
)
validation_fields = (
    transformed_ensemble[ensemble.attrs["precip_var"]].isel(ens_number=0).values
)
field_estimates = []
marginal_statuses = []
spectral_statuses = []
cascades = []
for field in validation_fields:
    statistics = compute_field_statistics(field, threshold_db, lower_truncation)
    frequency, power = rapsd_logarithmic(field)
    spectrum = fit_power_law_spectrum(
        frequency, power, scale_break=scale_break / kmperpixel
    )
    field_estimates.append(
        [
            statistics["rain_fraction"],
            statistics["latent_mean"],
            statistics["latent_stdev"],
            spectrum["beta_1"],
            spectrum["beta_2"],
        ]
    )
    marginal_statuses.append(statistics["nonzero_fit"]["status"])
    spectral_statuses.append(spectrum["status"])
    cascades.append(
        decomposition_fft(field, bandpass, compute_stats=True, normalize=True)[
            "cascade_levels"
        ]
    )
field_estimates = np.asarray(field_estimates)
cascades = np.stack(cascades)

# Show variability between fields alongside the target. These fields are
# temporally correlated: their standard deviation is not a confidence interval.
# Retain unavailable estimates as NaN and report finite counts; do not replace
# them with the targets or silently discard failed fits.
labels = (
    "Rain fraction",
    "Normal location (dBR)",
    "Normal scale (dBR)",
    "Large-scale slope",
    "Small-scale slope",
)
targets = [
    parameters[name]
    for name in ("rain_fraction", "mean_db", "stdev_db", "beta_1", "beta_2")
]
print(f"\nParameter recovery from {len(validation_fields)} fields of one realization:")
print(
    f"{'Parameter':<23} {'Target':>12} {'Mean':>12} {'SD':>12} {'Finite / total':>15}"
)
for label, target, values in zip(labels, targets, field_estimates.T):
    count = np.count_nonzero(np.isfinite(values))
    print(
        f"{label:<23} {target:12.4f} {np.mean(values):12.4f} "
        f"{np.std(values):12.4f} {str(count) + '/' + str(len(values)):>15}"
    )
for name, statuses in [
    ("Marginal", marginal_statuses),
    ("Spectral", spectral_statuses),
]:
    for status in sorted(set(statuses)):
        print(f"{name} fit status {status}: {statuses.count(status)}/{len(statuses)}")

###############################################################################
# Assess noisy spectral estimates with Monte Carlo sampling
# ---------------------------------------------------------
#
# A spectral slope fitted to one field can be noisy, especially at large scales
# represented by few Fourier modes. Use a Monte Carlo comparison to assess the
# spectral targets: generate independent initial fields and simulate independent
# one-hour realizations with the same parameters.
#
# Average the fitted slopes over the twelve fields within each run, then report
# the mean, standard deviation, and 5th/95th percentiles across independent runs.
# This treats each run as one replicate; its correlated frames are not separate
# independent samples. The percentile range describes the spread of run means,
# not a confidence interval for the overall mean. Calibration and nonlinear
# evolution can still leave systematic differences from the target slopes.

n_monte_carlo = 32
monte_carlo_slopes = []
for run_seed in np.random.SeedSequence(2026).spawn(n_monte_carlo):
    initial_seed, evolution_seed = run_seed.spawn(2)
    mc_transformed = generate_calibrated_field(
        shape, parameters, kmperpixel, scale_break, threshold_db, seed=initial_seed
    )
    mc_rain = 10 ** (mc_transformed / 10)
    mc_rain[mc_rain < rain_threshold] = 0
    mc_history = dataset.copy(deep=True)
    mc_history[dataset.attrs["precip_var"]].data = np.stack(
        [np.roll(mc_rain, -2, axis=1), np.roll(mc_rain, -1, axis=1), mc_rain]
    )
    mc_ensemble = simulate(
        mc_history,
        timesteps=len(validation_fields),
        parameters=parameters,
        reference_time=reference_time,
        reference_wavelength=reference_wavelength,
        temporal_exponent=temporal_exponent,
        alpha=alpha,
        n_cascade_levels=n_cascade_levels,
        rain_threshold=rain_threshold,
        scale_break=scale_break,
        n_ens_members=1,
        seed=evolution_seed,
    )
    mc_fields = (
        dB_transform(
            mc_ensemble, threshold=rain_threshold, zerovalue=threshold_db - 0.1
        )[ensemble.attrs["precip_var"]]
        .isel(ens_number=0)
        .values
    )
    mc_slopes = []
    for mc_field in mc_fields:
        frequency, power = rapsd_logarithmic(mc_field)
        spectrum = fit_power_law_spectrum(
            frequency, power, scale_break=scale_break / kmperpixel
        )
        mc_slopes.append([spectrum["beta_1"], spectrum["beta_2"]])
    # A failed field fit leaves its run mean unavailable; do not silently drop it.
    monte_carlo_slopes.append(np.mean(mc_slopes, axis=0))
monte_carlo_slopes = np.asarray(monte_carlo_slopes)
print(f"\nMonte Carlo spectral recovery: {n_monte_carlo} independent runs")
print(
    f"{'Parameter':<23} {'Target':>12} {'MC mean':>12} {'MC SD':>12} "
    f"{'P05':>12} {'P95':>12} {'Finite runs':>12}"
)
for label, name, values in zip(
    ["Large-scale slope", "Small-scale slope"],
    ["beta_1", "beta_2"],
    monte_carlo_slopes.T,
):
    lower, upper = np.percentile(values, [5, 95])
    count = np.count_nonzero(np.isfinite(values))
    print(
        f"{label:<23} {parameters[name]:12.4f} {np.mean(values):12.4f} "
        f"{np.std(values):12.4f} {lower:12.4f} {upper:12.4f} "
        f"{str(count) + '/' + str(n_monte_carlo):>12}"
    )

###############################################################################
# Compare prescribed and estimated temporal correlations
# ------------------------------------------------------
#
# Align each overlapping three-field history with the known velocity and
# estimate its spatial lag-1 and lag-2 correlations at every cascade level.
# Use the same zero-anomaly boundary convention as the forecast estimator.
# Average the lag pairs, then convert that average to an implied AR(2) time.
# These are estimates from the displayed realization, separate from the Monte
# Carlo spectral check. One hour is short relative to the 400-minute reference
# time, and shifted initial fields introduce a transient. Distribution matching
# and boundary feedback can also change the measured correlations.

velocity = np.stack([dataset.velocity_x.values, dataset.velocity_y.values])
measured_correlations = np.stack(
    [
        lagrangian_autocorrelation(cascades[end - 2 : end + 1], velocity)
        for end in range(2, len(cascades))
    ]
)
average_correlations = np.mean(measured_correlations, axis=0)
estimated_times = ar2_correlation_time(
    average_correlations[:, 0], average_correlations[:, 1], timestep
)
print(
    f"\nTemporal recovery from {len(measured_correlations)} overlapping histories "
    "(km and minutes):"
)
print(
    f"{'Wavelength':>12} {'Model T':>12} {'Implied T':>12} "
    f"{'Model lag-1':>12} {'Est. lag-1':>12} "
    f"{'Model lag-2':>12} {'Est. lag-2':>12}"
)
for row in np.column_stack(
    [
        wavelengths,
        temporal_diagnostics["achieved_times"],
        estimated_times,
        lag_correlations[:, 0],
        average_correlations[:, 0],
        lag_correlations[:, 1],
        average_correlations[:, 1],
    ]
):
    print(" ".join(f"{value:12.4f}" for value in row))

# Rescale the first three level times to 256 km with the prescribed exponent.
# All three must be valid; the estimator returns NaN rather than dropping one.
# The spectral break, temporal exponent, and alpha are held fixed in this check,
# not independently recovered from the simulation.
estimated_reference_time = estimate_reference_correlation_time(
    estimated_times,
    wavelengths,
    temporal_exponent,
    levels=(0, 1, 2),
    reference_wavelength=reference_wavelength,
)
print(
    f"\n{'Reference scale (km)':<23} {'Target time (min)':>18} {'Implied time (min)':>19}"
)
print(
    f"{reference_wavelength:<23.4f} {reference_time:18.4f} {estimated_reference_time:19.4f}"
)

# The implied times are diagnostic extrapolations from two lags, rather than
# direct measurements of long correlation times. Longer runs and a separate
# Monte Carlo temporal assessment are needed to assess long-run persistence.

###############################################################################
# Show the evolution of one realization
# -------------------------------------
#
# Plot the initial field and five times from the same realization. Use a common
# colour scale to see the translation and changes in rainfall structure.

steps = (0, 1, 3, 6, 9, 12)
fig, axes = plt.subplots(
    2, 3, figsize=(11, 7), sharex=True, sharey=True, constrained_layout=True
)
for ax, step in zip(axes.flat, steps):
    field = rain if step == 0 else simulated_rain.isel(time=step - 1).values
    image = ax.imshow(
        field,
        origin="lower",
        vmin=0,
        vmax=20,
        cmap="Blues",
        extent=(0, shape[1] * kmperpixel, 0, shape[0] * kmperpixel),
    )
    ax.set_title("Initial field" if step == 0 else f"+{step * timestep:g} min")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
fig.colorbar(image, ax=axes, label="Rain rate (mm/h)", shrink=0.8)
plt.show()
