"""
pysteps.nowcasts.parametric
===========================

Generate parametric ensemble nowcasts from three rainfall observations.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from pysteps import motion
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade.decomposition import decomposition_fft, recompose_fft
from pysteps.noise.parametric import generate_calibrated_field, GENERATOR_ID
from pysteps.postprocessing.probmatching import match_rainfall_distribution
from pysteps.timeseries.autoregression import estimate_ar_params_yw
from pysteps.timeseries.parametric import (
    ar2_correlation_time,
    estimate_reference_correlation_time,
    lagrangian_autocorrelation,
    model_cascade_correlations,
)
from pysteps.utils.spectral import fit_power_law_spectrum, rapsd_logarithmic
from pysteps.utils.statistics import compute_field_statistics
from pysteps.utils.transformation import dB_transform
from pysteps.xarray_helpers import convert_output_to_xarray_dataset

from ._cascade_advection import advect_field, prepare_advection_coordinates

# Configurable estimation defaults. Low-rain-fraction replacements are regional
# values from Auckland rainfall data; see the parametric nowcasting user guide.
_ESTIMATION_DEFAULTS = dict(
    lower_truncation=1.25,
    temporal_exponent=0.81,
    alpha=2.72,
    reference_levels=(0,),
    reference_time=None,
    moment_distance_limit=0.5,
    beta_1_bounds=(-3.9, float(np.nextafter(-1.25, -np.inf))),
    beta_2_min=-4.0,
    minimum_beta_gap=0.1,
    correlation_time_bounds=(60.0, 720.0),
    low_rain_fraction=0.05,
    fallback_beta_1=-1.45,
    fallback_beta_2=-3.253958088759979,
    fallback_reference_time=330.61752959848894,
)
_MOTION_DEFAULTS = dict(
    lk_kwargs=dict(winsize=(31, 31), nr_levels=3, min_eig_thr=3e-5),
    fd_method="shitomasi",
    fd_kwargs=dict(max_corners=400, quality_level=0.025, min_distance=5),
    interp_method="idwinterp2d",
    interp_kwargs=dict(k=8, power=1.25),
    nr_std_outlier=3.5,
    k_outlier=20,
    decl_scale=8,
)
_NOISE_DEFAULTS = dict(
    spectral_bias=-0.5,
    low_war_threshold=0.1,
    max_attempts=5,
    distance_threshold=1.0,
    beta_variances=(0.067738, 0.076405),
    nbins=500,
    eps=0.001,
)


def _finite(name, value):
    """Validate and return a finite real scalar."""
    if not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        valid = np.isfinite(value)
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _positive(name, value):
    """Validate and return a positive real scalar."""
    value = _finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _integer(name, value, minimum=1):
    """Validate an integer at or above the specified minimum."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _options(name, supplied, defaults):
    """Copy defaults and apply overrides, rejecting unknown setting names."""
    if supplied is None:
        return deepcopy(defaults)
    if not isinstance(supplied, Mapping):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name}: {sorted(unknown)}")
    result = deepcopy(defaults)
    result.update(deepcopy(dict(supplied)))
    return result


def _estimation_settings(supplied, threshold, n_levels):
    """Validate estimation settings against the threshold and cascade count."""
    settings = _options("estimation_kwargs", supplied, _ESTIMATION_DEFAULTS)
    for key in settings:
        if key not in (
            "reference_levels",
            "reference_time",
            "beta_1_bounds",
            "correlation_time_bounds",
        ):
            settings[key] = _finite(key, settings[key])
    for key in ("alpha", "minimum_beta_gap", "fallback_reference_time"):
        _positive(key, settings[key])
    if settings["moment_distance_limit"] < 0:
        raise ValueError("moment_distance_limit must be nonnegative")
    if not 0 <= settings["low_rain_fraction"] <= 1:
        raise ValueError("low_rain_fraction must lie in [0, 1]")
    if settings["lower_truncation"] < threshold:
        raise ValueError("lower_truncation must be at or above the dBR threshold")
    for key in ("beta_1_bounds", "correlation_time_bounds"):
        bounds = np.asarray(settings[key], float)
        if (
            bounds.shape != (2,)
            or not np.isfinite(bounds).all()
            or bounds[0] > bounds[1]
        ):
            raise ValueError(f"{key} must be two finite increasing bounds")
        settings[key] = tuple(float(v) for v in bounds)
    if settings["correlation_time_bounds"][0] <= 0:
        raise ValueError("correlation_time_bounds must be positive")
    if (
        settings["beta_2_min"]
        > settings["beta_1_bounds"][0] - settings["minimum_beta_gap"]
    ):
        raise ValueError("beta_2_min and beta_1_bounds cannot satisfy minimum_beta_gap")
    levels = np.asarray(settings["reference_levels"])
    if (
        levels.ndim != 1
        or not levels.size
        or not np.issubdtype(levels.dtype, np.integer)
        or np.any(levels < 0)
        or np.any(levels >= n_levels)
        or np.unique(levels).size != levels.size
    ):
        raise ValueError("reference_levels must contain distinct valid level indices")
    settings["reference_levels"] = tuple(int(v) for v in levels)
    if settings["reference_time"] is not None:
        settings["reference_time"] = _positive(
            "reference_time", settings["reference_time"]
        )
    return settings


def _noise_settings(supplied):
    """Validate spectral-calibration and distribution-matching settings."""
    settings = _options("noise_kwargs", supplied, _NOISE_DEFAULTS)
    for key in ("spectral_bias", "low_war_threshold", "distance_threshold", "eps"):
        settings[key] = _finite(key, settings[key])
    if (
        not 0 <= settings["low_war_threshold"] <= 1
        or settings["distance_threshold"] < 0
    ):
        raise ValueError(
            "noise fractions must lie in [0, 1] and distance must be nonnegative"
        )
    if not 0 < settings["eps"] < 0.5:
        raise ValueError("noise eps must lie in (0, 0.5)")
    settings["max_attempts"] = _integer("max_attempts", settings["max_attempts"])
    settings["nbins"] = _integer("nbins", settings["nbins"], 2)
    variances = np.asarray(settings["beta_variances"], float)
    if (
        variances.shape != (2,)
        or not np.isfinite(variances).all()
        or np.any(variances <= 0)
    ):
        raise ValueError("beta_variances must contain two positive finite values")
    settings["beta_variances"] = tuple(float(v) for v in variances)
    return settings


def _decompose(field, bp_filter):
    """Normalize each cascade level, using zero anomalies for constant levels."""
    result = decomposition_fft(field, bp_filter, compute_stats=True, normalize=False)
    levels = result["cascade_levels"]
    if np.ptp(field) == 0:
        levels.fill(0)
        result["stds"] = [0.0] * len(levels)
        result["normalized"] = True
        return result
    for k, (mean, stdev) in enumerate(zip(result["means"], result["stds"])):
        levels[k] = (levels[k] - mean) / stdev if stdev > 0 else 0
    result["normalized"] = True
    return result


def _generation_parameters(statistics, spectrum, reference_time, settings):
    """Select intensity parameters and constrain slopes and persistence.

    Prefer fitted Normal parameters when close to the wet-pixel sample moments.
    For low rain fractions, replace both spectral slopes and reference time
    together. Apply the slope and time bounds afterwards, retaining the raw
    estimates and each selection in the returned audit dictionary.
    """
    mu, sigma = statistics["ordinary_mean"], statistics["ordinary_stdev"]
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            "Cannot estimate rainfall marginal: newest field needs at least two "
            "distinct raining values; constant/single-pixel rain is not dry"
        )
    latent = np.array([statistics["latent_mean"], statistics["latent_stdev"]])
    valid = np.isfinite(latent).all() and latent[1] > 0
    distance = (
        float(np.hypot(latent[0] - mu, latent[1] - sigma) / sigma) if valid else None
    )
    use_latent = valid and distance <= settings["moment_distance_limit"]
    selected = dict(
        rain_fraction=statistics["rain_fraction"],
        mean_db=float(latent[0] if use_latent else mu),
        stdev_db=float(latent[1] if use_latent else sigma),
        beta_1=spectrum["beta_1"],
        beta_2=spectrum["beta_2"],
        reference_time=reference_time,
    )
    original = selected.copy()
    low_war = selected["rain_fraction"] < settings["low_rain_fraction"]
    if low_war:
        selected.update(
            beta_1=settings["fallback_beta_1"],
            beta_2=settings["fallback_beta_2"],
            reference_time=settings["fallback_reference_time"],
        )
    if not all(
        np.isfinite(selected[k]) for k in ("beta_1", "beta_2", "reference_time")
    ):
        raise ValueError(
            "Unavailable spectral/persistence estimates outside the low-rainfall "
            "fallback; inspect observations, scale_break, or reference_time"
        )
    if selected["reference_time"] <= 0:
        raise ValueError("Estimated reference_time must be positive")
    before_bounds = selected.copy()
    selected["beta_1"] = float(np.clip(selected["beta_1"], *settings["beta_1_bounds"]))
    selected["beta_2"] = float(
        np.clip(
            selected["beta_2"],
            settings["beta_2_min"],
            selected["beta_1"] - settings["minimum_beta_gap"],
        )
    )
    selected["reference_time"] = float(
        np.clip(selected["reference_time"], *settings["correlation_time_bounds"])
    )
    audit = dict(
        moment_source="latent" if use_latent else "ordinary",
        moment_distance=distance,
        moment_reason=(
            "within_limit"
            if use_latent
            else ("distance" if valid else "invalid_latent")
        ),
        low_rain_fraction_fallback=low_war,
        before_safeguards=original,
        before_bounds=before_bounds,
        bounded_parameters=[k for k in selected if selected[k] != before_bounds[k]],
    )
    return selected, audit


def _advance(
    histories,
    parameters,
    phi,
    bp_filter,
    coordinates,
    rng,
    threshold,
    zerovalue,
    kmperpixel,
    scale_break,
    noise_settings,
    min_rain_fraction,
):
    """Advance one member by one observation interval.

    Parameters
    ----------
    histories: sequence of dict
        Two normalized cascade decompositions, newest first. Each contains
        cascade_levels with shape (level, y, x). Histories are not modified.
    parameters: mapping
        Selected rain fraction, Normal intensity parameters, and spectral slopes.
    phi: ndarray
        Array of shape (level, 3): lag-1 coefficient, lag-2 coefficient, and
        innovation coefficient returned by estimate_ar_params_yw.
    bp_filter: dict
        Gaussian cascade filter used for both histories and random fields.
    coordinates: dict
        Backward sampling coordinates for lags 1 and 2, in (y, x) order.
    rng: numpy.random.Generator
        Member-specific random stream, advanced in place.
    threshold, zerovalue: float
        Rain occurrence threshold and dry value in dBR.
    kmperpixel, scale_break: float
        Pixel spacing and spectral break wavelength in km.
    noise_settings: dict
        Calibration and histogram-mapping options.
    min_rain_fraction: float
        Target fraction below which distribution matching returns dry output.

    Returns
    -------
    field: ndarray
        Forecast field in dBR with shape (y, x), before boundary tapering.
    diagnostics: dict
        Calibration results for the random field used in this update.
    """
    # Generate rainfall variability, then normalize its cascade levels to obtain
    # the innovations used in the autoregressive update.
    noise, diagnostics = generate_calibrated_field(
        bp_filter["shape"],
        parameters,
        kmperpixel,
        scale_break,
        threshold,
        rng=rng,
        zerovalue=zerovalue,
        return_diagnostics=True,
        **noise_settings,
    )
    innovation = _decompose(noise, bp_filter)
    levels = np.empty_like(innovation["cascade_levels"])
    # Bring both previous states to the forecast time before the AR(2) update.
    # Zero anomalies outside the grid allow innovations to populate inflow.
    for k in range(len(levels)):
        levels[k] = (
            phi[k, 0] * advect_field(histories[0]["cascade_levels"][k], coordinates[1])
            + phi[k, 1]
            * advect_field(histories[1]["cascade_levels"][k], coordinates[2])
            + phi[k, 2] * innovation["cascade_levels"][k]
        )
    # Normalize over all levels and spatial pixels together. This couples the
    # levels and differs from normalizing each updated level separately.
    stdev = np.std(levels)
    if stdev > 0.1:
        levels = (levels - np.mean(levels)) / stdev
    updated = dict(
        domain="spatial",
        normalized=True,
        compact_output=False,
        cascade_levels=levels,
        means=innovation["means"],
        stds=innovation["stds"],
    )
    # Recompose with the innovation moments and restore the target rainfall
    # distribution. The caller feeds this adjusted field into the next update.
    field = match_rainfall_distribution(
        recompose_fft(updated),
        parameters["rain_fraction"],
        parameters["mean_db"],
        parameters["stdev_db"],
        threshold,
        zerovalue=zerovalue,
        min_rain_fraction=min_rain_fraction,
        nbins=noise_settings["nbins"],
        eps=noise_settings["eps"],
    )
    return field, diagnostics


def _velocity_from_dataset(dataset):
    """Read fixed x/y displacements as an array with shape (2, y, x).

    Return None if both velocity variables are absent. Otherwise require both
    components, finite values, and dimensions (y, x) in either order. Values
    are displacements in pixels per observation interval. Extract arrays before
    stacking to avoid automatic xarray coordinate alignment.
    """
    names = ("velocity_x", "velocity_y")
    present = [name in dataset.data_vars for name in names]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("Supply both velocity_x and velocity_y, or neither")
    components = []
    for name in names:
        field = dataset[name]
        if set(field.dims) != {"y", "x"} or field.ndim != 2:
            raise ValueError(f"{name} must have dimensions (y, x)")
        if field.attrs.get("units") not in (
            None,
            "pixels/timestep",
            "pixels / timestep",
        ):
            raise ValueError(f"{name} must be in pixels per observation timestep")
        values = field.transpose("y", "x").values
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError(f"{name} must contain finite real values")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite real values")
        components.append(values)
    return np.array(components, dtype=float, copy=True)


def _prepare_dataset(dataset, kmperpixel, timestep):
    """Validate the grid and copy observations into (time, y, x) order.

    Require three finite rain-rate fields, regularly spaced observation times,
    and a regular grid with square pixels. Coordinate stepsize attributes must
    match the actual spacing. Return the copied dataset, optional velocity,
    pixel size in km, and observation interval in minutes. Explicit pixel-size
    or interval overrides must agree with the coordinates.
    """
    if not isinstance(dataset, xr.Dataset):
        raise ValueError("dataset must be an xarray.Dataset")
    precip_var = dataset.attrs.get("precip_var")
    if not isinstance(precip_var, str) or precip_var not in dataset.data_vars:
        raise ValueError(
            "dataset.attrs['precip_var'] must name a precipitation variable"
        )
    precip = dataset[precip_var]
    if precip.ndim != 3 or set(precip.dims) != {"time", "y", "x"}:
        raise ValueError("Precipitation must have dimensions (time, y, x)")
    precip = precip.transpose("time", "y", "x")
    if (
        precip.shape[0] != 3
        or min(precip.shape[1:]) < 2
        or max(precip.shape[1:]) < 3
        or not np.issubdtype(precip.dtype, np.floating)
    ):
        raise ValueError("Precipitation must be floating point with shape (3, y, x)")
    if precip.attrs.get("units") != "mm/h" or precip.attrs.get("transform") is not None:
        raise ValueError("Precipitation must be untransformed rain rates in mm/h")
    rain = precip.values
    if not np.isfinite(rain).all() or np.any(rain < 0):
        raise ValueError(
            "Precipitation must contain only finite nonnegative rain rates"
        )

    spacings = []
    for axis in ("x", "y"):
        if axis not in dataset.coords or dataset[axis].dims != (axis,):
            raise ValueError(f"A one-dimensional {axis} coordinate is required")
        coord = dataset[axis]
        if not np.issubdtype(coord.dtype, np.number) or np.iscomplexobj(coord):
            raise ValueError(f"{axis} coordinates must be finite real numbers")
        values = np.asarray(coord.values, dtype=float)
        differences = np.diff(values)
        step = differences[0]
        if (
            not np.isfinite(values).all()
            or step == 0
            or not np.allclose(differences, step, rtol=1e-7, atol=abs(step) * 1e-10)
        ):
            raise ValueError(
                f"{axis} coordinates must be regularly spaced and monotonic"
            )
        declared = _finite(f"{axis}.stepsize", coord.attrs.get("stepsize"))
        if not np.isclose(declared, step, rtol=1e-7, atol=abs(step) * 1e-10):
            raise ValueError(f"{axis}.stepsize disagrees with coordinate spacing")
        units = coord.attrs.get("units")
        if units not in ("m", "km"):
            raise ValueError(f"{axis} coordinate units must be 'm' or 'km'")
        spacings.append(abs(step) / (1000.0 if units == "m" else 1.0))
    if not np.isclose(*spacings, rtol=1e-7, atol=0):
        raise ValueError("The parametric forecast requires square pixels")

    if "time" not in dataset.coords or dataset.time.dims != ("time",):
        raise ValueError("A one-dimensional datetime time coordinate is required")
    times = dataset.time.values
    if not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any():
        raise ValueError("time must contain valid datetime64 observation times")
    intervals = np.diff(times) / np.timedelta64(1, "s")
    seconds = intervals[0]
    if seconds <= 0 or not np.allclose(intervals, seconds, rtol=1e-9, atol=1e-9):
        raise ValueError("Observation times must be increasing and regularly spaced")
    declared = _positive("time.stepsize", dataset.time.attrs.get("stepsize"))
    if not np.isclose(declared, seconds, rtol=1e-9, atol=1e-9):
        raise ValueError("time.stepsize must match observation spacing in seconds")
    derived_timestep = float(seconds / 60.0)
    for name, supplied, derived in (
        ("kmperpixel", kmperpixel, spacings[0]),
        ("timestep", timestep, derived_timestep),
    ):
        if supplied is not None and not np.isclose(
            _positive(name, supplied), derived, rtol=1e-7, atol=0
        ):
            raise ValueError(f"{name} disagrees with dataset coordinates")

    velocity = _velocity_from_dataset(dataset)
    prepared = dataset.copy(deep=True)
    prepared[precip_var] = precip.copy(deep=True).astype(float)
    return prepared, velocity, float(spacings[0]), derived_timestep


def forecast(
    dataset,
    timesteps=1,
    *,
    n_ens_members=24,
    kmperpixel=None,
    timestep=None,
    n_cascade_levels=6,
    rain_threshold=1.0,
    scale_break=20.0,
    zerovalue=None,
    taper_sigma=5.0,
    min_rain_fraction=0.025,
    seed=None,
    motion_kwargs=None,
    estimation_kwargs=None,
    temporal_fallback=None,
    noise_kwargs=None,
    callback=None,
    return_output=True,
    return_diagnostics=False,
):
    """Generate a parametric nowcast ensemble from three rain-rate observations.

    Parameters
    ----------
    dataset: xarray.Dataset
        Three complete, finite, nonnegative rain-rate observations in mm/h,
        ordered oldest first. The precipitation variable is named by the global
        ``precip_var`` attribute and has dimensions (time, y, x), in any order.
        Time must be datetime64 and regularly spaced; its ``stepsize`` is in
        seconds. Spatial coordinates must be regular with square pixels, units
        'm' or 'km', and signed ``stepsize`` attributes. Optional velocity_x and
        velocity_y variables have dimensions (y, x) and contain displacements
        in pixels per observation timestep. If absent, estimate Lucas-Kanade
        motion from the latest two dBR fields.
    timesteps: int, optional
        Positive number of leads at the observation interval. Lead zero omitted.
    n_ens_members, n_cascade_levels: int, optional
        Ensemble size (24) and Gaussian cascade levels (6, minimum 3).
    kmperpixel, timestep: float, optional
        Pixel size in km and interval in minutes, derived from coordinates by
        default. Explicit values must agree with the dataset. Fractional
        minutes are preserved.
    rain_threshold, scale_break: float, optional
        Rain occurrence threshold in mm/h (1) and spectral break in km (20).
    zerovalue: float, optional
        Dry dBR value below 10*log10(rain_threshold), default threshold_dbr-0.1.
    taper_sigma: float, optional
        Gaussian boundary-mask smoothing in pixels (5). Applied to transformed
        forecasts before feedback decomposition, after producing the output.
    min_rain_fraction: float, optional
        Post-AR dry-output cutoff (0.025), strictly applied below this fraction.
        Positive low fractions are still distinguished from missing estimates.
    seed: int or numpy.random.SeedSequence, optional
        Seed for independent member streams. Global RNG and supplied SeedSequence
        are not mutated. Prefix members agree when ensemble size is increased.
    motion_kwargs: dict, optional
        Overrides of the default dense Lucas-Kanade configuration. Ignored
        when velocity is supplied; diagnostics record that motion source.
    estimation_kwargs: dict, optional
        Override marginal fitting, temporal scaling, moment selection, bounds,
        and regional low-coverage fallbacks. See
        :doc:`/user_guide/parametric_nowcast` for all keys and defaults.
        Reference time is in minutes.
    temporal_fallback: dict, optional
        Configured temporal-only fallback with positive ``lzero`` (km) and
        ``tzero`` (minutes). If any observed AR(2) pair or the estimated
        reference time is invalid, use tzero*(wavelength/lzero)**temporal_exponent
        to construct all model correlations. Complete input grids and valid
        marginal/spatial estimates remain required. This mode disables the
        combined low-rain-fraction slope and reference-time replacement. Cannot
        be combined with an explicit estimation_kwargs reference_time override.
    noise_kwargs: dict, optional
        Calibration controls for generate_calibrated_field; nbins and eps also
        control post-AR marginal mapping. Randomness is controlled by seed.
    callback: callable, optional
        Called once per lead with a fresh (member, y, x) rain-rate array.
    return_output, return_diagnostics: bool, optional
        Return full ensemble (default True), and optional diagnostics (False).
        With return_output=False, callback receives output without retaining
        the full forecast. Errors from a callback propagate.

    Returns
    -------
    ensemble: xarray.Dataset or None
        Float64 mm/h precipitation with dimensions (ens_number, time, y, x),
        retaining spatial coordinates and projection metadata. Valid times
        start one observation interval after forecast_reference_time.
        None if return_output=False.
    diagnostics: dict, optional
        When requested, return (ensemble, diagnostics): raw estimates, selected
        and bounded generation parameters, settings, motion, seed information,
        and per-member/per-lead calibration results.

    Notes
    -----
    Fixed statistical parameters and motion are used throughout the forecast.
    The two previous cascade states are advected to the forecast time before
    adding normalized random innovations. Distribution matching, boundary taper,
    and re-decomposition feed back into the next state. In STEPS, output
    probability matching is separate from the autoregressive state. Zero
    normalized anomalies at inflow are supplemented by innovations. A newest
    field with no rain at the occurrence threshold gives a dry ensemble even
    if older fields were wet. Earlier dry fields use
    zero anomalies. Unavailable positive-rain moments raise rather than becoming
    dry forecasts. Dataset data are loaded eagerly; the numerical engine uses
    NumPy arrays.
    """
    # Resolve grid geometry and cadence before working with NumPy arrays.
    dataset, velocity, kmperpixel, timestep = _prepare_dataset(
        dataset, kmperpixel, timestep
    )
    precip_var = dataset.attrs["precip_var"]
    rain = dataset[precip_var].values
    shape = rain.shape[1:]
    timesteps = _integer("timesteps", timesteps)
    n_ens_members = _integer("n_ens_members", n_ens_members)
    n_cascade_levels = _integer("n_cascade_levels", n_cascade_levels, 3)
    kmperpixel = _positive("kmperpixel", kmperpixel)
    timestep = _positive("timestep", timestep)
    rain_threshold = _positive("rain_threshold", rain_threshold)
    scale_break = _positive("scale_break", scale_break)
    threshold = float(10 * np.log10(rain_threshold))
    zerovalue = (
        threshold - 0.1 if zerovalue is None else _finite("zerovalue", zerovalue)
    )
    if zerovalue >= threshold:
        raise ValueError("zerovalue must be below the dBR threshold")
    taper_sigma = _finite("taper_sigma", taper_sigma)
    min_rain_fraction = _finite("min_rain_fraction", min_rain_fraction)
    if taper_sigma < 0 or not 0 <= min_rain_fraction <= 1:
        raise ValueError("taper_sigma must be nonnegative; min_rain_fraction in [0, 1]")
    if callback is not None and not callable(callback):
        raise ValueError("callback must be callable")
    for name, flag in [
        ("return_output", return_output),
        ("return_diagnostics", return_diagnostics),
    ]:
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    estimation = _estimation_settings(estimation_kwargs, threshold, n_cascade_levels)
    if temporal_fallback is not None:
        if not isinstance(temporal_fallback, Mapping) or set(temporal_fallback) != {
            "lzero",
            "tzero",
        }:
            raise ValueError(
                "temporal_fallback requires exactly lzero (km) and tzero (minutes)"
            )
        temporal_fallback = {k: _positive(k, v) for k, v in temporal_fallback.items()}
        if estimation["reference_time"] is not None:
            raise ValueError(
                "temporal_fallback cannot be combined with reference_time override"
            )
        # Temporal fallback must never mask an unavailable spatial estimate.
        estimation["low_rain_fraction"] = 0.0
    noise_settings = _noise_settings(noise_kwargs)
    if motion_kwargs is not None and not isinstance(motion_kwargs, Mapping):
        raise ValueError("motion_kwargs must be a mapping")
    motion_settings = deepcopy(_MOTION_DEFAULTS)
    for key, value in dict(motion_kwargs or {}).items():
        if isinstance(value, Mapping) and isinstance(motion_settings.get(key), Mapping):
            motion_settings[key].update(deepcopy(dict(value)))
        else:
            motion_settings[key] = deepcopy(value)
    if velocity is not None:
        if np.iscomplexobj(velocity) or np.ma.isMaskedArray(velocity):
            raise ValueError("velocity must be a finite real array")
        velocity = np.array(velocity, dtype=float, copy=True)
        if velocity.shape != (2, *shape) or not np.isfinite(velocity).all():
            raise ValueError("velocity must be finite with shape (2, y, x)")
    # Give each member a separate stream; increasing ensemble size leaves the
    # existing member streams unchanged. Copy a supplied SeedSequence first.
    if isinstance(seed, np.random.SeedSequence):
        root_seed = np.random.SeedSequence(
            seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size
        )
    else:
        if seed is not None:
            _integer("seed", seed, 0)
        root_seed = np.random.SeedSequence(seed)
    member_seeds = root_seed.spawn(n_ens_members)
    streams = [np.random.default_rng(s) for s in member_seeds]
    settings = dict(
        timestep_minutes=timestep,
        kmperpixel=kmperpixel,
        n_cascade_levels=n_cascade_levels,
        n_ens_members=n_ens_members,
        timesteps=timesteps,
        rain_threshold=rain_threshold,
        scale_break_km=scale_break,
        threshold_dbr=threshold,
        zerovalue_dbr=zerovalue,
        taper_sigma_pixels=taper_sigma,
        min_rain_fraction=min_rain_fraction,
        estimation=estimation,
        temporal_fallback=temporal_fallback,
        noise=noise_settings,
        motion=motion_settings,
    )
    diagnostics = dict(
        method="parametric",
        generator=GENERATOR_ID,
        settings=deepcopy(settings),
        motion_source="supplied" if velocity is not None else "lucaskanade",
        seed=dict(
            entropy=root_seed.entropy,
            spawn_key=root_seed.spawn_key,
            pool_size=root_seed.pool_size,
            member_spawn_keys=[s.spawn_key for s in member_seeds],
        ),
        generation=[[] for _ in range(n_ens_members)] if return_diagnostics else None,
    )
    transformed_dataset = dB_transform(
        dataset, threshold=rain_threshold, zerovalue=zerovalue
    )
    transformed = transformed_dataset[precip_var].values
    # Untransformed rates omit the transform attribute so native NetCDF export
    # does not encounter a non-serializable None value.
    dataset[precip_var].attrs.pop("transform", None)
    dataset[precip_var].attrs.update(
        units="mm/h", threshold=rain_threshold, zerovalue=0.0
    )
    # Estimate occurrence and intensity from the newest observation. A dry
    # newest field has a defined result without estimating motion or spectra.
    statistics = compute_field_statistics(
        transformed[-1], threshold, estimation["lower_truncation"]
    )
    diagnostics["statistics"] = statistics
    output = np.empty((n_ens_members, timesteps, *shape)) if return_output else None
    if statistics["rain_fraction"] == 0:
        diagnostics.update(
            status="dry_latest_observation", velocity=velocity, parameters=None
        )
        if velocity is None:
            diagnostics["motion_source"] = "not_required_for_dry_input"
        for lead in range(timesteps):
            fields = np.zeros((n_ens_members, *shape))
            if output is not None:
                output[:, lead] = fields
            if callback is not None:
                callback(fields)
        if output is not None:
            output = convert_output_to_xarray_dataset(dataset, timesteps, output)
        return (output, diagnostics) if return_diagnostics else output
    if velocity is None:
        motion_dataset = motion.get_method("LK")(
            transformed_dataset.isel(time=slice(-2, None)), **motion_settings
        )
        try:
            velocity = (
                _velocity_from_dataset(motion_dataset)
                if isinstance(motion_dataset, xr.Dataset)
                else None
            )
        except ValueError as error:
            raise ValueError("Motion estimator returned invalid velocity") from error
        if velocity is None or velocity.shape != (2, *shape):
            raise ValueError("Motion estimator returned invalid velocity")
    diagnostics["velocity"] = velocity.copy()
    # Estimate scale-dependent persistence from the three aligned observations.
    bp_filter = filter_gaussian(shape, n_cascade_levels)
    observed = [_decompose(field, bp_filter) for field in transformed]
    wavelengths = kmperpixel / bp_filter["central_freqs"]
    correlations = lagrangian_autocorrelation(
        np.stack([d["cascade_levels"] for d in observed]), velocity
    )
    times = ar2_correlation_time(correlations[:, 0], correlations[:, 1], timestep)
    estimated_time = estimate_reference_correlation_time(
        times,
        wavelengths,
        estimation["temporal_exponent"],
        levels=estimation["reference_levels"],
    )
    reference_time = (
        estimated_time
        if estimation["reference_time"] is None
        else estimation["reference_time"]
    )
    temporal_selection = dict(source="observed", reason=None)
    if estimation["reference_time"] is not None:
        temporal_selection["source"] = "override"
    use_fallback = temporal_fallback is not None and (
        not np.isfinite(times).all()
        or np.any(times <= 0)
        or not np.isfinite(estimated_time)
        or estimated_time <= 0
    )
    if use_fallback:
        reference_time = (
            temporal_fallback["tzero"]
            * (wavelengths[0] / temporal_fallback["lzero"])
            ** estimation["temporal_exponent"]
        )
        temporal_selection.update(
            source="configured_fallback",
            reason="invalid_observed_ar2_or_reference_time",
            configuration=deepcopy(temporal_fallback),
        )
    temporal_selection["selected_reference_time_minutes"] = reference_time
    # Fit spatial structure and select the parameters held fixed for all leads.
    frequency, power = rapsd_logarithmic(transformed[-1])
    spectrum = fit_power_law_spectrum(frequency, power, scale_break / kmperpixel)
    parameters, safeguards = _generation_parameters(
        statistics, spectrum, reference_time, estimation
    )
    if use_fallback:
        # The explicit configured scale law is not clipped to regional bounds.
        parameters["reference_time"] = reference_time
        safeguards["bounded_parameters"] = [
            k for k in safeguards["bounded_parameters"] if k != "reference_time"
        ]
    model_lags, temporal_diagnostics = model_cascade_correlations(
        parameters["reference_time"],
        wavelengths,
        estimation["temporal_exponent"],
        estimation["alpha"],
        timestep,
        return_diagnostics=True,
    )
    # Convert model lag correlations to AR coefficients and innovation weights.
    phi = np.array([estimate_ar_params_yw(pair) for pair in model_lags])
    diagnostics.update(
        status="ok",
        spectrum=spectrum,
        observed_correlations=correlations,
        observed_correlation_times_minutes=times,
        estimated_reference_time_minutes=estimated_time,
        wavelengths_km=wavelengths,
        parameters=parameters.copy(),
        safeguards=safeguards,
        temporal_model=temporal_diagnostics,
        temporal_selection=temporal_selection,
        model_correlations=model_lags,
        ar_coefficients=phi,
        constant_observed_levels=[
            np.flatnonzero(np.asarray(d["stds"]) == 0) for d in observed
        ],
    )
    coordinates = prepare_advection_coordinates(velocity)
    taper = gaussian_filter(advect_field(np.ones(shape), coordinates[1]), taper_sigma)
    # Kernels never mutate histories. Members start from the same observations
    # and replace their own list entries after each update.
    histories = [[observed[-1], observed[-2]] for _ in range(n_ens_members)]
    for lead in range(timesteps):
        fields = np.empty((n_ens_members, *shape))
        for member in range(n_ens_members):
            field, generation = _advance(
                histories[member],
                parameters,
                phi,
                bp_filter,
                coordinates,
                streams[member],
                threshold,
                zerovalue,
                kmperpixel,
                scale_break,
                noise_settings,
                min_rain_fraction,
            )
            with np.errstate(over="ignore"):
                physical = 10 ** (field / 10)
            physical[physical < rain_threshold] = 0
            if not np.isfinite(physical).all():
                raise ValueError("Generated rainfall contains nonfinite values")
            # Return physical rainfall before tapering. Only the feedback state
            # is tapered and decomposed for the next forecast step.
            fields[member] = physical
            histories[member] = [
                _decompose(field * taper, bp_filter),
                histories[member][0],
            ]
            if return_diagnostics:
                diagnostics["generation"][member].append(generation)
        if output is not None:
            output[:, lead] = fields
        if callback is not None:
            callback(fields.copy())
    if output is not None:
        output = convert_output_to_xarray_dataset(dataset, timesteps, output)
    return (output, diagnostics) if return_diagnostics else output
