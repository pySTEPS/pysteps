"""
pysteps.nowcasts.parametric
===========================

Generate rainfall time series from estimated or prescribed parameters.

.. autosummary::
    :toctree: ../generated/

    forecast
    simulate
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


def _estimation_settings(settings, threshold, n_levels):
    """Validate estimation settings against the threshold and cascade count."""
    settings = deepcopy(settings)
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


def _random_streams(seed, n_ens_members):
    """Create independent member streams and record their seed identities."""
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
    information = dict(
        entropy=root_seed.entropy,
        spawn_key=root_seed.spawn_key,
        pool_size=root_seed.pool_size,
        member_spawn_keys=[s.spawn_key for s in member_seeds],
    )
    return streams, information


def _evolution_settings(
    timesteps,
    n_ens_members,
    n_cascade_levels,
    rain_threshold,
    scale_break,
    zerovalue,
    taper_sigma,
    min_rain_fraction,
    callback,
    return_output,
    return_diagnostics,
):
    """Validate shared evolution controls and return settings with explicit units."""
    timesteps = _integer("timesteps", timesteps)
    n_ens_members = _integer("n_ens_members", n_ens_members)
    n_cascade_levels = _integer("n_cascade_levels", n_cascade_levels, 3)
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
    return dict(
        timesteps=timesteps,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        rain_threshold=rain_threshold,
        scale_break_km=scale_break,
        threshold_dbr=threshold,
        zerovalue_dbr=zerovalue,
        taper_sigma_pixels=taper_sigma,
        min_rain_fraction=min_rain_fraction,
    )


def _run_ensemble(
    dataset,
    observed,
    velocity,
    parameters,
    phi,
    bp_filter,
    streams,
    settings,
    diagnostics,
    callback,
    return_output,
    return_diagnostics,
):
    """Evolve initialized members and collect or stream rain-rate fields.

    Both forecast and simulate use this loop. The observed sequence contains
    normalized cascade decompositions, oldest first. Settings contain validated
    physical units and calibration controls. A missing parameter set or zero target rain
    fraction returns dry fields without drawing noise or constructing trajectories.
    """
    shape = dataset[dataset.attrs["precip_var"]].shape[1:]
    n_ens_members = len(streams)
    timesteps = settings["timesteps"]
    threshold = settings["threshold_dbr"]
    zerovalue = settings["zerovalue_dbr"]
    rain_threshold = settings["rain_threshold"]
    kmperpixel = settings["kmperpixel"]
    scale_break = settings["scale_break_km"]
    noise_settings = settings["noise"]
    min_rain_fraction = settings["min_rain_fraction"]
    dry = parameters is None or parameters["rain_fraction"] == 0
    output = np.empty((n_ens_members, timesteps, *shape)) if return_output else None
    if not dry:
        coordinates = prepare_advection_coordinates(velocity)
        taper = gaussian_filter(
            advect_field(np.ones(shape), coordinates[1]), settings["taper_sigma_pixels"]
        )
        # Histories share read-only observations until each member replaces its
        # own list entries. The numerical kernels never mutate these observations.
        histories = [[observed[-1], observed[-2]] for _ in range(n_ens_members)]
    members = range(n_ens_members) if not dry else ()
    for lead in range(timesteps):
        fields = (
            np.zeros((n_ens_members, *shape))
            if dry
            else np.empty((n_ens_members, *shape))
        )
        for member in members:
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
    lower_truncation: float, optional
        Lower cutoff for fitting the parent Normal distribution, in dBR (1.25).
        Must be at or above the transformed rain occurrence threshold.
    temporal_exponent: float, optional
        Exponent relating correlation time to cascade wavelength (0.81).
    alpha: float, optional
        Power controlling the model lag-1 and lag-2 correlations (2.72).
    reference_levels: tuple of int, optional
        Zero-based cascade levels used to estimate reference time, default (0,).
        Every selected level must yield a valid estimate.
    reference_time: float, optional
        Reference correlation time in minutes. By default, estimate it from the
        observations. A supplied value replaces that estimate before applying
        low-coverage replacements and correlation-time bounds.
    moment_distance_limit: float, optional
        Maximum Euclidean distance between fitted and sample intensity moments,
        normalized by the sample standard deviation (0.5). Use the fitted
        moments only when valid and within this limit.
    beta_1_bounds: tuple of float, optional
        Lower and upper bounds for the large-scale spectral slope, default
        (-3.9, just below -1.25).
    beta_2_min: float, optional
        Lower bound for the small-scale spectral slope (-4.0).
    minimum_beta_gap: float, optional
        Minimum separation enforcing beta_2 <= beta_1 - gap (0.1).
    correlation_time_bounds: tuple of float, optional
        Lower and upper reference-time bounds in minutes, default (60, 720).
    low_rain_fraction: float, optional
        Rain fraction below which both slopes and reference time are replaced
        by the fallback values (0.05). Set to zero to disable replacement.
    fallback_beta_1, fallback_beta_2: float, optional
        Low-coverage spectral slope replacements, default -1.45 and
        -3.253958088759979. The small-scale value is an Auckland rainfall
        archive median; see :doc:`/user_guide/parametric_nowcast`.
    fallback_reference_time: float, optional
        Low-coverage reference-time replacement in minutes
        (330.61752959848894), the Auckland rainfall archive median.
    temporal_fallback: dict, optional
        Configured temporal-only fallback with positive ``lzero`` (km) and
        ``tzero`` (minutes). If any observed AR(2) pair or the estimated
        reference time is invalid, use tzero*(wavelength/lzero)**temporal_exponent
        to construct all model correlations. Complete input grids and valid
        marginal/spatial estimates remain required. This mode disables the
        combined low-rain-fraction slope and reference-time replacement. Cannot
        be combined with an explicit reference_time override.
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

    See also
    --------
    simulate

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
    settings = _evolution_settings(
        timesteps,
        n_ens_members,
        n_cascade_levels,
        rain_threshold,
        scale_break,
        zerovalue,
        taper_sigma,
        min_rain_fraction,
        callback,
        return_output,
        return_diagnostics,
    )
    timesteps = settings["timesteps"]
    n_ens_members = settings["n_ens_members"]
    n_cascade_levels = settings["n_cascade_levels"]
    rain_threshold = settings["rain_threshold"]
    threshold = settings["threshold_dbr"]
    zerovalue = settings["zerovalue_dbr"]
    estimation = _estimation_settings(
        dict(
            lower_truncation=lower_truncation,
            temporal_exponent=temporal_exponent,
            alpha=alpha,
            reference_levels=reference_levels,
            reference_time=reference_time,
            moment_distance_limit=moment_distance_limit,
            beta_1_bounds=beta_1_bounds,
            beta_2_min=beta_2_min,
            minimum_beta_gap=minimum_beta_gap,
            correlation_time_bounds=correlation_time_bounds,
            low_rain_fraction=low_rain_fraction,
            fallback_beta_1=fallback_beta_1,
            fallback_beta_2=fallback_beta_2,
            fallback_reference_time=fallback_reference_time,
        ),
        threshold,
        n_cascade_levels,
    )
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
    streams, seed_information = _random_streams(seed, n_ens_members)
    settings.update(
        timestep_minutes=timestep,
        kmperpixel=kmperpixel,
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
        seed=seed_information,
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
    if statistics["rain_fraction"] == 0:
        diagnostics.update(
            status="dry_latest_observation", velocity=velocity, parameters=None
        )
        if velocity is None:
            diagnostics["motion_source"] = "not_required_for_dry_input"
        return _run_ensemble(
            dataset,
            None,
            velocity,
            None,
            None,
            None,
            streams,
            settings,
            diagnostics,
            callback,
            return_output,
            return_diagnostics,
        )
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
    return _run_ensemble(
        dataset,
        observed,
        velocity,
        parameters,
        phi,
        bp_filter,
        streams,
        settings,
        diagnostics,
        callback,
        return_output,
        return_diagnostics,
    )


def simulate(
    dataset,
    timesteps=1,
    *,
    parameters,
    reference_time,
    reference_wavelength,
    temporal_exponent=0.81,
    alpha=2.72,
    n_ens_members=24,
    kmperpixel=None,
    timestep=None,
    n_cascade_levels=6,
    rain_threshold=1.0,
    scale_break=20.0,
    zerovalue=None,
    taper_sigma=5.0,
    min_rain_fraction=0.0,
    seed=None,
    noise_kwargs=None,
    callback=None,
    return_output=True,
    return_diagnostics=False,
):
    """Simulate rainfall time series using prescribed statistics and motion.

    Parameters
    ----------
    dataset: xarray.Dataset
        Three complete, finite, nonnegative rain-rate fields in mm/h, ordered
        oldest first, on a regular grid with square pixels and at least four
        pixels along each spatial dimension. Use the dataset
        conventions described in :func:`forecast`: a global precip_var attribute,
        precipitation dimensions (time, y, x) in any order, spatial units 'm' or
        'km', and coordinate stepsize attributes matching the grid and time
        interval. Time stepsize is in seconds. Supply both velocity_x and
        velocity_y with dimensions (y, x), in either order, and displacements
        in pixels per interval. Motion is not estimated. The latest two fields
        initialize the AR(2) state; the oldest is retained for consistency with
        the forecast input interface.
    timesteps: int, optional
        Positive number of intervals to simulate (default 1). The first output
        is one observation interval after the newest initial field.
    parameters: mapping
        Exactly five finite real values: beta_1 and beta_2 are signed spectral
        slopes; rain_fraction lies in [0, 1]; mean_db and positive stdev_db are
        the Normal location and scale in dBR. These describe the generation
        targets, not necessarily the initial fields' sample statistics. Values
        are copied and held fixed, without estimation, clipping, or regional
        replacements.
    reference_time: float
        Positive correlation time in minutes at reference_wavelength. This is
        the one-sided discrete AR(2) autocorrelation sum including lag zero,
        not an e-folding time or a forecast-skill horizon.
    reference_wavelength: float
        Positive reference wavelength in km. It need not equal a cascade level's
        nominal wavelength or the domain size.
    temporal_exponent: float, optional
        Finite exponent in T = reference_time *
        (wavelength/reference_wavelength)**temporal_exponent (default 0.81).
    alpha: float, optional
        Positive exponent in rho2 = rho1**alpha before the nonperiodic AR(2)
        adjustment (default 2.72). Lag correlations are converted to AR
        coefficients and innovation weights separately.
    n_ens_members: int, optional
        Number of independent realizations (default 24).
    kmperpixel, timestep: float, optional
        Pixel spacing in km and interval in minutes. Derived from coordinates
        by default; explicit values must agree. Fractional minutes are retained.
    n_cascade_levels: int, optional
        Number of Gaussian cascade levels (default 6, minimum 3).
    rain_threshold: float, optional
        Rain occurrence threshold in mm/h (default 1).
    scale_break: float, optional
        Prescribed spectral break wavelength in km (default 20).
    zerovalue: float, optional
        Dry dBR value below the transformed rain threshold. The default is
        10*log10(rain_threshold)-0.1.
    taper_sigma: float, optional
        Gaussian boundary-mask smoothing width in pixels (default 5). Tapering
        affects the feedback state after the output field has been produced.
    min_rain_fraction: float, optional
        Return dry output for target fractions strictly below this value.
        Default 0 preserves positive target fractions; forecast defaults to
        0.025. Failed positive-fraction mappings raise an error.
    seed: int or numpy.random.SeedSequence, optional
        Seed for separate member streams. Global random state and a supplied
        SeedSequence are unchanged. Increasing the ensemble size preserves the
        leading members for identical settings.
    noise_kwargs: dict, optional
        Overrides for spectral calibration and distribution matching, as in
        :func:`forecast`. Calibration may miss its tolerance; diagnostics
        report the selected candidate and its spectral error.
    callback: callable, optional
        Called once per interval with a fresh (member, y, x) array in mm/h.
        Modifying it does not affect the state or retained output. Exceptions
        propagate to the caller.
    return_output: bool, optional
        Retain and return the full time series (default True). Set False to
        stream through callback without retaining the full output.
    return_diagnostics: bool, optional
        Also return diagnostics (default False).

    Returns
    -------
    ensemble: xarray.Dataset or None
        Float64 rain rates with dimensions (ens_number, time, y, x), preserving
        spatial coordinates and projection metadata. Members are numbered from
        one. The forecast_reference_time coordinate identifies the newest
        initial field, following the forecast output convention. None if
        return_output=False.
    diagnostics: dict, optional
        When requested, return (ensemble, diagnostics). Record the prescribed
        parameters, effective settings, seed identities, supplied velocity,
        nominal wavelengths, requested and achieved correlation times, model
        lag correlations, AR coefficients, and calibration results indexed
        [member][interval].

    See also
    --------
    forecast, pysteps.timeseries.parametric.model_cascade_correlations

    Notes
    -----
    Simulation and forecasting share the same advection, stochastic AR(2)
    updates, distribution matching, and boundary feedback. Parameters and motion
    stay fixed while each member's rainfall pattern evolves. Matching of rain
    fraction, intensity distribution, and spectra is approximate on finite
    grids. Distribution matching and boundary feedback can change correlations
    measured from the resulting rainfall. Requested level times at or below one
    interval resolve to white noise, with achieved times recorded in diagnostics.

    A zero target rain fraction produces dry output even from wet initial
    fields. Conversely, a positive target can generate rain from dry initial
    fields. Missing initial values remain errors. Shifted copies of one field
    can initialize a simulation, but they represent pure translation; they do
    not establish a stationary history for the prescribed temporal model.
    """
    dataset, velocity, kmperpixel, timestep = _prepare_dataset(
        dataset, kmperpixel, timestep
    )
    if velocity is None:
        raise ValueError("simulate requires both velocity_x and velocity_y")
    precip_var = dataset.attrs["precip_var"]
    shape = dataset[precip_var].shape[1:]
    if min(shape) < 4:
        raise ValueError("Simulation requires at least four pixels per axis")
    settings = _evolution_settings(
        timesteps,
        n_ens_members,
        n_cascade_levels,
        rain_threshold,
        scale_break,
        zerovalue,
        taper_sigma,
        min_rain_fraction,
        callback,
        return_output,
        return_diagnostics,
    )
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping")
    names = ("beta_1", "beta_2", "rain_fraction", "mean_db", "stdev_db")
    if set(parameters) != set(names):
        raise ValueError(f"parameters must contain exactly {', '.join(names)}")
    parameters = {name: _finite(name, parameters[name]) for name in names}
    if not 0 <= parameters["rain_fraction"] <= 1:
        raise ValueError("rain_fraction must lie in [0, 1]")
    _positive("stdev_db", parameters["stdev_db"])
    reference_time = _positive("reference_time", reference_time)
    reference_wavelength = _positive("reference_wavelength", reference_wavelength)
    temporal_exponent = _finite("temporal_exponent", temporal_exponent)
    alpha = _positive("alpha", alpha)
    noise_settings = _noise_settings(noise_kwargs)
    streams, seed_information = _random_streams(seed, settings["n_ens_members"])
    settings.update(
        timestep_minutes=timestep,
        kmperpixel=kmperpixel,
        reference_time_minutes=reference_time,
        reference_wavelength_km=reference_wavelength,
        temporal_exponent=temporal_exponent,
        alpha=alpha,
        noise=noise_settings,
    )

    # Construct temporal correlations from the prescribed scale law, using the
    # actual filter wavelengths. No correlations are estimated from the history.
    bp_filter = filter_gaussian(shape, settings["n_cascade_levels"])
    wavelengths = kmperpixel / bp_filter["central_freqs"]
    model_lags, temporal_diagnostics = model_cascade_correlations(
        reference_time,
        wavelengths,
        temporal_exponent,
        alpha,
        timestep,
        reference_wavelength=reference_wavelength,
        return_diagnostics=True,
    )
    phi = np.array([estimate_ar_params_yw(pair) for pair in model_lags])

    # Initial rainfall supplies cascade states only. It does not alter the
    # supplied intensity, occurrence, spectral, or persistence parameters.
    transformed_dataset = dB_transform(
        dataset,
        threshold=settings["rain_threshold"],
        zerovalue=settings["zerovalue_dbr"],
    )
    observed = [
        _decompose(field, bp_filter) for field in transformed_dataset[precip_var].values
    ]
    dataset[precip_var].attrs.pop("transform", None)
    dataset[precip_var].attrs.update(
        units="mm/h", threshold=settings["rain_threshold"], zerovalue=0.0
    )
    diagnostics = dict(
        method="parametric_simulation",
        generator=GENERATOR_ID,
        status="dry_target" if parameters["rain_fraction"] == 0 else "ok",
        settings=deepcopy(settings),
        parameter_source="supplied",
        parameters=parameters.copy(),
        motion_source="supplied",
        velocity=velocity.copy(),
        seed=seed_information,
        wavelengths_km=wavelengths,
        temporal_model=temporal_diagnostics,
        model_correlations=model_lags,
        ar_coefficients=phi,
        generation=[[] for _ in streams] if return_diagnostics else None,
    )
    return _run_ensemble(
        dataset,
        observed,
        velocity,
        parameters,
        phi,
        bp_filter,
        streams,
        settings,
        diagnostics,
        callback,
        return_output,
        return_diagnostics,
    )
