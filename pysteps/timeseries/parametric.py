"""
pysteps.timeseries.parametric
=============================

Correlation estimators and temporal scaling for parametric nowcasting.

.. autosummary::
    :toctree: ../generated/

    lagrangian_autocorrelation
    ar2_correlation_time
    fit_ar2_correlations
    estimate_reference_correlation_time
    model_cascade_correlations
"""

import numpy as np
from scipy.optimize import brentq

from pysteps.extrapolation.semilagrangian import extrapolate

CORRELATION_TIME_DEFINITION = "ar_acf_sum_including_zero_v2"


def _finite(name, value):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive(name, value):
    _finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _wavelengths(values):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError("wavelengths must be a nonempty positive finite 1D array")
    return values


def ar2_correlation_time(lag1, lag2, timestep, tol=1e-5):
    """Calculate the one-sided stationary AR(2) discrete ACF sum including zero.

    Parameters
    ----------
    lag1, lag2: float or array_like
        Observed lag-1 and lag-2 correlations. Arrays are broadcast together.
    timestep: float
        Sampling interval. The returned times have the same unit.
    tol: float, optional
        Guard for singular or near-nonstationary pairs (default 1e-5).

    Returns
    -------
    time: float or ndarray
        timestep * sum(rho[k], k=0..infinity). Invalid, nonstationary, and
        numerically singular pairs return NaN, without AR(1) substitution.

    Notes
    -----
    The sum is timestep*(1+rho1-phi1)/(1-phi1-phi2), with Yule-Walker AR
    coefficients. White noise has time equal to one timestep. This differs
    from a trapezoidal ACF integral by timestep/2 and is not a skill horizon.
    """
    _positive("tol", tol)
    if tol >= 1:
        raise ValueError("tol must be less than one")
    lag1, lag2 = np.broadcast_arrays(np.asarray(lag1, float), np.asarray(lag2, float))
    result = np.full(lag1.shape, np.nan)
    if not np.isscalar(timestep) or not np.isfinite(timestep) or timestep <= 0:
        return float(result) if result.ndim == 0 else result
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = 1 - lag1**2
        phi1 = lag1 * (1 - lag2) / denominator
        phi2 = (lag2 - lag1**2) / denominator
        total_denominator = 1 - phi1 - phi2
        valid = (
            np.isfinite(lag1)
            & np.isfinite(lag2)
            & (np.abs(lag1) < 1 - tol)
            & (np.abs(lag2) < 1 - tol)
            & (np.abs(denominator) >= tol)
            & (np.abs(phi2) < 1 - tol)
            & (phi1 + phi2 < 1 - tol)
            & (phi2 - phi1 < 1 - tol)
            & (total_denominator > tol)
        )
        np.copyto(result, timestep * (1 + lag1 - phi1) / total_denominator, where=valid)
    return float(result) if result.ndim == 0 else result


def _adjust_nonperiodic_lag2(lag1, lag2):
    # Rationalized equivalent of adjust_lag2_corrcoef2, stable also at lag1=0.
    root = np.sqrt(1 - lag1**2)
    lower = lag1**2 * (1 + 2 * root) / (1 + root) ** 2
    return float(max(lag2, 2 * lag1 * lag2 - 1, lower))


def fit_ar2_correlations(correlation_time, alpha, timestep, return_diagnostics=False):
    """Invert a discrete timescale in the nonnegative, nonperiodic AR(2) family.

    Parameters
    ----------
    correlation_time, timestep: float
        Positive target time and sampling interval in the same unit. A target
        below timestep is unrepresentable and raises ValueError.
    alpha: float
        Positive exponent in rho2 = rho1**alpha before nonperiodic adjustment.
    return_diagnostics: bool, optional
        Also return requested and achieved times and their absolute difference.

    Returns
    -------
    correlations: ndarray
        The adjusted [rho1, rho2] used by the inversion. Exactly one timestep
        returns [0, 0]. The exponent relation can change at the adjustment bound.
    diagnostics: dict, optional
        Definition, requested_time, achieved_time, and absolute_error.
    """
    _positive("correlation_time", correlation_time)
    _positive("alpha", alpha)
    _positive("timestep", timestep)
    if correlation_time < timestep:
        raise ValueError("correlation_time is below the discrete minimum timestep")

    def pair(rho1):
        return _adjust_nonperiodic_lag2(rho1, rho1**alpha)

    def residual(rho1):
        return ar2_correlation_time(rho1, pair(rho1), timestep) - correlation_time

    rho1 = rho2 = 0.0
    if correlation_time > timestep:
        lower, upper, candidate = 0.0, 1.0, 0.5
        bracket = None
        for _ in range(80):
            value = residual(candidate)
            if np.isfinite(value):
                if value >= 0:
                    bracket = (0.0, candidate)
                    break
                lower = candidate
            else:
                upper = candidate
            candidate = (lower + upper) / 2
            if candidate == lower or candidate == upper:
                break
        if bracket is None:
            raise ValueError(
                "correlation_time exceeds the numerically stable AR(2) range"
            )
        rho1 = float(brentq(residual, *bracket, xtol=1e-14, rtol=1e-14))
        rho2 = pair(rho1)
    achieved = ar2_correlation_time(rho1, rho2, timestep)
    if not np.isfinite(achieved) or not np.isclose(
        achieved, correlation_time, rtol=1e-7, atol=1e-8
    ):
        raise ValueError("AR(2) inverse failed to achieve the requested timescale")
    correlations = np.array([rho1, rho2])
    diagnostics = dict(
        definition=CORRELATION_TIME_DEFINITION,
        requested_time=float(correlation_time),
        achieved_time=achieved,
        absolute_error=float(abs(achieved - correlation_time)),
    )
    return (correlations, diagnostics) if return_diagnostics else correlations


def estimate_reference_correlation_time(
    times, wavelengths, exponent, levels=(0, 1, 2), reference_wavelength=None
):
    """Estimate a reference time by geometric averaging selected scale-adjusted times.

    Parameters
    ----------
    times: array_like
        Observed correlation times with shape (..., level), in one time unit.
    wavelengths: array_like
        Positive nominal wavelengths, one per level, from the actual cascade
        filter (reciprocals of its central frequencies in consistent units).
    exponent: float
        Fixed finite temporal scaling exponent, not estimated here.
    levels: sequence of int, optional
        Zero-based contributing levels; defaults to (0, 1, 2), the selected
        paper method. (0,) reproduces the production largest-level convention.
    reference_wavelength: float, optional
        Reference length, default wavelengths[0].

    Returns
    -------
    time: float or ndarray
        exp(mean(log(T_j)-exponent*log(wavelength_j/reference_wavelength))).
        Every selected time must be positive and finite; otherwise NaN is
        returned for that sample. No level dropping or implicit fallback occurs.
    """
    wavelengths = _wavelengths(wavelengths)
    times = np.asarray(times, dtype=float)
    if times.ndim < 1 or times.shape[-1] != wavelengths.size:
        raise ValueError("times' last dimension must match wavelengths")
    _finite("exponent", exponent)
    levels = np.asarray(levels)
    if (
        levels.ndim != 1
        or levels.size == 0
        or not np.issubdtype(levels.dtype, np.integer)
        or np.any(levels < 0)
        or np.any(levels >= wavelengths.size)
        or np.unique(levels).size != levels.size
    ):
        raise ValueError("levels must contain distinct valid integer indices")
    reference = wavelengths[0] if reference_wavelength is None else reference_wavelength
    _positive("reference_wavelength", reference)
    selected = times[..., levels]
    valid = (np.isfinite(selected) & (selected > 0)).all(axis=-1)
    result = np.full(times.shape[:-1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = np.exp(
            np.mean(
                np.log(selected) - exponent * np.log(wavelengths[levels] / reference),
                axis=-1,
            )
        )
    np.copyto(result, values, where=valid & np.isfinite(values))
    return float(result) if result.ndim == 0 else result


def model_cascade_correlations(
    reference_time,
    wavelengths,
    exponent,
    alpha,
    timestep,
    reference_wavelength=None,
    return_diagnostics=False,
):
    """Construct AR(2) lag pairs from a reference time and nominal wavelengths.

    Parameters
    ----------
    reference_time, timestep: float
        Positive reference time and sampling interval, in the same unit.
    wavelengths: array_like
        Positive nominal wavelengths, from the actual cascade filter.
    exponent, alpha: float
        Finite scale-time exponent and positive pre-adjustment lag-pair exponent.
    reference_wavelength: float, optional
        Reference length, default wavelengths[0].
    return_diagnostics: bool, optional
        Also return requested and achieved times, and sub-timestep limit flags.

    Returns
    -------
    correlations: ndarray
        Shape (level, 2), containing adjusted lag-1/lag-2 pairs. Requested times
        at or below timestep use white noise [0, 0], achieving timestep.
    diagnostics: dict, optional
        The chosen ACF definition and per-level requested/achieved times. Length
        and time units are caller supplied; no implicit conversion is performed.
    """
    wavelengths = _wavelengths(wavelengths)
    for name, value in [
        ("reference_time", reference_time),
        ("alpha", alpha),
        ("timestep", timestep),
    ]:
        _positive(name, value)
    _finite("exponent", exponent)
    reference = wavelengths[0] if reference_wavelength is None else reference_wavelength
    _positive("reference_wavelength", reference)
    requested = reference_time * (wavelengths / reference) ** exponent
    if not np.isfinite(requested).all() or np.any(requested <= 0):
        raise ValueError("scale-adjusted times must be positive and finite")
    correlations = np.array(
        [
            fit_ar2_correlations(max(time, timestep), alpha, timestep)
            for time in requested
        ]
    )
    diagnostics = dict(
        definition=CORRELATION_TIME_DEFINITION,
        timestep=float(timestep),
        requested_times=requested,
        achieved_times=ar2_correlation_time(
            correlations[:, 0], correlations[:, 1], timestep
        ),
        limited_below_timestep=requested < timestep,
    )
    return (correlations, diagnostics) if return_diagnostics else correlations


def lagrangian_autocorrelation(
    cascades, velocity, ar_order=2, min_stdev=0.1, boundary="zero"
):
    """Estimate cascade correlations by successive unit-step backward advection.

    Parameters
    ----------
    cascades: array_like
        Finite history with shape (time, level, y, x), oldest first, containing
        at least ar_order+1 times. Only the latest required history is used.
    velocity: array_like
        Finite fixed flow with shape (2, y, x), in pixels per timestep.
    ar_order: {1, 2}, optional
        Number of observed lags to estimate (default 2).
    min_stdev: float, optional
        Both aligned fields must exceed this population standard deviation.
    boundary: {'zero', 'mask'}, optional
        'zero' preserves the evaluated production estimator: fill out-of-domain
        pixels with zero after each unit step and correlate the whole grid.
        'mask' preserves NaNs through advection and uses common finite support
        across all lags. It changes the estimator and is an explicit alternative.

    Returns
    -------
    correlations: ndarray
        Shape (level, ar_order), in ascending lag order. Insufficient support or
        variance produces NaN. There is no clipping or stationarity repair.
    """
    cascades, velocity = np.asarray(cascades, float), np.asarray(velocity, float)
    if not isinstance(ar_order, (int, np.integer)) or ar_order not in (1, 2):
        raise ValueError("ar_order must be 1 or 2")
    if (
        cascades.ndim != 4
        or cascades.shape[0] < ar_order + 1
        or any(size == 0 for size in cascades.shape)
    ):
        raise ValueError(
            "cascades must have shape (time, level, y, x) with sufficient history"
        )
    if velocity.shape != (2, *cascades.shape[-2:]):
        raise ValueError("velocity must have shape (2, y, x) matching cascades")
    if not np.isfinite(cascades).all() or not np.isfinite(velocity).all():
        raise ValueError("cascades and velocity must contain only finite values")
    if boundary not in ("zero", "mask"):
        raise ValueError("boundary must be 'zero' or 'mask'")
    if not np.isfinite(min_stdev) or min_stdev < 0:
        raise ValueError("min_stdev must be finite and nonnegative")
    result = np.full((cascades.shape[1], ar_order), np.nan)
    for level in range(cascades.shape[1]):
        aligned = [cascades[-1, level]]
        for lag in range(1, ar_order + 1):
            field = cascades[-(lag + 1), level]
            for _ in range(lag):
                field = extrapolate(field, velocity, 1, allow_nonfinite_values=True)[0]
                if boundary == "zero":
                    field = np.where(np.isfinite(field), field, 0)
            aligned.append(field)
        support = np.isfinite(np.stack(aligned)).all(axis=0)
        if np.count_nonzero(support) < 2:
            continue
        latest = aligned[0][support]
        for lag in range(1, ar_order + 1):
            previous = aligned[lag][support]
            if np.std(latest) > min_stdev and np.std(previous) > min_stdev:
                result[level, lag - 1] = np.corrcoef(latest, previous)[0, 1]
    return result
