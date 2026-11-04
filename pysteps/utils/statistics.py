"""
pysteps.utils.statistics
========================

Marginal parameter estimators for transformed precipitation.

.. autosummary::
    :toctree: ../generated/

    fit_lower_truncated_normal
    compute_field_statistics
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

__all__ = ["fit_lower_truncated_normal", "compute_field_statistics"]


def fit_lower_truncated_normal(data, lower_truncation, return_diagnostics=False):
    """Fit the latent parent Normal to samples above a lower cutoff.

    Parameters
    ----------
    data: array_like
        Observations in transformed units. Nonfinite and below-cutoff values
        are excluded. The input is not modified.
    lower_truncation: float
        Finite lower fitting cutoff in the same units as data.
    return_diagnostics: bool, optional
        Also return a dictionary describing sample count and fitting status.

    Returns
    -------
    location, scale: float
        Parent-Normal location and scale, distinct from retained-sample moments.
        Both are NaN for insufficient or constant samples or failed optimization.
    diagnostics: dict, optional
        Sample count, status, standardized cutoff, and boundary indicators.

    Notes
    -----
    Profiles scale analytically for a = (cutoff-location)/scale and minimizes
    the conditional negative log likelihood over -12 <= a <= 12, preserving
    the source ``pysteps_param`` method. Gaussian tail normalization uses log_ndtr.
    A finite bounded solution is not proof of an unrestricted finite MLE:
    ``boundary_limited`` identifies proximity to the bound or shifted-sample
    coefficient of variation >= 1. Its numerical solution is retained for
    compatibility and reported explicitly. Pixels have equal marginal weights;
    spatial dependence and parameter uncertainty are not estimated here.
    """
    if not np.isfinite(lower_truncation):
        raise ValueError("lower_truncation must be finite")
    if np.iscomplexobj(data):
        raise ValueError("data must contain real values")
    sample = np.asarray(data, dtype=float)
    sample = sample[np.isfinite(sample) & (sample >= lower_truncation)]
    diagnostics = dict(
        n_samples=int(sample.size),
        status="insufficient_samples",
        boundary_limited=False,
    )

    def output(location=np.nan, scale=np.nan):
        result = (float(location), float(scale))
        return (*result, diagnostics) if return_diagnostics else result

    if sample.size < 2:
        return output()
    if np.ptp(sample) <= 0:
        diagnostics["status"] = "constant_sample"
        return output()
    shifted = sample - lower_truncation
    mean_shifted = float(np.mean(shifted))
    second_moment = float(np.mean(shifted * shifted))

    def scale_for_cutoff(a):
        return 0.5 * (
            a * mean_shifted + np.sqrt(a**2 * mean_shifted**2 + 4 * second_moment)
        )

    def objective(a):
        scale = scale_for_cutoff(a)
        if not np.isfinite(scale) or scale <= 0:
            return np.inf
        standardized_second_moment = (
            second_moment / scale**2 + 2 * a * mean_shifted / scale + a**2
        )
        return float(np.log(scale) + 0.5 * standardized_second_moment + log_ndtr(-a))

    result = minimize_scalar(
        objective,
        bounds=(-12.0, 12.0),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )
    a = float(result.x)
    scale = float(scale_for_cutoff(a))
    location = float(lower_truncation - a * scale)
    if not result.success or not np.isfinite([location, scale]).all() or scale <= 0:
        diagnostics["status"] = "optimization_failed"
        return output()
    cv = float(np.std(shifted) / mean_shifted)
    boundary = abs(a) >= 12 - 1e-5 or cv >= 1
    diagnostics.update(
        status="bounded_solution" if boundary else "ok",
        boundary_limited=bool(boundary),
        standardized_cutoff=a,
        shifted_sample_cv=cv,
    )
    return output(location, scale)


def compute_field_statistics(data, threshold, lower_truncation=None):
    """Estimate rainfall occurrence and ordinary and latent marginal parameters.

    Parameters
    ----------
    data: array_like
        Nonempty, finite two-dimensional field. Units may be transformed rainfall
        (dBR for the reference parametric method). The input is not modified.
    threshold: float
        Finite rain-occurrence threshold in the same units as data. Values at
        or above the threshold are wet. The denominator is the whole valid grid.
    lower_truncation: float, optional
        Lower cutoff for fitting a latent Normal, at or above threshold. None
        skips fitting. This setting does not change rainfall occurrence.

    Returns
    -------
    result: dict
        Rain fraction, whole-field mean and standard deviation, ``ordinary_mean``
        and ``ordinary_stdev`` of wet pixels (population moments), ``latent_mean``
        and ``latent_stdev``, pixel counts, and ``nonzero_fit`` diagnostics.
        Unavailable moments are NaN. Estimates are not clipped or replaced.
    """
    if np.iscomplexobj(data):
        raise ValueError("data must contain real values")
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("data must be a nonempty two-dimensional field")
    if not np.isfinite(data).all():
        raise ValueError("data must contain only finite values")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if lower_truncation is not None and (
        not np.isfinite(lower_truncation) or lower_truncation < threshold
    ):
        raise ValueError("lower_truncation must be finite and at or above threshold")
    wet = data[data >= threshold]
    mean, stdev = (
        (float(np.mean(wet)), float(np.std(wet))) if wet.size else (np.nan, np.nan)
    )
    result = dict(
        rain_fraction=float(wet.size / data.size),
        mean=float(np.mean(data)),
        stdev=float(np.std(data)),
        ordinary_mean=mean,
        ordinary_stdev=stdev,
        latent_mean=np.nan,
        latent_stdev=np.nan,
        n_pixels=int(data.size),
        n_wet=int(wet.size),
        threshold=float(threshold),
        nonzero_fit={"status": "not_requested", "lower_truncation": None},
    )
    if lower_truncation is not None:
        location, scale, diagnostics = fit_lower_truncated_normal(
            wet, lower_truncation, return_diagnostics=True
        )
        diagnostics["lower_truncation"] = float(lower_truncation)
        result.update(latent_mean=location, latent_stdev=scale, nonzero_fit=diagnostics)
    return result
