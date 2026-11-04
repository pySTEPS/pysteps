"""
pysteps.noise.parametric
========================

Statistically parameterized rainfall fields with inverse spectral calibration.

.. autosummary::
    :toctree: ../generated/

    power_law_filter
    generate_stochastic_field
    generate_calibrated_field
"""

from collections.abc import Mapping

import numpy as np

from pysteps.postprocessing.probmatching import match_rainfall_distribution
from pysteps.utils.spectral import fit_power_law_spectrum, rapsd_logarithmic

GENERATOR_ID = "inverse_calibrated_v2"
SPECTRAL_BIAS = -0.5
LOW_WAR_THRESHOLD = 0.1
MAX_CALIBRATION_ATTEMPTS = 5
SPECTRAL_DISTANCE_THRESHOLD = 1.0
BETA_VARIANCES = (0.067738, 0.076405)


def _finite(name, value):
    if not np.isscalar(value) or np.iscomplexobj(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _shape(shape):
    try:
        shape = tuple(shape)
    except TypeError:
        raise ValueError(
            "shape must contain two positive integer dimensions (y, x)"
        ) from None
    if len(shape) != 2 or any(
        not isinstance(n, (int, np.integer)) or n < 1 for n in shape
    ):
        raise ValueError("shape must contain two positive integer dimensions (y, x)")
    return tuple(int(n) for n in shape)


def _parameters(parameters):
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping")
    names = ("beta_1", "beta_2", "rain_fraction", "mean_db", "stdev_db")
    if any(name not in parameters for name in names):
        raise ValueError(f"parameters must contain {', '.join(names)}")
    values = {name: _finite(name, parameters[name]) for name in names}
    if not 0 <= values["rain_fraction"] <= 1:
        raise ValueError("rain_fraction must lie in [0, 1]")
    if values["stdev_db"] <= 0:
        raise ValueError("stdev_db must be positive")
    return values


def _random_generator(rng, seed):
    if rng is not None:
        if seed is not None:
            raise ValueError("supply rng or seed, not both")
        if not isinstance(rng, (np.random.Generator, np.random.RandomState)):
            raise ValueError("rng must be a NumPy Generator or RandomState")
        return rng
    return np.random.default_rng(seed)


def power_law_filter(shape, beta_1, beta_2=None, kmperpixel=1.0, scale_break=None):
    """Construct a full-FFT amplitude filter from signed power-spectrum slopes.

    Parameters
    ----------
    shape: tuple of int
        Output shape (y, x), each dimension positive.
    beta_1, beta_2: float
        Signed power slopes, power proportional to frequency**beta. The second
        slope is optional; provide beta_2 and scale_break together.
    kmperpixel: float, optional
        Positive isotropic pixel spacing in km (default 1).
    scale_break: float, optional
        Positive break wavelength in km. beta_1 applies below its reciprocal
        frequency, beta_2 at and above it, with continuity at the break.

    Returns
    -------
    filter: ndarray
        Finite full-FFT amplitude filter of shape (y, x), using half the power
        exponents. DC amplitude equals one. The reference frequency is the
        smallest nonzero frequency along the longer grid dimension.
    """
    shape = _shape(shape)
    beta_1 = _finite("beta_1", beta_1)
    spacing = _finite("kmperpixel", kmperpixel)
    if spacing <= 0:
        raise ValueError("kmperpixel must be positive")
    if (beta_2 is None) != (scale_break is None):
        raise ValueError("provide beta_2 and scale_break together")
    fy = np.fft.fftfreq(shape[0], d=spacing)
    fx = np.fft.fftfreq(shape[1], d=spacing)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    reference = 1 / (max(shape) * spacing)
    result = np.ones(shape)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if beta_2 is None:
            selected = radius > 0
            result[selected] = (radius[selected] / reference) ** (beta_1 / 2)
        else:
            beta_2 = _finite("beta_2", beta_2)
            scale_break = _finite("scale_break", scale_break)
            if scale_break <= 0:
                raise ValueError("scale_break must be positive")
            breakpoint = 1 / scale_break
            low = radius < breakpoint
            result[low] = (radius[low] / reference) ** (beta_1 / 2)
            result[~low] = (breakpoint / reference) ** (beta_1 / 2) * (
                radius[~low] / breakpoint
            ) ** (beta_2 / 2)
        result[radius == 0] = 1.0
    if not np.isfinite(result).all():
        raise ValueError("parameters produce a nonfinite spectral filter")
    return result


def _field_from_filter(
    spectrum_filter, parameters, threshold, rng, zerovalue, nbins, eps
):
    white_noise = rng.uniform(low=0, high=1, size=spectrum_filter.shape)
    filtered = np.fft.ifft2(
        np.fft.fft2(white_noise, spectrum_filter.shape) * spectrum_filter
    ).real
    return match_rainfall_distribution(
        filtered,
        parameters["rain_fraction"],
        parameters["mean_db"],
        parameters["stdev_db"],
        threshold,
        zerovalue=zerovalue,
        nbins=nbins,
        eps=eps,
    )


def generate_stochastic_field(
    shape,
    parameters,
    kmperpixel,
    scale_break,
    threshold,
    *,
    rng=None,
    seed=None,
    zerovalue=None,
    nbins=500,
    eps=0.001,
):
    """Generate a rainfall-marginal field without inverse spectral correction.

    Parameters
    ----------
    shape: tuple of int
        Output array shape (y, x).
    parameters: mapping
        Finite beta_1, beta_2, rain_fraction, mean_db, and positive stdev_db.
        The last two are caller-selected parent-Normal generation moments.
        The mapping is not modified; no regional QA or parameter clipping occurs.
    kmperpixel, scale_break: float
        Positive pixel spacing and spectral break wavelength, both in km.
    threshold: float
        Finite rain/no-rain threshold in transformed units (dBR for the reference
        method). The returned field has the same transformed units.
    rng: numpy.random.Generator or RandomState, optional
        Caller-owned random stream; advanced in place.
    seed: int or SeedSequence, optional
        Seed for a local default_rng when rng is absent. Supply only one of them.
    zerovalue: float, optional
        Finite dry value below threshold, default threshold-0.1.
    nbins, eps: int, float, optional
        Marginal-mapping bins and quantile clip (defaults 500 and 0.001).

    Returns
    -------
    field: ndarray
        Float64 field of shape (y, x). Independent U(0,1) inputs are spectrally
        filtered, then mapped to the requested occurrence and Normal marginal.
        Dry anomalies remain continuous. This field is not globally standardized
        or converted to linear rainfall. Occurrence and marginal matching are
        approximate. An explicit zero fraction returns a constant dry field.
    """
    parameters = _parameters(parameters)
    threshold = _finite("threshold", threshold)
    spectrum_filter = power_law_filter(
        shape, parameters["beta_1"], parameters["beta_2"], kmperpixel, scale_break
    )
    rng = _random_generator(rng, seed)
    if parameters["rain_fraction"] == 0:
        # Validate mapping settings without consuming a random draw.
        return match_rainfall_distribution(
            np.zeros(spectrum_filter.shape),
            0,
            parameters["mean_db"],
            parameters["stdev_db"],
            threshold,
            zerovalue=zerovalue,
            nbins=nbins,
            eps=eps,
        )
    return _field_from_filter(
        spectrum_filter, parameters, threshold, rng, zerovalue, nbins, eps
    )


def _estimate_beta(field, threshold, zerovalue, scale_break_pixels):
    truncated = field.copy()
    truncated[truncated <= threshold] = zerovalue
    try:
        frequency, power = rapsd_logarithmic(truncated)
        result = fit_power_law_spectrum(frequency, power, scale_break_pixels)
    except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan
    return float(result["beta_1"]), float(result["beta_2"])


def generate_calibrated_field(
    shape,
    parameters,
    kmperpixel,
    scale_break,
    threshold,
    *,
    rng=None,
    seed=None,
    zerovalue=None,
    spectral_bias=SPECTRAL_BIAS,
    low_war_threshold=LOW_WAR_THRESHOLD,
    max_attempts=MAX_CALIBRATION_ATTEMPTS,
    distance_threshold=SPECTRAL_DISTANCE_THRESHOLD,
    beta_variances=BETA_VARIANCES,
    nbins=500,
    eps=0.001,
    return_diagnostics=False,
):
    """Generate a field with inverse calibration and assessed realised spectral slopes.

    Parameters
    ----------
    shape, parameters, kmperpixel, scale_break, threshold:
        As in :func:`generate_stochastic_field`. Each dimension must contain at
        least four pixels for spectral assessment. Parameters are not modified.
    rng, seed, zerovalue, nbins, eps:
        As in :func:`generate_stochastic_field`.
    spectral_bias: float, optional
        Additive correction to both input power slopes (default -0.5). Location
        and scale receive no inverse correction in this v2 method.
    low_war_threshold: float, optional
        Fractions at or below this value permit retries (default 0.1).
    max_attempts: int, optional
        Maximum attempts for low fractions (default 5). Higher fractions use one.
    distance_threshold: float, optional
        Accept when sqrt(sum((estimated_beta-target_beta)**2/beta_variances)) is
        at or below this nonnegative tolerance (default 1).
    beta_variances: array_like
        Two positive finite calibration error variances. Defaults are
        (0.067738, 0.076405); these weights are empirical, not universal errors.
    return_diagnostics: bool, optional
        Also return selected spectral estimates, distance, attempted/selected
        candidate indices, tolerance status, targets, effective calibration
        settings, and per-candidate estimates and distances.

    Returns
    -------
    field: ndarray
        Selected float64 rainfall-marginal field, not standardized. Slopes are
        assessed on a copy with values <= threshold replaced by zerovalue.
    diagnostics: dict, optional
        First in-tolerance candidate is selected. Otherwise use the closest
        finite-distance candidate, retaining the first tie. If all spectra are
        unavailable, use the last candidate and report unavailable_spectrum.

    Notes
    -----
    Preserves ``inverse_calibrated_v2`` defaults and selection policy. Calibration
    need not meet its tolerance; the outcome is explicit. A confirmed zero target
    returns dry output with zero attempts and no spectral-success claim. Invalid
    parameters or an unavailable positive-fraction marginal mapping raise errors.
    This function does not perform operational moment selection or slope clipping.
    """
    parameters = _parameters(parameters)
    shape = _shape(shape)
    if min(shape) < 4:
        raise ValueError("calibrated shape must have at least four pixels per axis")
    threshold = _finite("threshold", threshold)
    spectral_bias = _finite("spectral_bias", spectral_bias)
    low_war_threshold = _finite("low_war_threshold", low_war_threshold)
    distance_threshold = _finite("distance_threshold", distance_threshold)
    if not 0 <= low_war_threshold <= 1 or distance_threshold < 0:
        raise ValueError(
            "low_war_threshold must lie in [0,1] and distance_threshold must be nonnegative"
        )
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if np.iscomplexobj(beta_variances):
        raise ValueError("beta_variances must contain real values")
    variances = np.asarray(beta_variances, dtype=float)
    if (
        variances.shape != (2,)
        or not np.isfinite(variances).all()
        or np.any(variances <= 0)
    ):
        raise ValueError("beta_variances must contain two positive finite values")
    zerovalue = (
        threshold - 0.1 if zerovalue is None else _finite("zerovalue", zerovalue)
    )
    calibrated = dict(
        parameters,
        beta_1=parameters["beta_1"] + spectral_bias,
        beta_2=parameters["beta_2"] + spectral_bias,
    )
    spectrum_filter = power_law_filter(
        shape, calibrated["beta_1"], calibrated["beta_2"], kmperpixel, scale_break
    )
    rng = _random_generator(rng, seed)
    settings = dict(
        kmperpixel=float(kmperpixel),
        scale_break=float(scale_break),
        threshold=threshold,
        zerovalue=zerovalue,
        spectral_bias=spectral_bias,
        low_war_threshold=low_war_threshold,
        max_attempts=int(max_attempts),
        distance_threshold=distance_threshold,
        beta_variances=variances.copy(),
        nbins=nbins,
        eps=eps,
    )
    estimates, distances = [], []

    def output(candidate, status, tolerance_achieved):
        field, beta_1, beta_2, distance, selected = candidate
        diagnostics = dict(
            generator=GENERATOR_ID,
            status=status,
            estimated_beta_1=beta_1,
            estimated_beta_2=beta_2,
            distance=distance,
            attempts=len(distances),
            selected_attempt=selected,
            tolerance_achieved=tolerance_achieved,
            target_parameters=parameters.copy(),
            calibrated_parameters=calibrated.copy(),
            settings=settings,
            candidate_estimates=np.asarray(estimates).reshape(-1, 2),
            candidate_distances=np.asarray(distances),
        )
        return (field, diagnostics) if return_diagnostics else field

    if parameters["rain_fraction"] == 0:
        field = match_rainfall_distribution(
            np.zeros(shape),
            0,
            parameters["mean_db"],
            parameters["stdev_db"],
            threshold,
            zerovalue=zerovalue,
            nbins=nbins,
            eps=eps,
        )
        return output((field, np.nan, np.nan, np.nan, 0), "dry", False)
    best = last = None
    limit = max_attempts if parameters["rain_fraction"] <= low_war_threshold else 1
    for attempt in range(1, limit + 1):
        field = _field_from_filter(
            spectrum_filter, calibrated, threshold, rng, zerovalue, nbins, eps
        )
        beta_1, beta_2 = _estimate_beta(
            field, threshold, zerovalue, scale_break / kmperpixel
        )
        error = np.array([beta_1 - parameters["beta_1"], beta_2 - parameters["beta_2"]])
        distance = (
            float(np.sqrt(np.sum(error**2 / variances)))
            if np.isfinite(error).all()
            else np.inf
        )
        estimates.append([beta_1, beta_2])
        distances.append(distance)
        last = (field, beta_1, beta_2, distance, attempt)
        if distance <= distance_threshold:
            return output(last, "tolerance_achieved", True)
        if np.isfinite(distance) and (best is None or distance < best[3]):
            best = last
    return output(
        best if best is not None else last,
        "best_available" if best is not None else "unavailable_spectrum",
        False,
    )
