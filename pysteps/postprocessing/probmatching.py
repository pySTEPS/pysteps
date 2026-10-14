# -*- coding: utf-8 -*-
"""
pysteps.postprocessing.probmatching
===================================

Methods for matching the probability distribution of two data sets.

.. autosummary::
    :toctree: ../generated/

    compute_empirical_cdf
    nonparam_match_empirical_cdf
    pmm_init
    pmm_compute
    shift_scale
    resample_distributions
    match_rainfall_distribution
"""

import numpy as np
from scipy import interpolate as sip
from scipy import optimize as sop
from scipy import stats as sstats


def match_rainfall_distribution(
    field,
    rain_fraction,
    mean,
    stdev,
    threshold,
    *,
    zerovalue=None,
    min_rain_fraction=0.0,
    nbins=500,
    eps=0.001,
):
    """Map a finite field to a prescribed occurrence and Normal rainfall marginal.

    Parameters
    ----------
    field: array_like
        Nonempty, finite real 2D field, converted to float64 without modification.
    rain_fraction: float
        Requested wet fraction in [0, 1]. A histogram cutoff selects the wet
        pixels before marginal mapping; the final wet fraction is approximate.
    mean, stdev: float
        Finite parent-Normal location and positive scale in transformed units.
        These are caller-selected generation moments, not necessarily ordinary
        retained wet-pixel moments. No moment selection is performed here.
    threshold: float
        Rain/no-rain threshold in the same transformed units.
    zerovalue: float, optional
        Finite dry value below threshold, default threshold-0.1. Used for an
        explicit dry result; positive-fraction mapping retains continuous dry
        anomalies below threshold, as in the source generator.
    min_rain_fraction: float, optional
        Explicit dry-output policy for fractions below this value. Default 0
        suppresses no positive fraction. The source post-AR adjustment uses 0.025.
    nbins: int, optional
        Number of histogram bins (default 500).
    eps: float, optional
        Normal-quantile probability clip in (0, 0.5), default 0.001.

    Returns
    -------
    mapped: ndarray
        Field of the same shape and transformed units. Zero requested fraction
        or a fraction below min_rain_fraction returns the configured dry value.

    Notes
    -----
    Preserves the source ``normalize_stoch_field`` mapping: find the last bin
    whose cumulative probability is <= 1-rain_fraction, shift its upper edge
    to threshold, and map the strictly-above-threshold sample's binned CDF to
    clipped Normal quantiles using bin-centre linear interpolation. Wet mapped
    values can cross back below threshold, and spectra can change. Neither
    occurrence nor marginal moments are exact guarantees. Unavailable or
    degenerate positive-fraction mappings raise ValueError, not a dry fallback.
    """
    if np.iscomplexobj(field):
        raise ValueError("field must contain real values")
    field = np.asarray(field, dtype=float)
    if field.ndim != 2 or field.size == 0 or not np.isfinite(field).all():
        raise ValueError("field must be a nonempty finite two-dimensional array")
    for name, value in [
        ("rain_fraction", rain_fraction),
        ("mean", mean),
        ("stdev", stdev),
        ("threshold", threshold),
        ("min_rain_fraction", min_rain_fraction),
        ("eps", eps),
    ]:
        if not np.isscalar(value) or np.iscomplexobj(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= rain_fraction <= 1 or not 0 <= min_rain_fraction <= 1:
        raise ValueError("rain fractions must lie in [0, 1]")
    if stdev <= 0:
        raise ValueError("stdev must be positive")
    if not isinstance(nbins, (int, np.integer)) or nbins < 2:
        raise ValueError("nbins must be an integer of at least two")
    if not 0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    zerovalue = threshold - 0.1 if zerovalue is None else zerovalue
    if (
        not np.isscalar(zerovalue)
        or np.iscomplexobj(zerovalue)
        or not np.isfinite(zerovalue)
        or zerovalue >= threshold
    ):
        raise ValueError("zerovalue must be finite and below threshold")
    if rain_fraction == 0 or rain_fraction < min_rain_fraction:
        return np.full_like(field, zerovalue)

    distribution = sstats.cumfreq(field, numbins=nbins)
    edges = distribution.lowerlimit + np.arange(nbins + 1) * distribution.binsize
    cdf = distribution.cumcount / distribution.cumcount[-1]
    below = np.flatnonzero(cdf <= 1 - rain_fraction)
    rain_bin = int(below[-1]) if below.size else 0
    mapped = field + (threshold - edges[rain_bin + 1])
    wet = mapped > threshold
    sample = mapped[wet]
    if sample.size < 2 or np.ptp(sample) == 0:
        raise ValueError("mapping produced too few distinct raining pixels")
    distribution = sstats.cumfreq(sample, numbins=nbins)
    edges = distribution.lowerlimit + np.arange(nbins + 1) * distribution.binsize
    centres = 0.5 * (edges[:-1] + edges[1:])
    cdf = distribution.cumcount / distribution.cumcount[-1]
    quantiles = sstats.norm.ppf(np.clip(cdf, eps, 1 - eps), loc=mean, scale=stdev)
    transform = sip.interp1d(
        centres,
        quantiles,
        kind="linear",
        bounds_error=False,
        fill_value=(quantiles[0], quantiles[-1]),
    )
    mapped[wet] = transform(sample)
    if not np.isfinite(mapped).all():
        raise ValueError("marginal mapping produced nonfinite values")
    return mapped


def compute_empirical_cdf(bin_edges, hist):
    """
    Compute an empirical cumulative distribution function from the given
    histogram.

    Parameters
    ----------
    bin_edges: array_like
        Coordinates of left edges of the histogram bins.
    hist: array_like
        Histogram counts for each bin.

    Returns
    -------
    out: ndarray
        CDF values corresponding to the bin edges.

    """
    cdf = []
    xs = 0.0

    for x, h in zip(zip(bin_edges[:-1], bin_edges[1:]), hist):
        cdf.append(xs)
        xs += (x[1] - x[0]) * h

    cdf.append(xs)
    cdf = np.array(cdf) / xs

    return cdf


def nonparam_match_empirical_cdf(initial_array, target_array, ignore_indices=None):
    """
    Matches the empirical CDF of the initial array with the empirical CDF
    of a target array. Initial ranks are conserved, but empirical distribution
    matches the target one. Zero-pixels (i.e. pixels having the minimum value)
    in the initial array are conserved.

    Parameters
    ----------
    initial_array: array_like
        The initial array whose CDF is to be matched with the target.
    target_array: array_like
        The target array
    ignore_indices: array_like, optional
        Indices of pixels in the initial_array which are to be ignored (not
        rescaled) or an array of booleans with True at the pixel locations to
        be ignored in initial_array and False elsewhere.


    Returns
    -------
    output_array: ndarray
        The matched array of the same shape as the initial array.
    """

    if np.all(np.isnan(initial_array)):
        raise ValueError("Initial array contains only nans.")
    if initial_array.size != target_array.size:
        raise ValueError(
            "dimension mismatch between initial_array and target_array: "
            f"initial_array.shape={initial_array.shape}, target_array.shape={target_array.shape}"
        )

    initial_array_copy = np.array(initial_array, dtype=float)
    target_array = np.array(target_array, dtype=float)

    # Determine zero in initial array and set nans to zero
    zvalue = np.nanmin(initial_array_copy)
    if ignore_indices is not None:
        initial_array_copy[ignore_indices] = zvalue
    # Check if there are still nans left after setting the values at ignore_indices to zero.
    if np.any(~np.isfinite(initial_array_copy)):
        raise ValueError(
            "Initial array contains non-finite values outside ignore_indices mask."
        )

    idxzeros = initial_array_copy == zvalue

    # Determine zero of target_array and set nans to zero.
    zvalue_trg = np.nanmin(target_array)
    target_array = np.where(np.isnan(target_array), zvalue_trg, target_array)

    # adjust the fraction of rain in target distribution if the number of
    # nonzeros is greater than in the initial array (the lowest values will be set to zero)
    if np.sum(target_array > zvalue_trg) > np.sum(initial_array_copy > zvalue):
        war = np.sum(initial_array_copy > zvalue) / initial_array_copy.size
        p = np.percentile(target_array, 100 * (1 - war))
        target_array[target_array < p] = zvalue_trg

    # flatten the arrays without copying them
    arrayshape = initial_array_copy.shape
    target_array = target_array.reshape(-1)
    initial_array_copy = initial_array_copy.reshape(-1)

    # rank target values
    order = target_array.argsort()
    ranked = target_array[order]

    # rank initial values order
    orderin = initial_array_copy.argsort()
    ranks = np.empty(len(initial_array_copy), int)
    ranks[orderin] = np.arange(len(initial_array_copy))

    # get ranked values from target and rearrange with the initial order
    output_array = ranked[ranks]

    # reshape to the original array dimensions
    output_array = output_array.reshape(arrayshape)

    # read original zeros
    output_array[idxzeros] = zvalue_trg

    # Put back the original values outside the nan-mask of the target array.
    if ignore_indices is not None:
        output_array[ignore_indices] = initial_array[ignore_indices]
    return output_array


# TODO: A more detailed explanation of the PMM method + references.
def pmm_init(bin_edges_1, cdf_1, bin_edges_2, cdf_2):
    """
    Initialize a probability matching method (PMM) object from binned
    cumulative distribution functions (CDF).

    Parameters
    ----------
    bin_edges_1: array_like
        Coordinates of the left bin edges of the source cdf.
    cdf_1: array_like
        Values of the source CDF at the bin edges.
    bin_edges_2: array_like
        Coordinates of the left bin edges of the target cdf.
    cdf_2: array_like
        Values of the target CDF at the bin edges.
    """
    pmm = {}

    pmm["bin_edges_1"] = bin_edges_1.copy()
    pmm["cdf_1"] = cdf_1.copy()
    pmm["bin_edges_2"] = bin_edges_2.copy()
    pmm["cdf_2"] = cdf_2.copy()
    pmm["cdf_interpolator"] = sip.interp1d(bin_edges_1, cdf_1, kind="linear")

    return pmm


def pmm_compute(pmm, x):
    """
    For a given PMM object and x-coordinate, compute the probability matched
    value (i.e. the x-coordinate for which the target CDF has the same value as
    the source CDF).

    Parameters
    ----------
    pmm: dict
        A PMM object returned by pmm_init.
    x: float
        The coordinate for which to compute the probability matched value.
    """
    mask = np.logical_and(x >= pmm["bin_edges_1"][0], x <= pmm["bin_edges_1"][-1])
    p = pmm["cdf_interpolator"](x[mask])

    result = np.ones(len(mask)) * np.nan
    result[mask] = _invfunc(p, pmm["bin_edges_2"], pmm["cdf_2"])

    return result


def shift_scale(R, f, rain_fraction_trg, second_moment_trg, **kwargs):
    """
    Find shift and scale that is needed to return the required second_moment
    and rain area. The optimization is performed with the Nelder-Mead algorithm
    available in scipy.
    It assumes a forward transformation ln_rain = ln(rain)-ln(min_rain) if
    rain > min_rain, else 0.

    Parameters
    ----------
    R: array_like
        The initial array to be shift and scaled.
    f: function
        The inverse transformation that is applied after the shift and scale.
    rain_fraction_trg: float
        The required rain fraction to be matched by shifting.
    second_moment_trg: float
        The required second moment to be matched by scaling.
        The second_moment is defined as second_moment = var + mean^2.

    Other Parameters
    ----------------
    scale: float
        Optional initial value of the scale parameter for the Nelder-Mead
        optimisation.
        Typically, this would be the scale parameter estimated the previous
        time step.
        Default: 1.
    max_iterations: int
        Maximum allowed number of iterations and function evaluations.
        More details: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
        Deafult: 100.
    tol: float
        Tolerance for termination.
        More details: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
        Default: 0.05*second_moment_trg, i.e. terminate the search if the error
        is less than 5% since the second moment is a bit unstable.

    Returns
    -------
    shift: float
        The shift value that produces the required rain fraction.
    scale: float
        The scale value that produces the required second_moment.
    R: array_like
        The shifted, scaled and back-transformed array.
    """

    shape = R.shape
    R = R.flatten()

    # defaults
    scale = kwargs.get("scale", 1.0)
    max_iterations = kwargs.get("max_iterations", 100)
    tol = kwargs.get("tol", 0.05 * second_moment_trg)

    # calculate the shift parameter based on the required rain fraction
    shift = np.percentile(R, 100 * (1 - rain_fraction_trg))
    idx_wet = R > shift

    # define objective function
    def _get_error(scale):
        R_ = np.zeros_like(R)
        R_[idx_wet] = f((R[idx_wet] - shift) * scale)
        R_[~idx_wet] = 0
        second_moment = np.nanstd(R_) ** 2 + np.nanmean(R_) ** 2
        return np.abs(second_moment - second_moment_trg)

    # Nelder-Mead optimisation
    nm_scale = sop.minimize(
        _get_error,
        scale,
        method="Nelder-Mead",
        tol=tol,
        options={"disp": False, "maxiter": max_iterations},
    )
    scale = nm_scale["x"][0]

    R[idx_wet] = f((R[idx_wet] - shift) * scale)
    R[~idx_wet] = 0

    return shift, scale, R.reshape(shape)


def resample_distributions(
    first_array, second_array, probability_first_array, randgen=np.random
):
    """
    Merges two distributions (e.g., from the extrapolation nowcast and NWP in the blending module)
    to effectively combine two distributions for probability matching without losing extremes.
    Entries for which one array has a nan will not be included from the other array either.

    Parameters
    ----------
    first_array: array_like
        One of the two arrays from which the distribution should be sampled (e.g., the extrapolation
        cascade). It must be of the same shape as `second_array`. Input must not contain NaNs.
    second_array: array_like
        One of the two arrays from which the distribution should be sampled (e.g., the NWP (model)
        cascade). It must be of the same shape as `first_array`. Input must not contain NaNs.
    probability_first_array: float
        The weight that `first_array` should get (a value between 0 and 1). This determines the
        likelihood of selecting elements from `first_array` over `second_array`.
    randgen: numpy.random or numpy.RandomState
        The random number generator to be used for the binomial distribution. You can pass a seeded
        random state here for reproducibility. Default is numpy.random.

    Returns
    -------
    csort: array_like
        The combined output distribution. This is an array of the same shape as the input arrays,
        where each element is chosen from either `first_array` or `second_array` based on the specified
        probability, and then sorted in descending order.

    Raises
    ------
    ValueError
        If `first_array` and `second_array` do not have the same shape.
    """

    # Valide inputs
    if first_array.shape != second_array.shape:
        raise ValueError("first_array and second_array must have the same shape")
    probability_first_array = np.clip(probability_first_array, 0.0, 1.0)

    # Propagate the NaN values of the arrays to each other if there are any; convert to float to make sure this works.
    nanmask = np.isnan(first_array) | np.isnan(second_array)
    if np.any(nanmask):
        first_array = first_array.astype(float)
        first_array[nanmask] = np.nan
        second_array = second_array.astype(float)
        second_array[nanmask] = np.nan

    # Flatten and sort the arrays
    asort = np.sort(first_array, axis=None)[::-1]
    bsort = np.sort(second_array, axis=None)[::-1]
    n = asort.shape[0]

    # Resample the distributions
    idxsamples = randgen.binomial(1, probability_first_array, n).astype(bool)
    csort = np.where(idxsamples, asort, bsort)
    csort = np.sort(csort)[::-1]

    # Return the resampled array in descending order (starting with the nan values)
    return csort


def _invfunc(y, fx, fy):
    if len(y) == 0:
        return np.array([])

    b = np.digitize(y, fy)
    mask = np.logical_and(b > 0, b < len(fy))
    c = (y[mask] - fy[b[mask] - 1]) / (fy[b[mask]] - fy[b[mask] - 1])

    result = np.ones(len(y)) * np.nan
    result[mask] = c * fx[b[mask]] + (1.0 - c) * fx[b[mask] - 1]

    return result
