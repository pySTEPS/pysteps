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
"""

import numpy as np
from scipy import interpolate as sip
from scipy import optimize as sop


def compute_empirical_cdf(bin_edges, hist):
    """Compute an empirical cumulative distribution function from the given
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


def nonparam_match_empirical_cdf(initial_array, target_array):
    """Matches the empirical CDF of the initial array with the empirical CDF
    of a target array. Initial ranks are conserved, but empirical distribution
    matches the target one. Zero-pixels (i.e. pixels having the minimum value)
    in the initial array are conserved.

    Parameters
    ----------
    initial_array: array_like
        The initial array whose CDF is to be matched with the target.
    target_array: array_like
        The target array.

    Returns
    -------
    output_array: ndarray
        The matched array of the same shape as the initial array.
    """

    if np.any(~np.isfinite(initial_array)):
        raise ValueError("initial array contains non-finite values")
    if np.any(~np.isfinite(target_array)):
        raise ValueError("target array contains non-finite values")
    if initial_array.size != target_array.size:
        raise ValueError(
            "dimension mismatch between initial_array and target_array: "
            f"initial_array.shape={initial_array.shape}, target_array.shape={target_array.shape}"
        )

    initial_array = np.array(initial_array)
    target_array = np.array(target_array)

    # zeros in initial array
    zvalue = initial_array.min()
    idxzeros = initial_array == zvalue

    # zeros in the target array
    zvalue_trg = target_array.min()

    # adjust the fraction of rain in target distribution if the number of
    # nonzeros is greater than in the initial array
    if np.sum(target_array > zvalue_trg) > np.sum(initial_array > zvalue):
        war = np.sum(initial_array > zvalue) / initial_array.size
        p = np.percentile(target_array, 100 * (1 - war))
        target_array = target_array.copy()
        target_array[target_array < p] = zvalue_trg

    # flatten the arrays
    arrayshape = initial_array.shape
    target_array = target_array.flatten()
    initial_array = initial_array.flatten()

    # rank target values
    order = target_array.argsort()
    ranked = target_array[order]

    # rank initial values order
    orderin = initial_array.argsort()
    ranks = np.empty(len(initial_array), int)
    ranks[orderin] = np.arange(len(initial_array))

    # get ranked values from target and rearrange with the initial order
    output_array = ranked[ranks]

    # reshape to the original array dimensions
    output_array = output_array.reshape(arrayshape)

    # read original zeros
    output_array[idxzeros] = zvalue_trg

    return output_array


# TODO: A more detailed explanation of the PMM method + references.
def pmm_init(bin_edges_1, cdf_1, bin_edges_2, cdf_2):
    """Initialize a probability matching method (PMM) object from binned
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
    """For a given PMM object and x-coordinate, compute the probability matched
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
    """Find shift and scale that is needed to return the required second_moment
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


def _invfunc(y, fx, fy):
    if len(y) == 0:
        return np.array([])

    b = np.digitize(y, fy)
    mask = np.logical_and(b > 0, b < len(fy))
    c = (y[mask] - fy[b[mask] - 1]) / (fy[b[mask]] - fy[b[mask] - 1])

    result = np.ones(len(y)) * np.nan
    result[mask] = c * fx[b[mask]] + (1.0 - c) * fx[b[mask] - 1]

    return result
