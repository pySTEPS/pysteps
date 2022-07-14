# -*- coding: utf-8 -*-
"""
pysteps.utils.tapering
======================

Implementations of window functions for computing of the FFT.

.. autosummary::
    :toctree: ../generated/

    compute_mask_window_function
    compute_window_function
"""

import numpy as np
from scipy.spatial import cKDTree


def compute_mask_window_function(mask, func, **kwargs):
    """
    Compute window function for a two-dimensional area defined by a
    non-rectangular mask. The window function is computed based on the distance
    to the nearest boundary point of the mask. Window function-specific
    parameters are given as keyword arguments.

    Parameters
    ----------
    mask: array_like
        Two-dimensional boolean array containing the mask.
        Pixels with True/False are inside/outside the mask.
    func: str
        The name of the window function. The currently implemented function is
        'tukey'.

    Returns
    -------
    out: array
        Array containing the tapering weights.
    """
    R = _compute_mask_distances(mask)

    if func == "hann":
        raise NotImplementedError("Hann function has not been implemented")
    elif func == "tukey":
        r_max = kwargs.get("r_max", 10.0)

        return _tukey_masked(R, r_max, np.isfinite(R))
    else:
        raise ValueError("invalid window function '%s'" % func)


def compute_window_function(m, n, func, **kwargs):
    """
    Compute window function for a two-dimensional rectangular region. Window
    function-specific parameters are given as keyword arguments.

    Parameters
    ----------
    m: int
        Height of the array.
    n: int
        Width of the array.
    func: str
        The name of the window function.
        The currently implemented functions are
        'hann' and 'tukey'.

    Other Parameters
    ----------------
    alpha: float
        Applicable if func is 'tukey'.

    Notes
    -----
    Two-dimensional tapering weights are computed from one-dimensional window
    functions using w(r), where r is the distance from the center of the
    region.

    Returns
    -------
    out: array
        Array of shape (m, n) containing the tapering weights.
    """
    X, Y = np.meshgrid(np.arange(n), np.arange(m))
    R = np.sqrt((X - int(n / 2)) ** 2 + (Y - int(m / 2)) ** 2)

    if func == "hann":
        return _hann(R)
    elif func == "tukey":
        alpha = kwargs.get("alpha", 0.2)

        return _tukey(R, alpha)
    else:
        raise ValueError("invalid window function '%s'" % func)


def _compute_mask_distances(mask):
    X, Y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))

    tree = cKDTree(np.vstack([X[~mask], Y[~mask]]).T)
    r, i = tree.query(np.vstack([X[mask], Y[mask]]).T, k=1)

    R = np.ones(mask.shape) * np.nan
    R[Y[mask], X[mask]] = r

    return R


def _hann(R):
    W = np.ones_like(R)
    N = min(R.shape[0], R.shape[1])
    mask = R > int(N / 2)

    W[mask] = 0.0
    W[~mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * (R[~mask] + int(N / 2)) / N))

    return W


def _tukey(R, alpha):
    W = np.ones_like(R)
    N = min(R.shape[0], R.shape[1])

    mask1 = R < int(N / 2)
    mask2 = R > int(N / 2) * (1.0 - alpha)
    mask = np.logical_and(mask1, mask2)
    W[mask] = 0.5 * (
        1.0 + np.cos(np.pi * (R[mask] / (alpha * 0.5 * N) - 1.0 / alpha + 1.0))
    )
    mask = R >= int(N / 2)
    W[mask] = 0.0

    return W


def _tukey_masked(R, r_max, mask):
    W = np.ones_like(R)

    mask_r = R < r_max
    mask_ = np.logical_and(mask, mask_r)
    W[mask_] = 0.5 * (1.0 + np.cos(np.pi * (R[mask_] / r_max - 1.0)))
    W[~mask] = np.nan

    return W
