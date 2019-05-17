"""
pysteps.utils.tapering
======================

Implementations of window functions for computing of the FFT.
"""

import numpy as np

def compute_window_function(m, n, func, **kwargs):
    """Compute a window function for a two-dimensional array. Window
    function-specific parameters are given as keyword arguments.

    Parameters
    ----------
    m : int
        Height of the array.   
    n : int
        Width of the array.   
    func : str
        The name of the window function. The currently implemented functions are
        'hann' and 'tukey'.

    Other Parameters
    ----------------
    alpha : float
        Applicable if func is 'tukey'.

    Notes
    -----
    Two-dimensional tapering weights are computed from one-dimensional window
    functions using w(r), where r is the distance from the window center.

    Returns
    -------
    out : array
        Array of shape (m, n) containing the tapering weights.
    """
    X, Y = np.meshgrid(np.arange(n), np.arange(m))
    R = np.sqrt((X - int(n/2))**2 + (Y - int(m/2))**2)

    if func == "hann":
        return _hann(R)
    elif func == "tukey":
        alpha = kwargs.get("alpha", 0.2)

        return _tukey(R, alpha)
    else:
        raise ValueError("invalid window function '%s'" % func)

def _hann(R):
    W = np.ones_like(R)
    N = min(R.shape[0], R.shape[1])
    mask = R > int(N / 2)

    W[mask] = 0.0
    W[~mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * (R[~mask] + int(N/2)) / N))

    return W

def _tukey(R, alpha):
    W = np.ones_like(R)
    N = min(R.shape[0], R.shape[1])

    mask1 = R < int(N/2)
    mask2 = R > int(N / 2) * (1.0 - alpha)
    mask = np.logical_and(mask1, mask2)
    W[mask] = 0.5 * (1.0 + np.cos(np.pi*(R[mask] / (alpha * 0.5 * N) - 1.0/alpha + 1.0)))
    mask = R >= int(N/2)
    W[mask] = 0.0

    return W

