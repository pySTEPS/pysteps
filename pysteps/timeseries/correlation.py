"""
pysteps.timeseries.correlation
==============================

Methods for computing spatial and temporal correlation of time series of
two-dimensional fields.

.. autosummary::
    :toctree: ../generated/

    temporal_autocorrelation
"""

import numpy as np


def temporal_autocorrelation(x, mask=None):
    """Compute lag-l autocorrelation coefficients gamma_l, l=1,2,...,n-1, for a
    time series of n inputs of arbitrary dimension.

    Parameters
    ----------
    x : array_like
        Array of shape (n, ...) containing a time series of n inputs of arbitrary
        dimension. The inputs are assumed to be in increasing order with respect
        to time, and the time step is assumed to be regular. x is required to
        have finite values.
    mask : array_like
        Optional mask to use for computing the correlation coefficients. Input
        elements with mask==False are excluded from the computations.

    Returns
    -------
    out : list
        List of length n-1 containing the temporal autocorrelation coefficients
        for time lags l=1,2,...,n-1.

    """
    if len(x.shape) < 2:
        raise ValueError("the dimension of x must be >= 2")
    if mask is not None and mask.shape != x.shape[1:]:
        raise ValueError("dimension mismatch between x and mask: x.shape[1:]=%s, mask.shape=%s" % \
                         (str(x.shape), str(mask.shape)))
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    gamma = np.empty(x.shape[0]-1)

    if mask is None:
        mask = np.ones(x.shape[1:], dtype=bool)

    gamma = []
    for k in range(x.shape[0] - 1):
        gamma.append(np.corrcoef(x[-1, :][mask], x[-(k+2), :][mask])[0, 1])

    return gamma
