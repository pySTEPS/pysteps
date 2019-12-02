"""
pysteps.timeseries.correlation
==============================

Methods for computing spatial and temporal correlations from time series data.

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


def temporal_autocorrelation_multivariate(x, mask=None):
    """For a :math:`q`-variate time series of length :math:`n`, compute the lag-l
    correlation matrices :math:`\Gamma_l`, where :math:`\Gamma_{l,i,j}=\gamma_{l,i,j}`
    and :math:`\gamma_{l,i,j}=\mbox{corr}(x_i(t),x_j(t-l))` for
    :math:`i,j=1,2,\dots,q` and :math:`l=1,2,\dots,n-1`.

    Parameters
    ----------
    x : array_like
        Array of shape (q, n, ...) containing a time series of n q-variate inputs
        of arbitrary dimension. The inputs are assumed to be in increasing order
        with respect to time, and the time step is assumed to be regular. x is
        required to have finite values.

    Returns
    -------
    out : list
        List of correlation matrices :math:`\Gamma_l`, :math:`l=1,2,\dots,n-1`.

    References
    ----------
    :cite:`CP2002`
    """
    if len(x.shape) < 3:
        raise ValueError("the dimension of x must be >= 3")

    p = x.shape[1] - 1
    q = x.shape[0]

    gamma = []
    for k in range(p+1):
        gamma_k = np.empty((q, q))
        for i in range(q):
            x_i = x[i, -1, :]
            for j in range(q):
                x_j = x[j, -(k+1), :]
                gamma_k[i, j] = np.corrcoef(x_i.flatten(), x_j.flatten())[0, 1]
        gamma.append(gamma_k)

    return gamma
