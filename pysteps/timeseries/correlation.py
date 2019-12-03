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
    """Compute lag-l temporal autocorrelation coefficients
    :math:`\gamma_l=\mbox{corr}(x(t),x(t-l))`, :math:`l=1,2,\dots,n-1`,
    from a time series :math:`x_1,x_2,\dots,x_n`.

    Parameters
    ----------
    x : array_like
        Array of shape (n, ...) containing the time series :math:`x_i`. The
        inputs are assumed to be in increasing order with respect to time, and
        the time step is assumed to be regular. All inputs are required to have
        finite values. The remaining dimensions after the first one are
        flattened before computing the correlation coefficients.
    mask : array_like
        Optional mask to use for computing the correlation coefficients. Input
        elements with mask==False are excluded from the computations.

    Returns
    -------
    out : list
        List of length n-1 containing the temporal autocorrelation coefficients
        :math:`\gamma_i` for time lags :math:`l=1,2,...,n-1`.

    """
    if len(x.shape) < 2:
        raise ValueError("the dimension of x must be >= 2")
    if mask is not None and mask.shape != x.shape[1:]:
        raise ValueError("dimension mismatch between x and mask: x.shape[1:]=%s, mask.shape=%s" % \
                         (str(x.shape[1:]), str(mask.shape)))
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    gamma = np.empty(x.shape[0]-1)

    if mask is None:
        mask = np.ones(x.shape[1:], dtype=bool)

    gamma = []
    for k in range(x.shape[0] - 1):
        gamma.append(np.corrcoef(x[-1, :][mask], x[-(k+2), :][mask])[0, 1])

    return gamma


def temporal_autocorrelation_multivariate(x, d=0, mask=None):
    """For a :math:`q`-variate time series
    :math:`\mathbf{x}_1,\mathbf{x}_2,\dots,\mathbf{x}_n`, compute the lag-l
    correlation matrices :math:`\Gamma_l`, where :math:`\Gamma_{l,i,j}=\gamma_{l,i,j}`
    and :math:`\gamma_{l,i,j}=\mbox{corr}(x_i(t),x_j(t-l))` for
    :math:`i,j=1,2,\dots,q` and :math:`l=0,1,\dots,n-1`.

    Parameters
    ----------
    x : array_like
        Array of shape (q, n, ...) containing the time series :math:`\mathbf{x}_i`.
        The inputs are assumed to be in increasing order with respect to time,
        and the time step is assumed to be regular. All inputs are required to
        have finite values. The remaining dimensions after the second one are
        flattened before computing the correlation coefficients.
    d : int
        The order of differencing. If d>=1, a differencing operator
        :math:`\Delta=(1-L)^d`, where :math:`L` is a time lag operator, is
        applied before computing the correlation coefficients. In this case,
        a time series of length n+d is needed for computing the n correlation
        matrices.

    Returns
    -------
    out : list
        List of correlation matrices :math:`\Gamma_0,\Gamma_1,\dots,\Gamma_{n-1}`.

    References
    ----------
    :cite:`CP2002`
    """
    if len(x.shape) < 3:
        raise ValueError("the dimension of x must be >= 3")
    if mask is not None and mask.shape != x.shape[2:]:
        raise ValueError("dimension mismatch between x and mask: x.shape[2:]=%s, mask.shape=%s" % \
                         (str(x.shape[2:]), str(mask.shape)))
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    if d >= 1:
        x = np.diff(x, axis=1)

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
