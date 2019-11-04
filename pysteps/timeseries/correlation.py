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

def temporal_autocorrelation(x, domain="spatial", x_shape=None, mask=None):
    """Compute lag-l autocorrelation coefficients gamma_l, l=1,2,...,n-1, for a
    time series of n inputs of arbitraty dimension in spatial or spectral domain.

    Parameters
    ----------
    x : array_like
        Array of shape (n, ...) containing a time series of n inputs of arbitrary
        dimension. The inputs are assumed to be in increasing order with respect
        to time, and the time step is assumed to be regular. x is required to
        have finite values.
    domain : {"spatial", "spectral"}
        The domain of the time series x. If domain is "spectral", the elements
        of x are assumed to represent the FFTs of the original elements.
    x_shape : tuple
        The shape of the original arrays in the spatial domain before applying
        the FFT. Required if domain is "spectral".
    mask : array_like
        Optional mask to use for computing the correlation coefficients. Input
        elements with mask==False are excluded from the computations. Applicable
        if domain is "spatial".

    Returns
    -------
    out : ndarray
        Array of length n-1 containing the temporal autocorrelation coefficients
        for time lags l=1,2,...,n-1.

    Notes
    -----
    Computation of correlation coefficients in the spectral domain is currently
    implemented only for two-dimensional fields.

    """
    if len(x.shape) < 2:
        raise ValueError("the dimension of x must be >= 2")
    if len(x.shape) != 3 and domain == "spectral":
        raise NotImplementedError("with domain == 'spectral', this function has only been implemented for two-dimensional fields")
    if mask is not None and mask.shape != x.shape[1:]:
        raise ValueError("dimension mismatch between x and mask: x.shape[1:]=%s, mask.shape=%s" % \
                         (str(x.shape), str(mask.shape)))
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    if domain == "spatial" and mask is None:
        mask = np.ones(x.shape[1:], dtype=bool)

    if domain == "spectral":
        x = x.copy()
        for k in range(x.shape[0]):
            x[k, np.zeros(x.shape[1:], dtype=int)] = 0.0

    gamma = []
    for k in range(x.shape[0] - 1):
        if domain == "spatial":
            cc = np.corrcoef(x[-1, :][mask], x[-(k+2), :][mask])[0, 1]
        else:
            n = np.real(np.sum(x[-1, :, :]*np.conj(x[-(k+2), :, :])))
            d1 = np.sum(np.abs(x[-1, :, :])**2)
            d2 = np.sum(np.abs(x[-(k+2), :, :])**2)

            if x_shape[1] % 2 == 0:
                n += np.real(np.sum(x[-1, :, :][:, 1:-1]*np.conj(x[-(k+2), :, :][:, 1:-1])))
                d1 += np.sum(np.abs(x[-1, :, :][:, 1:-1])**2)
                d2 += np.sum(np.abs(x[-(k+2), :, :][:, 1:-1])**2)
            else:
                n += np.real(np.sum(x[-1, :, :][:, 1:]*np.conj(x[-(k+2), :, :][:, 1:])))
                d1 += np.sum(np.abs(x[-1, :, :][:, 1:])**2)
                d2 += np.sum(np.abs(x[-(k+2), :, :][:, 1:])**2)

            cc = n / np.sqrt(d1*d2)
        gamma.append(cc)

    return gamma
