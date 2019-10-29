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

def temporal_autocorrelation(X, domain="spatial", X_shape=None, MASK=None):
    """Compute lag-l autocorrelation coefficients gamma_l, l=1,2,...,n-1, for a
    time series of n two-dimensional input fields in spatial or spectral domain.

    Parameters
    ----------
    X : array_like
        Two-dimensional array of shape (n, w, h) containing a time series of n
        two-dimensional fields of shape (w, h). The input fields are assumed to
        be in increasing order with respect to time, and the time step is assumed
        to be regular (i.e. no missing data). X is required to have
        finite values.
    domain : {"spatial", "spectral"}
        The domain of the arrays X. If domain is "spectral", the arrays are
        assumed to represent the FFTs of the original arrays.
    X_shape : tuple
        The shape of the original arrays in the spatial domain before the FFT.
        Required if domain is "spectral".
    MASK : array_like
        Optional mask to use for computing the correlation coefficients. Pixels
        with MASK==False are excluded from the computations. Applicable if domain
        is "spatial".

    Returns
    -------
    out : ndarray
        Array of length n-1 containing the temporal autocorrelation coefficients
        for time lags l=1,2,...,n-1.

    """
    if len(X.shape) != 3:
        raise ValueError("the input X is not three-dimensional array")
    if MASK is not None and MASK.shape != X.shape[1:3]:
        raise ValueError("dimension mismatch between X and MASK: X.shape=%s, MASK.shape=%s" % \
                         (str(X.shape), str(MASK.shape)))
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values")

    if domain == "spatial" and MASK is None:
        MASK = np.ones((X.shape[1], X.shape[2]), dtype=bool)

    if domain == "spectral":
        X = X.copy()
        for k in range(X.shape[0]):
            X[k, 0, 0] = 0.0

    gamma = []
    for k in range(X.shape[0] - 1):
        if domain == "spatial":
            cc = np.corrcoef(X[-1, :, :][MASK], X[-(k+2), :, :][MASK])[0, 1]
        else:
            n = np.real(np.sum(X[-1, :, :]*np.conj(X[-(k+2), :, :])))
            d1 = np.sum(np.abs(X[-1, :, :])**2)
            d2 = np.sum(np.abs(X[-(k+2), :, :])**2)

            if X_shape[1] % 2 == 0:
                n += np.real(np.sum(X[-1, :, :][:, 1:-1]*np.conj(X[-(k+2), :, :][:, 1:-1])))
                d1 += np.sum(np.abs(X[-1, :, :][:, 1:-1])**2)
                d2 += np.sum(np.abs(X[-(k+2), :, :][:, 1:-1])**2)
            else:
                n += np.real(np.sum(X[-1, :, :][:, 1:]*np.conj(X[-(k+2), :, :][:, 1:])))
                d1 += np.sum(np.abs(X[-1, :, :][:, 1:])**2)
                d2 += np.sum(np.abs(X[-(k+2), :, :][:, 1:])**2)

            cc = n / np.sqrt(d1*d2)
        gamma.append(cc)

    return gamma
