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

def temporal_autocorrelation(X, MASK=None):
    """Compute lag-l autocorrelation coefficients gamma_l, l=1,2,...,n-1, for a
    time series of n two-dimensional input fields.

    Parameters
    ----------
    X : array_like
      Two-dimensional array of shape (n, w, h) containing a time series of n
      two-dimensional fields of shape (w, h). The input fields are assumed to
      be in increasing order with respect to time, and the time step is assumed
      to be regular (i.e. no missing data). X is required to have finite values.
    MASK : array_like
      Optional mask to use for computing the correlation coefficients. Pixels
      with MASK==False are excluded from the computations.

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

    gamma = np.empty(X.shape[0]-1)

    if MASK is None:
      MASK = np.ones((X.shape[1], X.shape[2]), dtype=bool)

    gamma = []
    for k in range(X.shape[0] - 1):
        gamma.append(np.corrcoef(X[-1, :, :][MASK], X[-(k+2), :, :][MASK])[0, 1])

    return gamma
