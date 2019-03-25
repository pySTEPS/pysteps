"""
pysteps.postprocessing.ensemblestats
====================================

Methods for the computation of ensemble statistics.

.. autosummary::
    :toctree: ../generated/

    mean
    excprob
"""

import numpy as np


def mean(X, ignore_nan=False, X_thr=None):
    """Compute the mean value from a forecast ensemble field.

    Parameters
    ----------
    X : array_like
        Array of shape (n_members,m,n) containing an ensemble of forecast
        fields of shape (m,n).
    ignore_nan : bool
        If True, ignore nan values.
    X_thr : float
        Optional threshold for computing the ensemble mean. Values below X_thr
        are ignored.

    Returns
    -------
    out : ndarray
        Array of shape (m,n) containing the ensemble mean.
    """

    X = np.asanyarray(X)
    X_ndim = X.ndim

    if X_ndim > 3 or X_ndim <= 1:
        raise Exception('Number of dimensions of X should be 2 or 3.' +
                        'It was: {}'.format(X_ndim))
    elif X.ndim == 2:
        X = X[None, ...]

    if ignore_nan or X_thr is not None:
        if X_thr is not None:
            X = X.copy()
            X[X < X_thr] = np.nan

        return np.nanmean(X, axis=0)
    else:
        return np.mean(X, axis=0)


def excprob(X, X_thr, ignore_nan=False):
    """For a given forecast ensemble field, compute exceedance probabilities
    for the given intensity thresholds.

    Parameters
    ----------
    X : array_like
        Array of shape (k,m,n,...) containing an k-member ensemble of forecasts
        with shape (m,n,...).
    X_thr : float or a sequence of floats
        Intensity threshold(s) for which the exceedance probabilities are
        computed.
    ignore_nan : bool
        If True, ignore nan values.

    Returns
    -------
    out : ndarray
        Array of shape (len(X_thr),m,n) containing the exceedance probabilities
        for the given intensity thresholds.
        If len(X_thr)=1, the first dimension is dropped.
    """
    #  Checks
    X = np.asanyarray(X)
    X_ndim = X.ndim

    if X_ndim < 3:
        raise Exception('Number of dimensions of X should be 3 or more.' +
                        ' It was: {}'.format(X_ndim))

    P = []

    if np.isscalar(X_thr):
        X_thr = [X_thr]
        scalar_thr = True
    else:
        scalar_thr = False

    for x in X_thr:
        X_ = X.copy()

        X_[X >= x] = 1.0
        X_[X < x] = 0.0

        if ignore_nan:
            P.append(np.nanmean(X_, axis=0))
        else:
            P.append(np.mean(X_, axis=0))

    if not scalar_thr:
        return np.stack(P)
    else:
        return P[0]
