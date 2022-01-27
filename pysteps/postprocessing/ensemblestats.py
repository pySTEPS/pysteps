# -*- coding: utf-8 -*-
"""
pysteps.postprocessing.ensemblestats
====================================

Methods for the computation of ensemble statistics.

.. autosummary::
    :toctree: ../generated/

    mean
    excprob
    banddepth
"""

import numpy as np
from scipy.special import comb


def mean(X, ignore_nan=False, X_thr=None):
    """
    Compute the mean value from a forecast ensemble field.

    Parameters
    ----------
    X: array_like
        Array of shape (k, m, n) containing a k-member ensemble of forecast
        fields of shape (m, n).
    ignore_nan: bool
        If True, ignore nan values.
    X_thr: float
        Optional threshold for computing the ensemble mean.
        Values below **X_thr** are ignored.

    Returns
    -------
    out: ndarray
        Array of shape (m, n) containing the ensemble mean.
    """

    X = np.asanyarray(X)
    X_ndim = X.ndim

    if X_ndim > 3 or X_ndim <= 1:
        raise Exception(
            "Number of dimensions of X should be 2 or 3." + "It was: {}".format(X_ndim)
        )
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
    """
    For a given forecast ensemble field, compute exceedance probabilities
    for the given intensity thresholds.

    Parameters
    ----------
    X: array_like
        Array of shape (k, m, n, ...) containing an k-member ensemble of
        forecasts with shape (m, n, ...).
    X_thr: float or a sequence of floats
        Intensity threshold(s) for which the exceedance probabilities are
        computed.
    ignore_nan: bool
        If True, ignore nan values.

    Returns
    -------
    out: ndarray
        Array of shape (len(X_thr), m, n) containing the exceedance
        probabilities for the given intensity thresholds.
        If len(X_thr)=1, the first dimension is dropped.
    """
    #  Checks
    X = np.asanyarray(X)
    X_ndim = X.ndim

    if X_ndim < 3:
        raise Exception(
            f"Number of dimensions of X should be 3 or more. It was: {X_ndim}"
        )

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
        X_[~np.isfinite(X)] = np.nan

        if ignore_nan:
            P.append(np.nanmean(X_, axis=0))
        else:
            P.append(np.mean(X_, axis=0))

    if not scalar_thr:
        return np.stack(P)
    else:
        return P[0]


def banddepth(X, thr=None, norm=False):
    """
    Compute the modified band depth (Lopez-Pintado and Romo, 2009) for a
    k-member ensemble data set.

    Implementation of the exact fast algorithm for computing the modified band
    depth as described in Sun et al (2012).

    Parameters
    ----------
    X: array_like
        Array of shape (k, m, ...) representing an ensemble of *k* members
        (i.e., samples) with shape (m, ...).
    thr: float
        Optional threshold for excluding pixels that have no samples equal or
        above the **thr** value.

    Returns
    -------
    out: array_like
        Array of shape *k* containing the (normalized) band depth values for
        each ensemble member.

    References
    ----------
    Lopez-Pintado, Sara, and Juan Romo. 2009. "On the Concept of Depth for
    Functional Data." Journal of the American Statistical Association
    104 (486): 718–34. https://doi.org/10.1198/jasa.2009.0108.

    Sun, Ying, Marc G. Genton, and Douglas W. Nychka. 2012. "Exact Fast
    Computation of Band Depth for Large Functional Datasets: How Quickly Can
    One Million Curves Be Ranked?" Stat 1 (1): 68–74.
    https://doi.org/10.1002/sta4.8.
    """

    # mask invalid pixels
    if thr is None:
        thr = np.nanmin(X)
    mask = np.logical_and(np.all(np.isfinite(X), axis=0), np.any(X >= thr, axis=0))

    n = X.shape[0]
    p = np.sum(mask)
    depth = np.zeros(n)

    # assign ranks
    b = np.random.random((n, p))
    order = np.lexsort((b, X[:, mask]), axis=0)  # random rank for ties
    rank = order.argsort(axis=0) + 1

    # compute band depth
    nabove = n - rank
    nbelow = rank - 1
    match = nabove * nbelow
    nchoose2 = comb(n, 2)
    proportion = np.sum(match, axis=1) / p
    depth = (proportion + n - 1) / nchoose2

    # normalize depth between 0 and 1
    if norm:
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    return depth
