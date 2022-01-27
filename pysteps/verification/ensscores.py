# -- coding: utf-8 --
"""
pysteps.verification.ensscores
==============================

Evaluation and skill scores for ensemble forecasts.

.. autosummary::
    :toctree: ../generated/

    ensemble_skill
    ensemble_spread
    rankhist
    rankhist_init
    rankhist_accum
    rankhist_compute
"""

import numpy as np
from .interface import get_method


def ensemble_skill(X_f, X_o, metric, **kwargs):
    """
    Compute mean ensemble skill for a given skill metric.

    Parameters
    ----------
    X_f: array-like
        Array of shape (l,m,n) containing the forecast fields of shape (m,n)
        from l ensemble members.
    X_o: array_like
        Array of shape (m,n) containing the observed field corresponding to
        the forecast.
    metric: str
        The deterministic skill metric to be used (list available in
        :func:`~pysteps.verification.interface.get_method`).

    Returns
    -------
    out: float
        The mean skill of all ensemble members that is used as defintion of
        ensemble skill (as in Zacharov and Rezcova 2009 with the FSS).

    References
    ----------
    :cite:`ZR2009`
    """

    if len(X_f.shape) != 3:
        raise ValueError(
            "the number of dimensions of X_f must be equal to 3, "
            + "but %i dimensions were passed" % len(X_f.shape)
        )
    if X_f.shape[1:] != X_o.shape:
        raise ValueError(
            "the shape of X_f does not match the shape of "
            + "X_o (%d,%d)!=(%d,%d)"
            % (X_f.shape[1], X_f.shape[2], X_o.shape[0], X_o.shape[1])
        )

    compute_skill = get_method(metric, type="deterministic")

    lolo = X_f.shape[0]
    skill = []
    for member in range(lolo):
        skill_ = compute_skill(X_f[member, :, :], X_o, **kwargs)
        if isinstance(skill_, dict):
            skill_ = skill_[metric]
        skill.append(skill_)

    return np.mean(skill)


def ensemble_spread(X_f, metric, **kwargs):
    """
    Compute mean ensemble spread for a given skill metric.

    Parameters
    ----------
    X_f: array-like
        Array of shape (l,m,n) containing the forecast fields of shape (m,n)
        from l ensemble members.
    metric: str
        The deterministic skill metric to be used (list available in
        :func:`~pysteps.verification.interface.get_method`).

    Returns
    -------
    out: float
        The mean skill compted between all possible pairs of
        the ensemble members,
        which can be used as definition of mean ensemble spread (as in Zacharov
        and Rezcova 2009 with the FSS).

    References
    ----------
    :cite:`ZR2009`
    """
    if len(X_f.shape) != 3:
        raise ValueError(
            "the number of dimensions of X_f must be equal to 3, "
            + "but %i dimensions were passed" % len(X_f.shape)
        )
    if X_f.shape[0] < 2:
        raise ValueError(
            "the number of members in X_f must be greater than 1,"
            + " but %i members were passed" % X_f.shape[0]
        )

    compute_spread = get_method(metric, type="deterministic")

    lolo = X_f.shape[0]
    spread = []
    for member in range(lolo):
        for othermember in range(member + 1, lolo):
            spread_ = compute_spread(
                X_f[member, :, :], X_f[othermember, :, :], **kwargs
            )
            if isinstance(spread_, dict):
                spread_ = spread_[metric]
            spread.append(spread_)

    return np.mean(spread)


def rankhist(X_f, X_o, X_min=None, normalize=True):
    """
    Compute a rank histogram counts and optionally normalize the histogram.

    Parameters
    ----------
    X_f: array-like
        Array of shape (k,m,n,...) containing the values from an ensemble
        forecast of k members with shape (m,n,...).
    X_o: array_like
        Array of shape (m,n,...) containing the observed values corresponding
        to the forecast.
    X_min: {float,None}
        Threshold for minimum intensity. Forecast-observation pairs, where all
        ensemble members and verifying observations are below X_min, are not
        counted in the rank histogram.
        If set to None, thresholding is not used.
    normalize: {bool, True}
        If True, normalize the rank histogram so that
        the bin counts sum to one.
    """

    X_f = X_f.copy()
    X_o = X_o.copy()
    num_ens_members = X_f.shape[0]
    rhist = rankhist_init(num_ens_members, X_min)
    rankhist_accum(rhist, X_f, X_o)
    return rankhist_compute(rhist, normalize)


def rankhist_init(num_ens_members, X_min=None):
    """
    Initialize a rank histogram object.

    Parameters
    ----------
    num_ens_members: int
        Number ensemble members in the forecasts to accumulate into the rank
        histogram.
    X_min: {float,None}
        Threshold for minimum intensity. Forecast-observation pairs, where all
        ensemble members and verifying observations are below X_min, are not
        counted in the rank histogram.
        If set to None, thresholding is not used.

    Returns
    -------
    out: dict
        The rank histogram object.
    """
    rankhist = {}

    rankhist["num_ens_members"] = num_ens_members
    rankhist["n"] = np.zeros(num_ens_members + 1, dtype=int)
    rankhist["X_min"] = X_min

    return rankhist


def rankhist_accum(rankhist, X_f, X_o):
    """Accumulate forecast-observation pairs to the given rank histogram.

    Parameters
    ----------
    rankhist: dict
      The rank histogram object.
    X_f: array-like
        Array of shape (k,m,n,...) containing the values from an ensemble
        forecast of k members with shape (m,n,...).
    X_o: array_like
        Array of shape (m,n,...) containing the observed values corresponding
        to the forecast.
    """
    if X_f.shape[0] != rankhist["num_ens_members"]:
        raise ValueError(
            "the number of ensemble members in X_f does not "
            + "match the number of members in the rank "
            + "histogram (%d!=%d)" % (X_f.shape[0], rankhist["num_ens_members"])
        )

    X_f = np.vstack([X_f[i, :].flatten() for i in range(X_f.shape[0])]).T
    X_o = X_o.flatten()

    X_min = rankhist["X_min"]

    mask = np.logical_and(np.isfinite(X_o), np.all(np.isfinite(X_f), axis=1))
    # ignore pairs where the verifying observations and all ensemble members
    # are below the threshold X_min
    if X_min is not None:
        mask_nz = np.logical_or(X_o >= X_min, np.any(X_f >= X_min, axis=1))
        mask = np.logical_and(mask, mask_nz)

    X_f = X_f[mask, :].copy()
    X_o = X_o[mask].copy()
    if X_min is not None:
        X_f[X_f < X_min] = X_min - 1
        X_o[X_o < X_min] = X_min - 1

    X_o = np.reshape(X_o, (len(X_o), 1))

    X_c = np.hstack([X_f, X_o])
    X_c.sort(axis=1)

    idx1 = np.where(X_c == X_o)
    _, idx2, idx_counts = np.unique(idx1[0], return_index=True, return_counts=True)
    bin_idx_1 = idx1[1][idx2]

    bin_idx = list(bin_idx_1[np.where(idx_counts == 1)[0]])

    # handle ties, where the verifying observation lies between ensemble
    # members having the same value
    idxdup = np.where(idx_counts > 1)[0]
    if len(idxdup) > 0:
        X_c_ = np.fliplr(X_c)
        idx1 = np.where(X_c_ == X_o)
        _, idx2 = np.unique(idx1[0], return_index=True)
        bin_idx_2 = X_f.shape[1] - idx1[1][idx2]

        idxr = np.random.uniform(low=0.0, high=1.0, size=len(idxdup))
        idxr = bin_idx_1[idxdup] + idxr * (bin_idx_2[idxdup] + 1 - bin_idx_1[idxdup])
        bin_idx.extend(idxr.astype(int))

    for bi in bin_idx:
        rankhist["n"][bi] += 1


def rankhist_compute(rankhist, normalize=True):
    """
    Return the rank histogram counts and optionally normalize the histogram.

    Parameters
    ----------
    rankhist: dict
        A rank histogram object created with rankhist_init.
    normalize: bool
        If True, normalize the rank histogram so that
        the bin counts sum to one.

    Returns
    -------
    out: array_like
        The counts for the n+1 bins in the rank histogram,
        where n is the number of ensemble members.
    """
    if normalize:
        return 1.0 * rankhist["n"] / sum(rankhist["n"])
    else:
        return rankhist["n"]
