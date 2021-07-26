# -- coding: utf-8 --
"""
pysteps.verification.probscores
===============================

Evaluation and skill scores for probabilistic forecasts.

.. autosummary::
    :toctree: ../generated/

    CRPS
    CRPS_init
    CRPS_accum
    CRPS_compute
    reldiag
    reldiag_init
    reldiag_accum
    reldiag_compute
    ROC_curve
    ROC_curve_init
    ROC_curve_accum
    ROC_curve_compute
"""

import numpy as np


def CRPS(X_f, X_o):
    """Compute the continuous ranked probability score (CRPS).

    Parameters
    ----------
    X_f: array_like
      Array of shape (k,m,n,...) containing the values from an ensemble
      forecast of k members with shape (m,n,...).
    X_o: array_like
      Array of shape (m,n,...) containing the observed values corresponding
      to the forecast.

    Returns
    -------
    out: float
      The computed CRPS.

    References
    ----------
    :cite:`Her2000`
    """

    X_f = X_f.copy()
    X_o = X_o.copy()
    crps = CRPS_init()
    CRPS_accum(crps, X_f, X_o)
    return CRPS_compute(crps)


def CRPS_init():
    """Initialize a CRPS object.

    Returns
    -------
    out: dict
      The CRPS object.
    """
    return {"CRPS_sum": 0.0, "n": 0.0}


def CRPS_accum(CRPS, X_f, X_o):
    """Compute the average continuous ranked probability score (CRPS) for a set
    of forecast ensembles and the corresponding observations and accumulate the
    result to the given CRPS object.

    Parameters
    ----------
    CRPS: dict
      The CRPS object.
    X_f: array_like
      Array of shape (k,m,n,...) containing the values from an ensemble
      forecast of k members with shape (m,n,...).
    X_o: array_like
      Array of shape (m,n,...) containing the observed values corresponding
      to the forecast.

    References
    ----------
    :cite:`Her2000`
    """
    X_f = np.vstack([X_f[i, :].flatten() for i in range(X_f.shape[0])]).T
    X_o = X_o.flatten()

    mask = np.logical_and(np.all(np.isfinite(X_f), axis=1), np.isfinite(X_o))

    X_f = X_f[mask, :].copy()
    X_f.sort(axis=1)
    X_o = X_o[mask]

    n = X_f.shape[0]
    m = X_f.shape[1]

    alpha = np.zeros((n, m + 1))
    beta = np.zeros((n, m + 1))

    for i in range(1, m):
        mask = X_o > X_f[:, i]
        alpha[mask, i] = X_f[mask, i] - X_f[mask, i - 1]
        beta[mask, i] = 0.0

        mask = np.logical_and(X_f[:, i] > X_o, X_o > X_f[:, i - 1])
        alpha[mask, i] = X_o[mask] - X_f[mask, i - 1]
        beta[mask, i] = X_f[mask, i] - X_o[mask]

        mask = X_o < X_f[:, i - 1]
        alpha[mask, i] = 0.0
        beta[mask, i] = X_f[mask, i] - X_f[mask, i - 1]

    mask = X_o < X_f[:, 0]
    alpha[mask, 0] = 0.0
    beta[mask, 0] = X_f[mask, 0] - X_o[mask]

    mask = X_f[:, -1] < X_o
    alpha[mask, -1] = X_o[mask] - X_f[mask, -1]
    beta[mask, -1] = 0.0

    p = 1.0 * np.arange(m + 1) / m
    res = np.sum(alpha * p ** 2.0 + beta * (1.0 - p) ** 2.0, axis=1)

    CRPS["CRPS_sum"] += np.sum(res)
    CRPS["n"] += len(res)


def CRPS_compute(CRPS):
    """Compute the averaged values from the given CRPS object.

    Parameters
    ----------
    CRPS: dict
      A CRPS object created with CRPS_init.

    Returns
    -------
    out: float
      The computed CRPS.
    """
    return 1.0 * CRPS["CRPS_sum"] / CRPS["n"]


def reldiag(P_f, X_o, X_min, n_bins=10, min_count=10):
    """Compute the x- and y- coordinates of the points in the reliability diagram.

    Parameters
    ----------
    P_f: array-like
      Forecast probabilities for exceeding the intensity threshold specified
      in the reliability diagram object.
    X_o: array-like
      Observed values.
    X_min: float
      Precipitation intensity threshold for yes/no prediction.
    n_bins: int
        Number of bins to use in the reliability diagram.
    min_count: int
      Minimum number of samples required for each bin. A zero value is assigned
      if the number of samples in a bin is smaller than bin_count.

    Returns
    -------
    out: tuple
      Two-element tuple containing the x- and y-coordinates of the points in
      the reliability diagram.
    """

    P_f = P_f.copy()
    X_o = X_o.copy()
    rdiag = reldiag_init(X_min, n_bins, min_count)
    reldiag_accum(rdiag, P_f, X_o)
    return reldiag_compute(rdiag)


def reldiag_init(X_min, n_bins=10, min_count=10):
    """Initialize a reliability diagram object.

    Parameters
    ----------
    X_min: float
      Precipitation intensity threshold for yes/no prediction.
    n_bins: int
        Number of bins to use in the reliability diagram.
    min_count: int
      Minimum number of samples required for each bin. A zero value is assigned
      if the number of samples in a bin is smaller than bin_count.

    Returns
    -------
    out: dict
      The reliability diagram object.

    References
    ----------
    :cite:`BS2007`
    """
    reldiag = {}

    reldiag["X_min"] = X_min
    reldiag["bin_edges"] = np.linspace(-1e-6, 1 + 1e-6, int(n_bins + 1))
    reldiag["n_bins"] = n_bins
    reldiag["X_sum"] = np.zeros(n_bins)
    reldiag["Y_sum"] = np.zeros(n_bins, dtype=int)
    reldiag["num_idx"] = np.zeros(n_bins, dtype=int)
    reldiag["sample_size"] = np.zeros(n_bins, dtype=int)
    reldiag["min_count"] = min_count

    return reldiag


def reldiag_accum(reldiag, P_f, X_o):
    """Accumulate the given probability-observation pairs into the reliability
    diagram.

    Parameters
    ----------
    reldiag: dict
      A reliability diagram object created with reldiag_init.
    P_f: array-like
      Forecast probabilities for exceeding the intensity threshold specified
      in the reliability diagram object.
    X_o: array-like
      Observed values.
    """
    mask = np.logical_and(np.isfinite(P_f), np.isfinite(X_o))

    P_f = P_f[mask]
    X_o = X_o[mask]

    idx = np.digitize(P_f, reldiag["bin_edges"], right=True)

    x = []
    y = []
    num_idx = []
    ss = []

    for k in range(1, len(reldiag["bin_edges"])):
        I_k = np.where(idx == k)[0]
        if len(I_k) >= reldiag["min_count"]:
            X_o_above_thr = (X_o[I_k] >= reldiag["X_min"]).astype(int)
            y.append(np.sum(X_o_above_thr))
            x.append(np.sum(P_f[I_k]))
            num_idx.append(len(I_k))
            ss.append(len(I_k))
        else:
            y.append(0.0)
            x.append(0.0)
            num_idx.append(0.0)
            ss.append(0)

    reldiag["X_sum"] += np.array(x)
    reldiag["Y_sum"] += np.array(y, dtype=int)
    reldiag["num_idx"] += np.array(num_idx, dtype=int)
    reldiag["sample_size"] += ss


def reldiag_compute(reldiag):
    """Compute the x- and y- coordinates of the points in the reliability diagram.

    Parameters
    ----------
    reldiag: dict
      A reliability diagram object created with reldiag_init.

    Returns
    -------
    out: tuple
      Two-element tuple containing the x- and y-coordinates of the points in
      the reliability diagram.
    """
    f = 1.0 * reldiag["Y_sum"] / reldiag["num_idx"]
    r = 1.0 * reldiag["X_sum"] / reldiag["num_idx"]

    return r, f


def ROC_curve(P_f, X_o, X_min, n_prob_thrs=10, compute_area=False):
    """Compute the ROC curve and its area from the given ROC object.

    Parameters
    ----------
    P_f: array_like
      Forecasted probabilities for exceeding the threshold specified in the ROC
      object. Non-finite values are ignored.
    X_o: array_like
      Observed values. Non-finite values are ignored.
    X_min: float
      Precipitation intensity threshold for yes/no prediction.
    n_prob_thrs: int
      The number of probability thresholds to use.
      The interval [0,1] is divided into n_prob_thrs evenly spaced values.
    compute_area: bool
      If True, compute the area under the ROC curve (between 0.5 and 1).

    Returns
    -------
    out: tuple
      A two-element tuple containing the probability of detection (POD) and
      probability of false detection (POFD) for the probability thresholds
      specified in the ROC curve object. If compute_area is True, return the
      area under the ROC curve as the third element of the tuple.
    """

    P_f = P_f.copy()
    X_o = X_o.copy()
    roc = ROC_curve_init(X_min, n_prob_thrs)
    ROC_curve_accum(roc, P_f, X_o)
    return ROC_curve_compute(roc, compute_area)


def ROC_curve_init(X_min, n_prob_thrs=10):
    """Initialize a ROC curve object.

    Parameters
    ----------
    X_min: float
      Precipitation intensity threshold for yes/no prediction.
    n_prob_thrs: int
      The number of probability thresholds to use.
      The interval [0,1] is divided into n_prob_thrs evenly spaced values.

    Returns
    -------
    out: dict
      The ROC curve object.
    """
    ROC = {}

    ROC["X_min"] = X_min
    ROC["hits"] = np.zeros(n_prob_thrs, dtype=int)
    ROC["misses"] = np.zeros(n_prob_thrs, dtype=int)
    ROC["false_alarms"] = np.zeros(n_prob_thrs, dtype=int)
    ROC["corr_neg"] = np.zeros(n_prob_thrs, dtype=int)
    ROC["prob_thrs"] = np.linspace(0.0, 1.0, int(n_prob_thrs))

    return ROC


def ROC_curve_accum(ROC, P_f, X_o):
    """Accumulate the given probability-observation pairs into the given ROC
    object.

    Parameters
    ----------
    ROC: dict
      A ROC curve object created with ROC_curve_init.
    P_f: array_like
      Forecasted probabilities for exceeding the threshold specified in the ROC
      object. Non-finite values are ignored.
    X_o: array_like
      Observed values. Non-finite values are ignored.
    """
    mask = np.logical_and(np.isfinite(P_f), np.isfinite(X_o))

    P_f = P_f[mask]
    X_o = X_o[mask]

    for i, p in enumerate(ROC["prob_thrs"]):
        mask = np.logical_and(P_f >= p, X_o >= ROC["X_min"])
        ROC["hits"][i] += np.sum(mask.astype(int))
        mask = np.logical_and(P_f < p, X_o >= ROC["X_min"])
        ROC["misses"][i] += np.sum(mask.astype(int))
        mask = np.logical_and(P_f >= p, X_o < ROC["X_min"])
        ROC["false_alarms"][i] += np.sum(mask.astype(int))
        mask = np.logical_and(P_f < p, X_o < ROC["X_min"])
        ROC["corr_neg"][i] += np.sum(mask.astype(int))


def ROC_curve_compute(ROC, compute_area=False):
    """Compute the ROC curve and its area from the given ROC object.

    Parameters
    ----------
    ROC: dict
      A ROC curve object created with ROC_curve_init.
    compute_area: bool
      If True, compute the area under the ROC curve (between 0.5 and 1).

    Returns
    -------
    out: tuple
      A two-element tuple containing the probability of detection (POD) and
      probability of false detection (POFD) for the probability thresholds
      specified in the ROC curve object. If compute_area is True, return the
      area under the ROC curve as the third element of the tuple.
    """
    POD_vals = []
    POFD_vals = []

    for i in range(len(ROC["prob_thrs"])):
        POD_vals.append(1.0 * ROC["hits"][i] / (ROC["hits"][i] + ROC["misses"][i]))
        POFD_vals.append(
            1.0 * ROC["false_alarms"][i] / (ROC["corr_neg"][i] + ROC["false_alarms"][i])
        )

    if compute_area:
        # Compute the total area of parallelepipeds under the ROC curve.
        area = (1.0 - POFD_vals[0]) * (1.0 + POD_vals[0]) / 2.0
        for i in range(len(ROC["prob_thrs"]) - 1):
            area += (
                (POFD_vals[i] - POFD_vals[i + 1])
                * (POD_vals[i + 1] + POD_vals[i])
                / 2.0
            )
        area += POFD_vals[-1] * POD_vals[-1] / 2.0

        return POFD_vals, POD_vals, area
    else:
        return POFD_vals, POD_vals
