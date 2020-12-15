# -- coding: utf-8 --
"""
pysteps.verification.detcatscores
=================================

Forecast evaluation and skill scores for deterministic categorial (dichotomous)
forecasts.

.. autosummary::
    :toctree: ../generated/

    det_cat_fct
    det_cat_fct_init
    det_cat_fct_accum
    det_cat_fct_merge
    det_cat_fct_compute
"""

import collections
import numpy as np


def det_cat_fct(pred, obs, thr, scores="", axis=None):
    """Calculate simple and skill scores for deterministic categorical
    (dichotomous) forecasts.

    Parameters
    ----------
    pred: array_like
        Array of predictions. NaNs are ignored.
    obs: array_like
        Array of verifying observations. NaNs are ignored.
    thr: float
        The threshold that is applied to predictions and observations in order
        to define events vs no events (yes/no).
    scores: {string, list of strings}, optional
        The name(s) of the scores. The default, scores="", will compute all
        available scores.
        The available score names are:

        .. tabularcolumns:: |p{2cm}|L|

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  ACC       | accuracy (proportion correct)                          |
        +------------+--------------------------------------------------------+
        |  BIAS      | frequency bias                                         |
        +------------+--------------------------------------------------------+
        |  CSI       | critical success index (threat score)                  |
        +------------+--------------------------------------------------------+
        |  ETS       | equitable threat score                                 |
        +------------+--------------------------------------------------------+
        |  F1        | the harmonic mean of precision and sensitivity         |
        +------------+--------------------------------------------------------+
        |  FA        | false alarm rate (prob. of false detection, fall-out,  |
        |            | false positive rate)                                   |
        +------------+--------------------------------------------------------+
        |  FAR       | false alarm ratio (false discovery rate)               |
        +------------+--------------------------------------------------------+
        |  GSS       | Gilbert skill score (equitable threat score)           |
        +------------+--------------------------------------------------------+
        |  HK        | Hanssen-Kuipers discriminant (Pierce skill score)      |
        +------------+--------------------------------------------------------+
        |  HSS       | Heidke skill score                                     |
        +------------+--------------------------------------------------------+
        |  MCC       | Matthews correlation coefficient                       |
        +------------+--------------------------------------------------------+
        |  POD       | probability of detection (hit rate, sensitivity,       |
        |            | recall, true positive rate)                            |
        +------------+--------------------------------------------------------+
        |  SEDI      | symmetric extremal dependency index                    |
        +------------+--------------------------------------------------------+

    axis: None or int or tuple of ints, optional
        Axis or axes along which a score is integrated. The default, axis=None,
        will integrate all of the elements of the input arrays.\n
        If axis is -1 (or any negative integer),
        the integration is not performed
        and scores are computed on all of the elements in the input arrays.\n
        If axis is a tuple of ints, the integration is performed on all of the
        axes specified in the tuple.

    Returns
    -------
    result: dict
        Dictionary containing the verification results.

    See also
    --------
    pysteps.verification.detcontscores.det_cont_fct
    """

    contab = det_cat_fct_init(thr, axis)
    det_cat_fct_accum(contab, pred, obs)
    return det_cat_fct_compute(contab, scores)


def det_cat_fct_init(thr, axis=None):
    """Initialize a contingency table object.

    Parameters
    ----------
    thr: float
        threshold that is applied to predictions and observations in order
        to define events vs no events (yes/no).
    axis: None or int or tuple of ints, optional
        Axis or axes along which a score is integrated. The default, axis=None,
        will integrate all of the elements of the input arrays.\n
        If axis is -1 (or any negative integer),
        the integration is not performed
        and scores are computed on all of the elements in the input arrays.\n
        If axis is a tuple of ints, the integration is performed on all of the
        axes specified in the tuple.

    Returns
    -------
    out: dict
      The contingency table object.
    """

    contab = {}

    # catch case of axis passed as integer
    def get_iterable(x):
        if x is None or (
            isinstance(x, collections.abc.Iterable) and not isinstance(x, int)
        ):
            return x
        else:
            return (x,)

    contab["thr"] = thr
    contab["axis"] = get_iterable(axis)
    contab["hits"] = None
    contab["false_alarms"] = None
    contab["misses"] = None
    contab["correct_negatives"] = None

    return contab


def det_cat_fct_accum(contab, pred, obs):
    """Accumulate the frequency of "yes" and "no" forecasts and observations
    in the contingency table.

    Parameters
    ----------
    contab: dict
      A contingency table object initialized with
      pysteps.verification.detcatscores.det_cat_fct_init.
    pred: array_like
        Array of predictions. NaNs are ignored.
    obs: array_like
        Array of verifying observations. NaNs are ignored.
    """

    pred = np.asarray(pred.copy())
    obs = np.asarray(obs.copy())
    axis = tuple(range(pred.ndim)) if contab["axis"] is None else contab["axis"]

    # checks
    if pred.shape != obs.shape:
        raise ValueError(
            "the shape of pred does not match the shape of obs %s!=%s"
            % (pred.shape, obs.shape)
        )

    if pred.ndim <= np.max(axis):
        raise ValueError(
            "axis %d is out of bounds for array of dimension %d"
            % (np.max(axis), len(pred.shape))
        )

    idims = [dim not in axis for dim in range(pred.ndim)]
    nshape = tuple(np.array(pred.shape)[np.array(idims)])
    if contab["hits"] is None:
        # initialize the count arrays in the contingency table
        contab["hits"] = np.zeros(nshape, dtype=int)
        contab["false_alarms"] = np.zeros(nshape, dtype=int)
        contab["misses"] = np.zeros(nshape, dtype=int)
        contab["correct_negatives"] = np.zeros(nshape, dtype=int)

    else:
        # check dimensions
        if contab["hits"].shape != nshape:
            raise ValueError(
                "the shape of the input arrays does not match "
                + "the shape of the "
                + "contingency table %s!=%s" % (nshape, contab["hits"].shape)
            )

    # add dummy axis in case integration is not required
    if np.max(axis) < 0:
        pred = pred[None, :]
        obs = obs[None, :]
        axis = (0,)
    axis = tuple([a for a in axis if a >= 0])

    # apply threshold
    predb = pred > contab["thr"]
    obsb = obs > contab["thr"]

    # calculate hits, misses, false positives, correct rejects
    H_idx = np.logical_and(predb == 1, obsb == 1)
    F_idx = np.logical_and(predb == 1, obsb == 0)
    M_idx = np.logical_and(predb == 0, obsb == 1)
    R_idx = np.logical_and(predb == 0, obsb == 0)

    # accumulate in the contingency table
    contab["hits"] += np.nansum(H_idx.astype(int), axis=axis)
    contab["misses"] += np.nansum(M_idx.astype(int), axis=axis)
    contab["false_alarms"] += np.nansum(F_idx.astype(int), axis=axis)
    contab["correct_negatives"] += np.nansum(R_idx.astype(int), axis=axis)


def det_cat_fct_merge(contab_1, contab_2):
    """Merge two contingency table objects.

    Parameters
    ----------
    contab_1: dict
      A contingency table object initialized with
      :py:func:`pysteps.verification.detcatscores.det_cat_fct_init`
      and populated with
      :py:func:`pysteps.verification.detcatscores.det_cat_fct_accum`.
    contab_2: dict
      Another contingency table object initialized with
      :py:func:`pysteps.verification.detcatscores.det_cat_fct_init`
      and populated with
      :py:func:`pysteps.verification.detcatscores.det_cat_fct_accum`.

    Returns
    -------
    out: dict
      The merged contingency table object.
    """

    # checks
    if contab_1["thr"] != contab_2["thr"]:
        raise ValueError(
            "cannot merge: the thresholds are not same %s!=%s"
            % (contab_1["thr"], contab_2["thr"])
        )
    if contab_1["axis"] != contab_2["axis"]:
        raise ValueError(
            "cannot merge: the axis are not same %s!=%s"
            % (contab_1["axis"], contab_2["axis"])
        )
    if contab_1["hits"] is None or contab_2["hits"] is None:
        raise ValueError("cannot merge: no data found")

    # merge the contingency tables
    contab = contab_1.copy()
    contab["hits"] += contab_2["hits"]
    contab["misses"] += contab_2["misses"]
    contab["false_alarms"] += contab_2["false_alarms"]
    contab["correct_negatives"] += contab_2["correct_negatives"]

    return contab


def det_cat_fct_compute(contab, scores=""):
    """Compute simple and skill scores for deterministic categorical
    (dichotomous) forecasts from a contingency table object.

    Parameters
    ----------
    contab: dict
      A contingency table object initialized with
      pysteps.verification.detcatscores.det_cat_fct_init and populated with
      pysteps.verification.detcatscores.det_cat_fct_accum.
    scores: {string, list of strings}, optional
        The name(s) of the scores. The default, scores="", will compute all
        available scores.
        The available score names a

        .. tabularcolumns:: |p{2cm}|L|

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  ACC       | accuracy (proportion correct)                          |
        +------------+--------------------------------------------------------+
        |  BIAS      | frequency bias                                         |
        +------------+--------------------------------------------------------+
        |  CSI       | critical success index (threat score)                  |
        +------------+--------------------------------------------------------+
        |  ETS       | equitable threat score                                 |
        +------------+--------------------------------------------------------+
        |  F1        | the harmonic mean of precision and sensitivity         |
        +------------+--------------------------------------------------------+
        |  FA        | false alarm rate (prob. of false detection, fall-out,  |
        |            | false positive rate)                                   |
        +------------+--------------------------------------------------------+
        |  FAR       | false alarm ratio (false discovery rate)               |
        +------------+--------------------------------------------------------+
        |  GSS       | Gilbert skill score (equitable threat score)           |
        +------------+--------------------------------------------------------+
        |  HK        | Hanssen-Kuipers discriminant (Pierce skill score)      |
        +------------+--------------------------------------------------------+
        |  HSS       | Heidke skill score                                     |
        +------------+--------------------------------------------------------+
        |  MCC       | Matthews correlation coefficient                       |
        +------------+--------------------------------------------------------+
        |  POD       | probability of detection (hit rate, sensitivity,       |
        |            | recall, true positive rate)                            |
        +------------+--------------------------------------------------------+
        |  SEDI      | symmetric extremal dependency index                    |
        +------------+--------------------------------------------------------+

    Returns
    -------
    result: dict
        Dictionary containing the verification results.
    """

    # catch case of single score passed as string
    def get_iterable(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        else:
            return (x,)

    scores = get_iterable(scores)

    H = 1.0 * contab["hits"]  # true positives
    M = 1.0 * contab["misses"]  # false negatives
    F = 1.0 * contab["false_alarms"]  # false positives
    R = 1.0 * contab["correct_negatives"]  # true negatives

    result = {}
    for score in scores:
        # catch None passed as score
        if score is None:
            continue

        score_ = score.lower()

        # simple scores
        POD = H / (H + M)
        FAR = F / (H + F)
        FA = F / (F + R)
        s = (H + M) / (H + M + F + R)

        if score_ in ["pod", ""]:
            # probability of detection
            result["POD"] = POD
        if score_ in ["far", ""]:
            # false alarm ratio
            result["FAR"] = FAR
        if score_ in ["fa", ""]:
            # false alarm rate (prob of false detection)
            result["FA"] = FA
        if score_ in ["acc", ""]:
            # accuracy (fraction correct)
            ACC = (H + R) / (H + M + F + R)
            result["ACC"] = ACC
        if score_ in ["csi", ""]:
            # critical success index
            CSI = H / (H + M + F)
            result["CSI"] = CSI
        if score_ in ["bias", ""]:
            # frequency bias
            B = (H + F) / (H + M)
            result["BIAS"] = B

        # skill scores
        if score_ in ["hss", ""]:
            # Heidke Skill Score (-1 < HSS < 1) < 0 implies no skill
            HSS = 2 * (H * R - F * M) / ((H + M) * (M + R) + (H + F) * (F + R))
            result["HSS"] = HSS
        if score_ in ["hk", ""]:
            # Hanssen-Kuipers Discriminant
            HK = POD - FA
            result["HK"] = HK
        if score_ in ["gss", "ets", ""]:
            # Gilbert Skill Score
            GSS = (POD - FA) / ((1 - s * POD) / (1 - s) + FA * (1 - s) / s)
            if score_ == "ets":
                result["ETS"] = GSS
            else:
                result["GSS"] = GSS
        if score_ in ["sedi", ""]:
            # Symmetric extremal dependence index
            SEDI = (np.log(FA) - np.log(POD) + np.log(1 - POD) - np.log(1 - FA)) / (
                np.log(FA) + np.log(POD) + np.log(1 - POD) + np.log(1 - FA)
            )
            result["SEDI"] = SEDI
        if score_ in ["mcc", ""]:
            # Matthews correlation coefficient
            MCC = (H * R - F * M) / np.sqrt((H + F) * (H + M) * (R + F) * (R + M))
            result["MCC"] = MCC
        if score_ in ["f1", ""]:
            # F1 score
            F1 = 2 * H / (2 * H + F + M)
            result["F1"] = F1

    return result
