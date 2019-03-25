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
    det_cat_fct_compute
"""

import collections
import numpy as np


def det_cat_fct(pred, obs, thr, scores="", axis=None):

    """Calculate simple and skill scores for deterministic categorical
    (dichotomous) forecasts.

    Parameters
    ----------
    pred : array_like
        Array of predictions. NaNs are ignored.

    obs : array_like
        Array of verifying observations. NaNs are ignored.

    thr : float
        The threshold that is applied to predictions and observations in order
        to define events vs no events (yes/no).

    scores : {string, list of strings}, optional
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
        |  FA        | false alarm rate (prob. of false detection)            |
        +------------+--------------------------------------------------------+
        |  FAR       | false alarm ratio                                      |
        +------------+--------------------------------------------------------+
        |  GSS       | Gilbert skill score (equitable threat score)           |
        +------------+--------------------------------------------------------+
        |  HK        | Hanssen-Kuipers discriminant (Pierce skill score)      |
        +------------+--------------------------------------------------------+
        |  HSS       | Heidke skill score                                     |
        +------------+--------------------------------------------------------+
        |  POD       | probability of detection (hit rate)                    |
        +------------+--------------------------------------------------------+
        |  SEDI      | symmetric extremal dependency index                    |
        +------------+--------------------------------------------------------+

    axis : None or int or tuple of ints, optional
        Axis or axes along which a score is integrated. The default, axis=None,
        will integrate all of the elements of the input arrays.\n
        If axis is -1 (or any negative integer), the integration is not performed
        and scores are computed on all of the elements in the input arrays.\n
        If axis is a tuple of ints, the integration is performed on all of the
        axes specified in the tuple.

    Returns
    -------
    result : dict
        Dictionary containing the verification results.

    See also
    --------
    pysteps.verification.detcontscores.det_cont_fct


    """

    contab = det_cat_fct_init(thr)
    det_cat_fct_accum(contab, pred, obs)
    return det_cat_fct_compute(contab, scores)


def det_cat_fct_init(thr, axis=None):
    """Initialize a contingency table object.

    Parameters
    ----------
    thr : float
        threshold that is applied to predictions and observations in order
        to define events vs no events (yes/no).

    axis : None or int or tuple of ints, optional
        Axis or axes along which a score is integrated. The default, axis=None,
        will integrate all of the elements of the input arrays.\n
        If axis is -1 (or any negative integer), the integration is not performed
        and scores are computed on all of the elements in the input arrays.\n
        If axis is a tuple of ints, the integration is performed on all of the
        axes specified in the tuple.

    Returns
    -------
    out : dict
      The contingency table object.

    """

    contab = {}

    # catch case of axis passed as integer
    def get_iterable(x):
        if x is None or \
            (isinstance(x, collections.Iterable) and not isinstance(x, int)):
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
    contab : dict
      A contingency table object initialized with
      pysteps.verification.detcatscores.det_cat_fct_init.

    pred : array_like
        Array of predictions. NaNs are ignored.

    obs : array_like
        Array of verifying observations. NaNs are ignored.

    """

    pred = np.asarray(pred.copy())
    obs = np.asarray(obs.copy())
    axis = tuple(range(pred.ndim)) if contab["axis"] is None else contab["axis"]

    # checks
    if pred.shape != obs.shape:
        raise ValueError("the shape of pred does not match the shape of obs %s!=%s"
                         % (pred.shape, obs.shape))

    if pred.ndim <= np.max(axis):
        raise ValueError("axis %d is out of bounds for array of dimension %d"
                         % (np.max(axis), len(pred.shape)))

    idims = [dim not in axis for dim in range(pred.ndim)]
    nshape = tuple(np.array(pred.shape)[np.array(idims)])
    if contab["hits"] is None:
        # initialize the count arrays in the contingency table
        contab["hits"] = np.zeros(nshape)
        contab["false_alarms"] = np.zeros(nshape)
        contab["misses"] = np.zeros(nshape)
        contab["correct_negatives"] = np.zeros(nshape)

    else:
        # check dimensions
        if contab["hits"].shape != nshape:
            raise ValueError(
                "the shape of the input arrays does not match the shape of the "
                + "contingency table %s!=%s" % (nshape, contab["hits"].shape))

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


def det_cat_fct_compute(contab, scores=""):
    """Compute simple and skill scores for deterministic categorical
    (dichotomous) forecasts from a contingency table object.

    Parameters
    ----------
    contab : dict
      A contingency table object initialized with
      pysteps.verification.detcatscores.det_cat_fct_init and populated with
      pysteps.verification.detcatscores.det_cat_fct_accum.

    scores : {string, list of strings}, optional
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
        |  FA        | false alarm rate (prob. of false detection)            |
        +------------+--------------------------------------------------------+
        |  FAR       | false alarm ratio                                      |
        +------------+--------------------------------------------------------+
        |  GSS       | Gilbert skill score (equitable threat score)           |
        +------------+--------------------------------------------------------+
        |  HK        | Hanssen-Kuipers discriminant (Pierce skill score)      |
        +------------+--------------------------------------------------------+
        |  HSS       | Heidke skill score                                     |
        +------------+--------------------------------------------------------+
        |  POD       | probability of detection (hit rate)                    |
        +------------+--------------------------------------------------------+
        |  SEDI      | symmetric extremal dependency index                    |
        +------------+--------------------------------------------------------+

    Returns
    -------
    result : dict
        Dictionary containing the verification results.

    """

    # catch case of single score passed as string
    def get_iterable(x):
        if isinstance(x, collections.Iterable) and not isinstance(x, str):
            return x
        else:
            return (x,)
    scores = get_iterable(scores)

    H = 1.*contab["hits"]
    M = 1.*contab["misses"]
    F = 1.*contab["false_alarms"]
    R = 1.*contab["correct_negatives"]

    result = {}
    for score in scores:
        # catch None passed as score
        if score is None:
            continue

        score_ = score.lower()

        # simple scores
        POD = H/(H + M)
        FAR = F/(H + F)
        FA = F/(F + R)
        s = (H + M)/(H + M + F + R)

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
            ACC = (H + R)/(H + M + F + R)
            result["ACC"] = ACC
        if score_ in ["csi", ""]:
            # critical success index
            CSI = H/(H + M + F)
            result["CSI"] = CSI
        if score_ in ["bias", ""]:
            # frequency bias
            B = (H + F) / (H + M)
            result["BIAS"] = B

        # skill scores
        if score_ in ["hss", ""]:
            # Heidke Skill Score (-1 < HSS < 1) < 0 implies no skill
            HSS = 2*(H*R - F*M)/((H + M)*(M + R) + (H + F)*(F + R))
            result["HSS"] = HSS
        if score_ in ["hk", ""]:
            # Hanssen-Kuipers Discriminant
            HK = POD - FA
            result["HK"] = HK
        if score_ in ["gss", ""]:
            # Gilbert Skill Score
            GSS = (POD - FA)/((1 - s*POD)/(1 - s) + FA*(1 - s)/s)
            result["GSS"] = GSS
        if score_ in ["ets", ""]:
            # Equitable Threat Score
            N = H + M + R + F
            HR = ((H + M)*(H + F)) / N
            if (H + M + F - HR) == 0:
                ETS = np.nan
            else:
                ETS = (H - HR) / (H + M + F - HR)
            result["ETS"] = ETS
        if score_ in ["sedi", ""]:
            # Symmetric extremal dependence index
            SEDI = (np.log(FA) - np.log(POD) + np.log(1 - POD) - np.log(1 - FA))\
                  /(np.log(FA) + np.log(POD) + np.log(1 - POD) + np.log(1 - FA))
            result["SEDI"] = SEDI

    return result
