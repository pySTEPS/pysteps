"""
pysteps.verification.detcontscores
==================================

Forecast evaluation and skill scores for deterministic continuous forecasts.

.. autosummary::
    :toctree: ../generated/

    det_cont_fcst
    det_cont_fcst_init
    det_cont_fcst_accum
    det_cont_fcst_compute
"""

import collections
import numpy as np
from scipy.stats import spearmanr


def det_cont_fcst(pred, obs, scores, axis=None, conditioning=None):
    """Calculate simple and skill scores for deterministic continuous forecasts.

    Parameters
    ----------
    pred : array_like
        Array of predictions. NaNs are ignored.
    obs : array_like
        Array of verifying observations. NaNs are ignored.
    scores : string or list of strings
        The name(s) of the scores. The list of possible score names is:

        .. tabularcolumns:: |p{2cm}|L|

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  beta      | linear regression slope (conditional bias)             |
        +------------+--------------------------------------------------------+
        |  corr_p    | pearson's correleation coefficien (linear correlation) |
        +------------+--------------------------------------------------------+
        |  corr_s*   | spearman's correlation coefficient (rank correlation)  |
        +------------+--------------------------------------------------------+
        |  DRMSE     | debiased root mean squared error                       |
        +------------+--------------------------------------------------------+
        |  MAE       | mean absolute error                                    |
        +------------+--------------------------------------------------------+
        |  ME        | mean error or bias                                     |
        +------------+--------------------------------------------------------+
        |  MSE        | mean squared error                                    |
        +------------+--------------------------------------------------------+
        |  RMSE      | root mean squared error                                |
        +------------+--------------------------------------------------------+
        |  RV        | reduction of variance                                  |
        |            | (Brier Score, Nash-Sutcliffe Efficiency)               |
        +------------+--------------------------------------------------------+

    axis : None or int or tuple of ints, optional
        Axis or axes along which a score is integrated. The default, axis=None,
        will integrate all of the elements of the input arrays.\n
        If axis is -1 (or any negative integer), the integration is not performed
        and scores are computed on all of the elements in the input arrays.\n
        If axis is a tuple of ints, the integration is performed on all of the
        axes specified in the tuple.

    conditioning : {None, 'single', 'double'}, optional
        The type of conditioning on zeros used for the verification.
        The default, conditioning=None, includes zero pairs. With
        conditioning='single', only pairs with either pred or obs > 0 are
        included. With conditioning='double', only pairs with both pred and
        obs > 0 are included.

    Returns
    -------
    result : list
        list containing the verification results

    Note
    ----
    Score names denoted by * can only be computed offline.

    Multiplicative scores can be computed by passing log-tranformed values.

    See also
    --------
    pysteps.verification.detcatscores

    """

    # catch case of single score passed as string
    def get_iterable(x):
        if isinstance(x, collections.Iterable) and not isinstance(x, str):
            return x
        else:
            return (x,)
    scores = get_iterable(scores)

    # split between online and offline scores
    loffline = ["scatter", "corr_s"]
    onscores = [score for score in scores if str(score).lower() not in loffline]
    offscores = [score for score in scores if str(score).lower() in loffline]

    # online scores
    onresult = []
    if any(onscores):

        err = det_cont_fcst_init(axis=axis, conditioning=conditioning)
        det_cont_fcst_accum(err, pred, obs)
        onresult = det_cont_fcst_compute(err, onscores)

    # offline scores
    offresult = []
    if any(offscores):

        pred = np.asarray(pred.copy())
        obs = np.asarray(obs.copy())

        if pred.shape != obs.shape:
            raise ValueError(
                "the shape of pred does not match the shape of obs %s!=%s"
                 % (pred.shape, obs.shape))

        # conditioning
        if conditioning is not None:
            if conditioning == "single":
                idx = np.logical_or(obs > 0, pred > 0)
            elif conditioning == "double":
                idx = np.logical_and(obs > 0, pred > 0)
            else:
                raise ValueError("unkown conditioning %s" % err["conditioning"])
            obs[~idx] = np.nan
            pred[~idx] = np.nan

        for score in offscores:
            # catch None passed as score
            if score is None:
                continue

            score = score.lower()

            # spearman corr (rank correlation)
            if score in ["corr_s", "spearmanr"]:
                corr_s = spearmanr(pred, obs, axis=axis, nan_policy="omit")[0]
                offresult.append(corr_s)

    # pool online and offline results together in the original order
    result = []
    for score in scores:
        if score is None:
            continue
        elif score in onscores:
            result += [b for a, b in zip(onscores, onresult) if a == score]
        elif score in offscores:
            result += [b for a, b in zip(offscores, offresult) if a == score]

    return result


def det_cont_fcst_init(axis=None, conditioning=None):
    """Initialize a verification error object.

    Parameters
    ----------
    axis : None or int or tuple of ints, optional
        Axis or axes along which a score is integrated. The default, axis=None,
        will integrate all of the elements of the input arrays.\n
        If axis is -1 (or any negative integer), the integration is not performed
        and scores are computed on all of the elements in the input arrays.\n
        If axis is a tuple of ints, the integration is performed on all of the
        axes specified in the tuple.

    conditioning : {None, 'single', 'double'}, optional
        The type of conditioning on zeros used for the verification.
        The default, conditioning=None, includes zero pairs. With
        conditioning='single', only pairs with either pred or obs > 0 are
        included. With conditioning='double', only pairs with both pred and
        obs > 0 are included.

    Returns
    -------
    out : dict
        The verification error object.
    """

    err = {}

    # catch case of axis passed as integer
    def get_iterable(x):
        if x is None or \
            (isinstance(x, collections.Iterable) and not isinstance(x, int)):
            return x
        else:
            return (x,)

    err["axis"] = get_iterable(axis)
    err["conditioning"] = conditioning
    err["cov"] = None
    err["vobs"] = None
    err["vpred"] = None
    err["mobs"] = None
    err["mpred"] = None
    err["me"] = None
    err["mse"] = None
    err["mae"] = None
    err["n"] = None

    return err


def det_cont_fcst_accum(err, pred, obs):
    """Accumulate the forecast error in the verification error object.

    Parameters
    ----------
    err : dict
        A verification error object initialized with
        :py:func:`pysteps.verification.detcontscores.det_cont_fcst_init`.
    pred : array_like
        Array of predictions. NaNs are ignored.
    obs : array_like
        Array of verifying observations. NaNs are ignored.

    References
    ----------
    Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1979), "Updating
    Formulae and a Pairwise Algorithm for Computing Sample Variances.",
    Technical Report STAN-CS-79-773, Department of Computer Science,
    Stanford University.

    Schubert, Erich; Gertz, Michael (2018-07-09). "Numerically stable parallel
    computation of (co-)variance". ACM: 10. doi:10.1145/3221269.3223036.

    """

    pred = np.asarray(pred.copy())
    obs = np.asarray(obs.copy())
    axis = tuple(range(pred.ndim)) if err["axis"] is None else err["axis"]

    # checks
    if pred.shape != obs.shape:
        raise ValueError("the shape of pred does not match the shape of obs %s!=%s"
                         % (pred.shape, obs.shape))

    if pred.ndim <= np.max(axis):
        raise ValueError("axis %d is out of bounds for array of dimension %d"
                         % (np.max(axis), len(pred.shape)))

    idims = [dim not in axis for dim in range(pred.ndim)]
    nshape = tuple(np.array(pred.shape)[np.array(idims)])
    if err["cov"] is None:
        # initialize the error arrays in the verification object
        err["cov"] = np.zeros(nshape)
        err["vobs"] = np.zeros(nshape)
        err["vpred"] = np.zeros(nshape)
        err["mobs"] = np.zeros(nshape)
        err["mpred"] = np.zeros(nshape)
        err["me"] = np.zeros(nshape)
        err["mse"] = np.zeros(nshape)
        err["mae"] = np.zeros(nshape)
        err["n"] = np.zeros(nshape)

    else:
        # check dimensions
        if err["cov"].shape != nshape:
            raise ValueError(
                "the shape of the input arrays does not match the shape of the "
                + "verification object %s!=%s" % (nshape, err["cov"].shape))

    # conditioning
    if err["conditioning"] is not None:
        if err["conditioning"] == "single":
            idx = np.logical_or(obs > 0, pred > 0)
        elif err["conditioning"] == "double":
            idx = np.logical_and(obs > 0, pred > 0)
        else:
            raise ValueError("unkown conditioning %s" % err["conditioning"])
        obs[~idx] = np.nan
        pred[~idx] = np.nan

    # add dummy axis in case integration is not required
    if np.max(axis) < 0:
        pred = pred[None, :]
        obs = obs[None, :]
        axis = (0,)
    axis = tuple([a for a in axis if a >= 0])

    # compute residuals
    res = pred - obs
    n = np.sum(np.isfinite(res), axis=axis)

    # new means
    mobs = np.nanmean(obs, axis=axis)
    mpred = np.nanmean(pred, axis=axis)
    me = np.nanmean(res, axis=axis)
    mse = np.nanmean(res**2, axis=axis)
    mae = np.nanmean(np.abs(res), axis=axis)

    # new cov matrix
    cov = np.nanmean((obs - mobs)*(pred - mpred), axis=axis)
    vobs = np.nanmean(np.abs(obs - mobs)**2, axis=axis)
    vpred = np.nanmean(np.abs(pred - mpred)**2, axis=axis)

    # update variances
    err["vobs"] = _parallel_var(
                    err["mobs"], err["n"], err["vobs"],
                    mobs, n, vobs)
    err["vpred"] = _parallel_var(
                    err["mpred"], err["n"], err["vpred"],
                    mpred, n, vpred)

    # update covariance
    err["cov"] = _parallel_cov(
                    err["cov"], err["mobs"], err["mpred"], err["n"],
                    cov, mobs, mpred, n)

    # update means
    err["mobs"] = _parallel_mean(
                    err["mobs"], err["n"],
                    mobs, n)
    err["mpred"] = _parallel_mean(
                    err["mpred"], err["n"],
                    mpred, n)
    err["me"] = _parallel_mean(
                    err["me"], err["n"],
                    me, n)
    err["mse"] = _parallel_mean(
                    err["mse"], err["n"],
                    mse, n)
    err["mae"] = _parallel_mean(
                    err["mae"], err["n"],
                    mae, n)

    # update number of samples
    err["n"] += n


def det_cont_fcst_compute(err, scores):
    """Compute simple and skill scores for deterministic continuous forecasts
    from a verification error object.

    Parameters
    ----------
    err : dict
        A verification error object initialized with
        :py:func:`pysteps.verification.detcontscores.det_cont_fcst_init` and
        populated with
        :py:func:`pysteps.verification.detcontscores.det_cont_fcst_accum`.
    scores : string or list of strings
        The name(s) of the scores. The list of possible score names is:

        .. tabularcolumns:: |p{2cm}|L|

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  beta      | linear regression slope (conditional bias)             |
        +------------+--------------------------------------------------------+
        |  corr_p    | pearson's correleation coefficien (linear correlation) |
        +------------+--------------------------------------------------------+
        |  DRMSE     | debiased root mean squared error                       |
        +------------+--------------------------------------------------------+
        |  MAE       | mean absolute error                                    |
        +------------+--------------------------------------------------------+
        |  ME        | mean error or bias                                     |
        +------------+--------------------------------------------------------+
        |  MSE        | mean squared error                                    |
        +------------+--------------------------------------------------------+
        |  RMSE      | root mean squared error                                |
        +------------+--------------------------------------------------------+
        |  RV        | reduction of variance                                  |
        |            | (Brier Score, Nash-Sutcliffe Efficiency)               |
        +------------+--------------------------------------------------------+

    Returns
    -------
    result : list
        list containing the verification results
    """

    # catch case of single score passed as string
    def get_iterable(x):
        if isinstance(x, collections.Iterable) and not isinstance(x, str):
            return x
        else:
            return (x,)
    scores = get_iterable(scores)

    result = []
    for score in scores:
        # catch None passed as score
        if score is None:
            continue

        score = score.lower()

        # bias (mean error, systematic error)
        if score in ["bias", "me"]:
            bias = err["me"]
            result.append(bias)

        # mean absolute error
        if score in ["mae"]:
            MAE = err["mae"]
            result.append(MAE)

        # mean squared error
        if score in ["mse"]:
            MAE = err["mse"]
            result.append(MAE)

        # root mean squared error
        if score == 'rmse':
            RMSE = np.sqrt(err["mse"])
            result.append(RMSE)

        # linear correlation coeff (pearson corr)
        if score in ["corr_p", "pearsonr"]:
            corr_p = err["cov"]/np.sqrt(err["vobs"])/np.sqrt(err["vpred"])
            result.append(corr_p)

        # beta (linear regression slope)
        if score in ["beta"]:
            beta = err["cov"]/err["vpred"]
            result.append(beta)

        # debiased RMSE
        if score in ["drmse"]:
            RMSE_d = (err["mse"] - err["me"]**2)/err["vobs"]
            result.append(RMSE_d)

        # reduction of variance (Brier Score, Nash-Sutcliffe efficiency coefficient,
        # MSE skill score)
        if score in ["rv", "brier_score", "nse"]:
            RV = 1.0 - err["mse"]/err["vobs"]
            result.append(RV)

    return result


def _parallel_mean(avg_a, count_a, avg_b, count_b):
    return (count_a*avg_a + count_b*avg_b)/(count_a + count_b)


def _parallel_var(avg_a, count_a, var_a, avg_b, count_b, var_b):
    # source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    delta = avg_b - avg_a
    m_a = var_a*count_a
    m_b = var_b*count_b
    M2 = m_a + m_b + delta**2*count_a*count_b/(count_a + count_b)
    return M2/(count_a + count_b)


def _parallel_cov(cov_a, avg_xa, avg_ya, count_a,
                  cov_b, avg_xb, avg_yb, count_b):
    deltax = avg_xb - avg_xa
    deltay = avg_yb - avg_ya
    c_a = cov_a*count_a
    c_b = cov_b*count_b
    C2 = c_a + c_b + deltax*deltay*count_a*count_b/(count_a + count_b)
    return C2/(count_a + count_b)