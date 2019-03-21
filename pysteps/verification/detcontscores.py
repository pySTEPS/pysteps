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


def det_cont_fcst(pred, obs, scores):
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
        |  MAE       | mean absolute error of additive residuals              |
        +------------+--------------------------------------------------------+
        |  ME        | mean error or bias of additive residuals               |
        +------------+--------------------------------------------------------+
        |  RMSE      | root mean squared additive error                       |
        +------------+--------------------------------------------------------+
        |  RV        | reduction of variance                                  |
        |            | (Brier Score, Nash-Sutcliffe Efficiency)               |
        +------------+--------------------------------------------------------+
        |  scatter*  | half the distance between the 16% and 84% percentiles  |
        |            | of the weighted error distribution                     |
        +------------+--------------------------------------------------------+

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

    onresult = []
    if any(onscores):

        err = det_cont_fcst_init()
        det_cont_fcst_accum(err, pred, obs)
        onresult = det_cont_fcst_compute(err, onscores)

    offresult = []
    if any(offscores):

        pred = np.asarray(pred.copy())
        obs = np.asarray(obs.copy())

        # checks
        if pred.shape != obs.shape:
            raise ValueError(
                "the shape of pred does not match the shape of obs %s!=%s"
                 % (pred.shape, obs.shape))

        # flatten array if 2D
        pred = pred.flatten()
        obs = obs.flatten()

        isNaN = np.logical_or(np.isnan(pred), np.isnan(obs))
        pred = pred[~isNaN]
        obs = obs[~isNaN]

        for score in offscores:
            # catch None passed as score
            if score is None:
                continue

            score = score.lower()

            # spearman corr (rank correlation)
            if score in ["corr_s", "spearmanr"]:
                corr_s = spearmanr(pred, obs)[0]
                offresult.append(corr_s)

            # scatter
            if score in ["scatter"]:
                q = 10*np.log10(pred/obs)
                xs = np.sort(q)
                ixs = np.argsort(q)
                xs = np.insert(xs, 0, xs[0])
                ws = obs[ixs]
                ws = np.insert(ws, 0, 0)
                wsc = np.cumsum(ws)/np.sum(ws)
                xint = np.interp([0.16, 0.50, 0.84], wsc, xs)
                scatter = (xint[2] - xint[0])/2.
                offresult.append(scatter)

    # pool online and offline results together
    result = []
    for score in scores:
        if score is None:
            continue
        elif score in onscores:
            result += [b for a, b in zip(onscores, onresult) if a == score]
        elif score in offscores:
            result += [b for a, b in zip(offscores, offresult) if a == score]

    return result


def det_cont_fcst_init():
    """Initialize a verification error object.

    Returns
    -------
    out : dict
        The verification error object.
    """

    err = {}

    err["cov"] = 0
    err["vobs"] = 0
    err["vpred"] = 0
    err["mobs"] = 0
    err["mpred"] = 0
    err["me"] = 0
    err["mse"] = 0
    err["mae"] = 0
    err["n"] = 0

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

    """

    pred = np.asarray(pred.copy())
    obs = np.asarray(obs.copy())

    # checks
    if pred.shape != obs.shape:
        raise ValueError(
            "the shape of pred does not match the shape of obs %s!=%s"
             % (pred.shape, obs.shape))

    # flatten array if 2D
    pred = pred.flatten()
    obs = obs.flatten()

    isNaN = np.logical_or(np.isnan(pred), np.isnan(obs))
    pred = pred[~isNaN]
    obs = obs[~isNaN]

    # compute residuals
    res = pred - obs
    n = len(res)

    # new means
    mobs = np.mean(obs)
    mpred = np.mean(pred)
    me = np.mean(res)
    mse = np.mean(res**2)
    mae = np.mean(np.abs(res))

    # new cov matrix
    covm = np.cov(obs, pred, ddof=0)
    cov = covm[0,1]
    vobs = covm[0,0]
    vpred = covm[1,1]

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
        |  MAE       | mean absolute error of additive residuals              |
        +------------+--------------------------------------------------------+
        |  ME        | mean error or bias of additive residuals               |
        +------------+--------------------------------------------------------+
        |  RMSE      | root mean squared additive error                       |
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

        # root mean squared error
        if score == 'rmse':
            RMSE = np.sqrt(err["mse"])
            result.append(RMSE)

        # linear correlation coeff (pearson corr)
        if score in ["corr_p", "pearsonr"]:
            corr_p = err["cov"]/np.sqrt(err["vobs"])/np.sqrt(err["vpred"])
            result.append(corr_p)

        # beta (linear regression slope)
        if score == 'beta':
            beta = err["cov"]/err["vpred"]
            result.append(beta)

        # debiased RMSE
        if score == 'RMSE_d':
            RMSE_d = (err["mse"] - err["me"]**2)/err["vobs"]
            result.append(RMSE_d)

        # reduction of variance (Brier Score, NSE)
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

