# -- coding: utf-8 --
"""
pysteps.verification.detcontscores
==================================

Forecast evaluation and skill scores for deterministic continuous forecasts.

.. autosummary::
    :toctree: ../generated/

    det_cont_fct
    det_cont_fct_init
    det_cont_fct_accum
    det_cont_fct_merge
    det_cont_fct_compute
"""

import collections
import numpy as np
from scipy.stats import spearmanr


def det_cont_fct(pred, obs, scores="", axis=None, conditioning=None, thr=0.0):
    """Calculate simple and skill scores for deterministic continuous forecasts.

    Parameters
    ----------
    pred: array_like
        Array of predictions. NaNs are ignored.
    obs: array_like
        Array of verifying observations. NaNs are ignored.
    scores: {string, list of strings}, optional
        The name(s) of the scores. The default, scores="", will compute all
        available scores.
        The available score names are:

        .. tabularcolumns:: |p{2cm}|L|

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  beta1     | linear regression slope (type 1 conditional bias)      |
        +------------+--------------------------------------------------------+
        |  beta2     | linear regression slope (type 2 conditional bias)      |
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
        |  MSE       | mean squared error                                     |
        +------------+--------------------------------------------------------+
        |  NMSE      | normalized mean squared error                          |
        +------------+--------------------------------------------------------+
        |  RMSE      | root mean squared error                                |
        +------------+--------------------------------------------------------+
        |  RV        | reduction of variance                                  |
        |            | (Brier Score, Nash-Sutcliffe Efficiency)               |
        +------------+--------------------------------------------------------+
        |  scatter*  | half the distance between the 16% and 84% percentiles  |
        |            | of the weighted cumulative error distribution,         |
        |            | where error = dB(pred/obs),                            |
        |            | as in Germann et al. (2006)                            |
        +------------+--------------------------------------------------------+

    axis: {int, tuple of int, None}, optional
        Axis or axes along which a score is integrated. The default, axis=None,
        will integrate all of the elements of the input arrays.\n
        If axis is -1 (or any negative integer),
        the integration is not performed
        and scores are computed on all of the elements in the input arrays.\n
        If axis is a tuple of ints, the integration is performed on all of the
        axes specified in the tuple.
    conditioning: {None, "single", "double"}, optional
        The type of conditioning used for the verification.
        The default, conditioning=None, includes all pairs. With
        conditioning="single", only pairs with either pred or obs > thr are
        included. With conditioning="double", only pairs with both pred and
        obs > thr are included.
    thr: float
        Optional threshold value for conditioning. Defaults to 0.

    Returns
    -------
    result: dict
        Dictionary containing the verification results.

    Notes
    -----
    Multiplicative scores can be computed by passing log-tranformed values.
    Note that "scatter" is the only score that will be computed in dB units of
    the multiplicative error, i.e.: 10*log10(pred/obs).

    beta1 measures the degree of conditional bias of the observations given the
    forecasts (type 1).

    beta2 measures the degree of conditional bias of the forecasts given the
    observations (type 2).

    The normalized MSE is computed as
    NMSE = E[(pred - obs)^2]/E[(pred + obs)^2].

    The debiased RMSE is computed as DRMSE = sqrt(MSE - ME^2).

    The reduction of variance score is computed as RV = 1 - MSE/Var(obs).

    Score names denoted by * can only be computed offline, meaning that the
    these cannot be computed using _init, _accum and _compute methods of this
    module.

    References
    ----------
    Germann, U. , Galli, G. , Boscacci, M. and Bolliger, M. (2006), Radar
    precipitation measurement in a mountainous region. Q.J.R. Meteorol. Soc.,
    132: 1669-1692. doi:10.1256/qj.05.190

    Potts, J. (2012), Chapter 2 - Basic concepts. Forecast verification: a
    practitioner’s guide in atmospheric sciences, I. T. Jolliffe, and D. B.
    Stephenson, Eds., Wiley-Blackwell, 11–29.

    See also
    --------
    pysteps.verification.detcatscores.det_cat_fct
    """

    # catch case of single score passed as string
    def get_iterable(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        else:
            return (x,)

    scores = get_iterable(scores)

    # split between online and offline scores
    loffline = ["scatter", "corr_s"]
    onscores = [
        score for score in scores if str(score).lower() not in loffline or score == ""
    ]
    offscores = [
        score for score in scores if str(score).lower() in loffline or score == ""
    ]

    # unique lists
    onscores = _uniquelist(onscores)
    offscores = _uniquelist(offscores)

    # online scores
    onresult = {}
    if onscores:

        err = det_cont_fct_init(axis=axis, conditioning=conditioning, thr=thr)
        det_cont_fct_accum(err, pred, obs)
        onresult = det_cont_fct_compute(err, onscores)

    # offline scores
    offresult = {}
    if offscores:

        pred = np.asarray(pred.copy())
        obs = np.asarray(obs.copy())

        if pred.shape != obs.shape:
            raise ValueError(
                "the shape of pred does not match the shape of obs %s!=%s"
                % (pred.shape, obs.shape)
            )

        # conditioning
        if conditioning is not None:
            if conditioning == "single":
                idx = np.logical_or(obs > thr, pred > thr)
            elif conditioning == "double":
                idx = np.logical_and(obs > thr, pred > thr)
            else:
                raise ValueError("unkown conditioning %s" % conditioning)
            obs[~idx] = np.nan
            pred[~idx] = np.nan

        for score in offscores:
            # catch None passed as score
            if score is None:
                continue

            score_ = score.lower()

            # spearman corr (rank correlation)
            if score_ in ["corr_s", "spearmanr", ""]:
                corr_s = _spearmanr(pred, obs, axis=axis)
                offresult["corr_s"] = corr_s

            # scatter
            if score_ in ["scatter", ""]:
                scatter = _scatter(pred, obs, axis=axis)
                offresult["scatter"] = scatter

    # pull all results together
    result = onresult
    result.update(offresult)

    return result


def det_cont_fct_init(axis=None, conditioning=None, thr=0.0):
    """Initialize a verification error object.

    Parameters
    ----------
    axis: {int, tuple of int, None}, optional
        Axis or axes along which a score is integrated. The default, axis=None,
        will integrate all of the elements of the input arrays.\n
        If axis is -1 (or any negative integer),
        the integration is not performed
        and scores are computed on all of the elements in the input arrays.\n
        If axis is a tuple of ints, the integration is performed on all of the
        axes specified in the tuple.
    conditioning: {None, "single", "double"}, optional
        The type of conditioning used for the verification.
        The default, conditioning=None, includes all pairs. With
        conditioning="single", only pairs with either pred or obs > thr are
        included. With conditioning="double", only pairs with both pred and
        obs > thr are included.
    thr: float
        Optional threshold value for conditioning. Defaults to 0.

    Returns
    -------
    out: dict
        The verification error object.
    """

    err = {}

    # catch case of axis passed as integer
    def get_iterable(x):
        if x is None or (
            isinstance(x, collections.abc.Iterable) and not isinstance(x, int)
        ):
            return x
        else:
            return (x,)

    err["axis"] = get_iterable(axis)
    err["conditioning"] = conditioning
    err["thr"] = thr
    err["cov"] = None
    err["vobs"] = None
    err["vpred"] = None
    err["mobs"] = None
    err["mpred"] = None
    err["me"] = None
    err["mse"] = None
    err["mss"] = None  # mean square sum, i.e. E[(pred + obs)^2]
    err["mae"] = None
    err["n"] = None

    return err


def det_cont_fct_accum(err, pred, obs):
    """Accumulate the forecast error in the verification error object.

    Parameters
    ----------
    err: dict
        A verification error object initialized with
        :py:func:`pysteps.verification.detcontscores.det_cont_fct_init`.
    pred: array_like
        Array of predictions. NaNs are ignored.
    obs: array_like
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
    if err["cov"] is None:
        # initialize the error arrays in the verification object
        err["cov"] = np.zeros(nshape)
        err["vobs"] = np.zeros(nshape)
        err["vpred"] = np.zeros(nshape)
        err["mobs"] = np.zeros(nshape)
        err["mpred"] = np.zeros(nshape)
        err["me"] = np.zeros(nshape)
        err["mse"] = np.zeros(nshape)
        err["mss"] = np.zeros(nshape)
        err["mae"] = np.zeros(nshape)
        err["n"] = np.zeros(nshape)

    else:
        # check dimensions
        if err["cov"].shape != nshape:
            raise ValueError(
                "the shape of the input arrays does not match "
                + "the shape of the "
                + "verification object %s!=%s" % (nshape, err["cov"].shape)
            )

    # conditioning
    if err["conditioning"] is not None:
        if err["conditioning"] == "single":
            idx = np.logical_or(obs > err["thr"], pred > err["thr"])
        elif err["conditioning"] == "double":
            idx = np.logical_and(obs > err["thr"], pred > err["thr"])
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
    sum = pred + obs
    n = np.sum(np.isfinite(res), axis=axis)

    # new means
    mobs = np.nanmean(obs, axis=axis)
    mpred = np.nanmean(pred, axis=axis)
    me = np.nanmean(res, axis=axis)
    mse = np.nanmean(res ** 2, axis=axis)
    mss = np.nanmean(sum ** 2, axis=axis)
    mae = np.nanmean(np.abs(res), axis=axis)

    # expand axes for broadcasting
    for ax in sorted(axis):
        mobs = np.expand_dims(mobs, ax)
        mpred = np.expand_dims(mpred, ax)

    # new cov matrix
    cov = np.nanmean((obs - mobs) * (pred - mpred), axis=axis)
    vobs = np.nanmean(np.abs(obs - mobs) ** 2, axis=axis)
    vpred = np.nanmean(np.abs(pred - mpred) ** 2, axis=axis)

    mobs = mobs.squeeze()
    mpred = mpred.squeeze()

    # update variances
    _parallel_var(err["mobs"], err["n"], err["vobs"], mobs, n, vobs)
    _parallel_var(err["mpred"], err["n"], err["vpred"], mpred, n, vpred)

    # update covariance
    _parallel_cov(err["cov"], err["mobs"], err["mpred"], err["n"], cov, mobs, mpred, n)

    # update means
    _parallel_mean(err["mobs"], err["n"], mobs, n)
    _parallel_mean(err["mpred"], err["n"], mpred, n)
    _parallel_mean(err["me"], err["n"], me, n)
    _parallel_mean(err["mse"], err["n"], mse, n)
    _parallel_mean(err["mss"], err["n"], mss, n)
    _parallel_mean(err["mae"], err["n"], mae, n)

    # update number of samples
    err["n"] += n


def det_cont_fct_merge(err_1, err_2):
    """Merge two verification error objects.

    Parameters
    ----------
    err_1: dict
      A verification error object initialized with
      :py:func:`pysteps.verification.detcontscores.det_cont_fct_init`
      and populated with
      :py:func:`pysteps.verification.detcontscores.det_cont_fct_accum`.
    err_2: dict
      Another verification error object initialized with
      :py:func:`pysteps.verification.detcontscores.det_cont_fct_init`
      and populated with
      :py:func:`pysteps.verification.detcontscores.det_cont_fct_accum`.

    Returns
    -------
    out: dict
      The merged verification error object.
    """

    # checks
    if err_1["axis"] != err_2["axis"]:
        raise ValueError(
            "cannot merge: the axis are not same %s!=%s"
            % (err_1["axis"], err_2["axis"])
        )
    if err_1["conditioning"] != err_2["conditioning"]:
        raise ValueError(
            "cannot merge: the conditioning is not same %s!=%s"
            % (err_1["conditioning"], err_2["conditioning"])
        )
    if err_1["thr"] != err_2["thr"]:
        raise ValueError(
            "cannot merge: the threshold is not same %s!=%s"
            % (err_1["thr"], err_2["thr"])
        )
    if err_1["cov"] is None or err_2["cov"] is None:
        raise ValueError("cannot merge: no data found")

    # merge the two verification error objects
    err = err_1.copy()

    # update variances
    _parallel_var(
        err["mobs"], err["n"], err["vobs"], err_2["mobs"], err_2["n"], err_2["vobs"]
    )
    _parallel_var(
        err["mpred"],
        err["n"],
        err["vpred"],
        err_2["mpred"],
        err_2["n"],
        err_2["vpred"],
    )

    # update covariance
    _parallel_cov(
        err["cov"],
        err["mobs"],
        err["mpred"],
        err["n"],
        err_2["cov"],
        err_2["mobs"],
        err_2["mpred"],
        err_2["n"],
    )

    # update means
    _parallel_mean(err["mobs"], err["n"], err_2["mobs"], err_2["n"])
    _parallel_mean(err["mpred"], err["n"], err_2["mpred"], err_2["n"])
    _parallel_mean(err["me"], err["n"], err_2["me"], err_2["n"])
    _parallel_mean(err["mse"], err["n"], err_2["mse"], err_2["n"])
    _parallel_mean(err["mss"], err["n"], err_2["mss"], err_2["n"])
    _parallel_mean(err["mae"], err["n"], err_2["mae"], err_2["n"])

    # update number of samples
    err["n"] += err_2["n"]

    return err


def det_cont_fct_compute(err, scores=""):
    """Compute simple and skill scores for deterministic continuous forecasts
    from a verification error object.

    Parameters
    ----------
    err: dict
        A verification error object initialized with
        :py:func:`pysteps.verification.detcontscores.det_cont_fct_init` and
        populated with
        :py:func:`pysteps.verification.detcontscores.det_cont_fct_accum`.
    scores: {string, list of strings}, optional
        The name(s) of the scores. The default, scores="", will compute all
        available scores.
        The available score names are:

        .. tabularcolumns:: |p{2cm}|L|

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  beta1      | linear regression slope (type 1 conditional bias)     |
        +------------+--------------------------------------------------------+
        |  beta2      | linear regression slope (type 2 conditional bias)     |
        +------------+--------------------------------------------------------+
        |  corr_p    | pearson's correleation coefficien (linear correlation) |
        +------------+--------------------------------------------------------+
        |  DRMSE     | debiased root mean squared error, i.e.                 |
        |            | :math:`DRMSE = \\sqrt{RMSE - ME^2}`                     |
        +------------+--------------------------------------------------------+
        |  MAE       | mean absolute error                                    |
        +------------+--------------------------------------------------------+
        |  ME        | mean error or bias                                     |
        +------------+--------------------------------------------------------+
        |  MSE       | mean squared error                                     |
        +------------+--------------------------------------------------------+
        |  NMSE      | normalized mean squared error                          |
        +------------+--------------------------------------------------------+
        |  RMSE      | root mean squared error                                |
        +------------+--------------------------------------------------------+
        |  RV        | reduction of variance                                  |
        |            | (Brier Score, Nash-Sutcliffe Efficiency), i.e.         |
        |            | :math:`RV = 1 - \\frac{MSE}{s^2_o}`                     |
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

    result = {}
    for score in scores:
        # catch None passed as score
        if score is None:
            continue

        score_ = score.lower()

        # bias (mean error, systematic error)
        if score_ in ["bias", "me", ""]:
            bias = err["me"]
            result["ME"] = bias

        # mean absolute error
        if score_ in ["mae", ""]:
            MAE = err["mae"]
            result["MAE"] = MAE

        # mean squared error
        if score_ in ["mse", ""]:
            MSE = err["mse"]
            result["MSE"] = MSE

        # normalized mean squared error
        if score_ in ["nmse", ""]:
            NMSE = err["mse"] / err["mss"]
            result["NMSE"] = NMSE

        # root mean squared error
        if score_ in ["rmse", ""]:
            RMSE = np.sqrt(err["mse"])
            result["RMSE"] = RMSE

        # linear correlation coeff (pearson corr)
        if score_ in ["corr_p", "pearsonr", ""]:
            corr_p = err["cov"] / np.sqrt(err["vobs"]) / np.sqrt(err["vpred"])
            result["corr_p"] = corr_p

        # beta1 (linear regression slope)
        if score_ in ["beta", "beta1", ""]:
            beta1 = err["cov"] / err["vpred"]
            result["beta1"] = beta1

        # beta2 (linear regression slope)
        if score_ in ["beta2", ""]:
            beta2 = err["cov"] / err["vobs"]
            result["beta2"] = beta2

        # debiased RMSE
        if score_ in ["drmse", ""]:
            RMSE_d = np.sqrt(err["mse"] - err["me"] ** 2)
            result["DRMSE"] = RMSE_d

        # reduction of variance
        # (Brier Score, Nash-Sutcliffe efficiency coefficient,
        # MSE skill score)
        if score_ in ["rv", "brier_score", "nse", ""]:
            RV = 1.0 - err["mse"] / err["vobs"]
            result["RV"] = RV

    return result


def _parallel_mean(avg_a, count_a, avg_b, count_b):
    """Update avg_a with avg_b."""
    idx = count_b > 0
    avg_a[idx] = (count_a[idx] * avg_a[idx] + count_b[idx] * avg_b[idx]) / (
        count_a[idx] + count_b[idx]
    )


def _parallel_var(avg_a, count_a, var_a, avg_b, count_b, var_b):
    """Update var_a with var_b.
    source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    idx = count_b > 0
    delta = avg_b - avg_a
    m_a = var_a * count_a
    m_b = var_b * count_b
    var_a[idx] = (
        m_a[idx]
        + m_b[idx]
        + delta[idx] ** 2 * count_a[idx] * count_b[idx] / (count_a[idx] + count_b[idx])
    )
    var_a[idx] = var_a[idx] / (count_a[idx] + count_b[idx])


def _parallel_cov(cov_a, avg_xa, avg_ya, count_a, cov_b, avg_xb, avg_yb, count_b):
    """Update cov_a with cov_b."""
    idx = count_b > 0
    deltax = avg_xb - avg_xa
    deltay = avg_yb - avg_ya
    c_a = cov_a * count_a
    c_b = cov_b * count_b
    cov_a[idx] = (
        c_a[idx]
        + c_b[idx]
        + deltax[idx]
        * deltay[idx]
        * count_a[idx]
        * count_b[idx]
        / (count_a[idx] + count_b[idx])
    )
    cov_a[idx] = cov_a[idx] / (count_a[idx] + count_b[idx])


def _uniquelist(mylist):
    used = set()
    return [x for x in mylist if x not in used and (used.add(x) or True)]


def _scatter(pred, obs, axis=None):

    pred = pred.copy()
    obs = obs.copy()

    # catch case of axis passed as integer
    def get_iterable(x):
        if x is None or (
            isinstance(x, collections.abc.Iterable) and not isinstance(x, int)
        ):
            return x
        else:
            return (x,)

    axis = get_iterable(axis)

    # reshape arrays as 2d matrices
    # rows: samples; columns: variables
    axis = tuple(range(pred.ndim)) if axis is None else axis
    axis = tuple(np.sort(axis))
    for ax in axis:
        pred = np.rollaxis(pred, ax, 0)
        obs = np.rollaxis(obs, ax, 0)
    shp_rows = pred.shape[: len(axis)]
    shp_cols = pred.shape[len(axis) :]
    pred = np.reshape(pred, (np.prod(shp_rows), -1))
    obs = np.reshape(obs, (np.prod(shp_rows), -1))

    # compute multiplicative erros in dB
    q = 10 * np.log10(pred / obs)

    # nans are given zero weight and are set equal to (min value - 1)
    idkeep = np.isfinite(q)
    q[~idkeep] = q[idkeep].min() - 1
    obs[~idkeep] = 0

    # compute scatter along rows
    xs = np.sort(q, axis=0)
    xs = np.vstack((xs[0, :], xs))
    ixs = np.argsort(q, axis=0)
    ws = np.take_along_axis(obs, ixs, axis=0)
    ws = np.vstack((ws[0, :] * 0.0, ws))
    wsc = np.cumsum(ws, axis=0) / np.sum(ws, axis=0)
    xint = np.zeros((2, xs.shape[1]))
    for i in range(xint.shape[1]):
        xint[:, i] = np.interp([0.16, 0.84], wsc[:, i], xs[:, i])
    scatter = (xint[1, :] - xint[0, :]) / 2.0

    # reshape back
    scatter = scatter.reshape(shp_cols)

    return float(scatter) if scatter.size == 1 else scatter


def _spearmanr(pred, obs, axis=None):

    pred = pred.copy()
    obs = obs.copy()

    # catch case of axis passed as integer
    def get_iterable(x):
        if x is None or (
            isinstance(x, collections.abc.Iterable) and not isinstance(x, int)
        ):
            return x
        else:
            return (x,)

    axis = get_iterable(axis)

    # reshape arrays as 2d matrices
    # rows: samples; columns: variables
    axis = tuple(range(pred.ndim)) if axis is None else axis
    axis = tuple(np.sort(axis))
    for ax in axis:
        pred = np.rollaxis(pred, ax, 0)
        obs = np.rollaxis(obs, ax, 0)
    shp_rows = pred.shape[: len(axis)]
    shp_cols = pred.shape[len(axis) :]
    pred = np.reshape(pred, (np.prod(shp_rows), -1))
    obs = np.reshape(obs, (np.prod(shp_rows), -1))

    # apply only with more than 2 valid samples
    # although this does not seem to solve the error
    # "ValueError: The input must have at least 3 entries!" ...
    corr_s = np.zeros(pred.shape[1]) * np.nan
    nsamp = np.sum(np.logical_and(np.isfinite(pred), np.isfinite(obs)), axis=0)
    idx = nsamp > 2
    if np.any(idx):
        corr_s_ = spearmanr(pred[:, idx], obs[:, idx], axis=0, nan_policy="omit")[0]

        if corr_s_.size > 1:
            corr_s[idx] = np.diag(corr_s_, idx.sum())
        else:
            corr_s = corr_s_

    return float(corr_s) if corr_s.size == 1 else corr_s.reshape(shp_cols)
