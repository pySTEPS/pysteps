"""
pysteps.verification.interface
==============================

Interface for the verification module.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

def get_method(name, type="deterministic"):
    """Return a callable function for the method corresponding to the given
    verification score.

    Parameters
    ----------
    name : str
        Name of the verification method. The available options are:\n\

        type: deterministic

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
        |  beta      | linear regression slope (conditional bias)             |
        +------------+--------------------------------------------------------+
        |  corr_p    | pearson's correleation coefficien (linear correlation) |
        +------------+--------------------------------------------------------+
        |  corr_s*   | spearman's correlation coefficient (rank correlation)  |
        +------------+--------------------------------------------------------+
        |  DRMSE     | debiased root mean squared error                       |
        +------------+--------------------------------------------------------+
        |  MAE       | mean absolute error of residuals                       |
        +------------+--------------------------------------------------------+
        |  ME        | mean error or bias of residuals                        |
        +------------+--------------------------------------------------------+
        |  MSE       | mean squared error                                     |
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
        |  binary_mse| binary MSE                                             |
        +------------+--------------------------------------------------------+
        |  FSS       | fractions skill score                                  |
        +------------+--------------------------------------------------------+

        type: ensemble

        .. tabularcolumns:: |p{2cm}|L|

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        | ens_skill  | mean ensemble skill                                    |
        +------------+--------------------------------------------------------+
        | ens_spread | mean ensemble spread                                   |
        +------------+--------------------------------------------------------+
        | rankhist   | rank histogram                                         |
        +------------+--------------------------------------------------------+

        type: probabilistic

        .. tabularcolumns:: |p{2cm}|L|

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  CRPS      | continuous ranked probability score                    |
        +------------+--------------------------------------------------------+
        |  reldiag   | reliability diagram                                    |
        +------------+--------------------------------------------------------+
        |  ROC       | ROC curve                                              |
        +------------+--------------------------------------------------------+

    type : {'deterministic', 'ensemble', 'probabilistic'}, optional
        Type of the verification method.

    Notes
    -----

    Multiplicative scores can be computed by passing log-tranformed values.
    Note that "scatter" is the only score that will be computed in dB units of
    the multiplicative error, i.e.: 10log10(pred/obs).

    The debiased RMSE is computed as DRMSE = sqrt(RMSE - ME^2)

    The reduction of variance score is computed as RV = 1 - MSE/Var(obs)

    Score names denoted by * can only be computed offline, meaning that the
    these cannot be update using _init, _accum and _compute methods of this
    module.

    Score names denoted by * can only be computed offline.

    References
    ----------

    Germann, U. , Galli, G. , Boscacci, M. and Bolliger, M. (2006), Radar
    precipitation measurement in a mountainous region. Q.J.R. Meteorol. Soc.,
    132: 1669-1692. doi:10.1256/qj.05.190

    """

    if name is None:
        name = 'none'
    if type is None:
        type = 'none'

    name = name.lower()
    type = type.lower()

    if type in ["deterministic"]:

        from .detcatscores import det_cat_fct
        from .detcontscores import det_cont_fct
        from .spatialscores import fss, binary_mse

        # categorical
        if name in ["acc", "csi", "fa", "far", "gss", "hk", "hss", "pod", "sedi"]:
            def f(fct, obs, **kwargs):
                return det_cat_fct(fct, obs, kwargs.pop("thr"), [name])
            return f

        # continuous
        elif name in ["beta", "corr_p", "corr_s", "mae", "mse",
                      "me", "drmse", "rmse", "rv", "scatter"]:
            def f(fct, obs, **kwargs):
                return det_cont_fct(fct, obs, [name], **kwargs)
            return f

        # spatial
        elif name in ["binary_mse"]:
            return binary_mse
        elif name in ["fss"]:
            return fss

        else:
            raise ValueError("unknown deterministic method %s" % name)

    elif type in ["ensemble"]:

        from .ensscores import ensemble_skill, ensemble_spread, rankhist

        if name in ["ens_skill"]:
            return ensemble_skill
        elif name in ["ens_spread"]:
            return ensemble_spread
        elif name in ["rankhist"]:
            return rankhist
        else:
            raise ValueError("unknown ensemble method %s" % name)

    elif type in ["probabilistic"]:

        from .probscores import CRPS, reldiag, ROC_curve

        if name in ["crps"]:
            return CRPS
        elif name in ["reldiag"]:
            return reldiag
        elif name in ["roc"]:
            return ROC_curve
        else:
            raise ValueError("unknown probabilistic method %s" % name)

    else:
        raise ValueError("unknown type %s" % name)
