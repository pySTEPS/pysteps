
def get_method(name, type="deterministic"):
    """Return a callable function for the method corresponding to the given
    verification score.

    Parameters
    ----------
    name : str
        Name of the verification method. The available options are:\n\

        type: deterministic

        +------------+----------------------------------------------------------+
        | Name       | Description                                              |
        +============+==========================================================+
        |  ACC       | accuracy (proportion correct)                            |
        +------------+----------------------------------------------------------+
        |  BIAS      | frequency bias                                           |
        +------------+----------------------------------------------------------+
        |  CSI       | critical success index (threat score)                    |
        +------------+----------------------------------------------------------+
        |  FA        | false alarm rate (prob. of false detection)              |
        +------------+----------------------------------------------------------+
        |  FAR       | false alarm ratio                                        |
        +------------+----------------------------------------------------------+
        |  GSS       | Gilbert skill score (equitable threat score)             |
        +------------+----------------------------------------------------------+
        |  HK        | Hanssen-Kuipers discriminant (Pierce skill score)        |
        +------------+----------------------------------------------------------+
        |  HSS       | Heidke skill score                                       |
        +------------+----------------------------------------------------------+
        |  POD       | probability of detection (hit rate)                      |
        +------------+----------------------------------------------------------+
        |  SEDI      | symmetric extremal dependency index                      |
        +------------+----------------------------------------------------------+
        |  beta      | linear regression slope (conditional bias)               |
        +------------+----------------------------------------------------------+
        |  corr_p    | pearson's correleation coefficien (linear correlation)   |
        +------------+----------------------------------------------------------+
        |  corr_s    | spearman's correlation coefficient (rank correlation)    |
        +------------+----------------------------------------------------------+
        |  MAE_add   | mean absolute error of additive residuals                |
        +------------+----------------------------------------------------------+
        |  MAE_mul   | mean absolute error of multiplicative residuals          |
        +------------+----------------------------------------------------------+
        |  ME_add    | mean error or bias of additive residuals                 |
        +------------+----------------------------------------------------------+
        |  ME_mult   | mean error or bias of multiplicative residuals           |
        +------------+----------------------------------------------------------+
        |  RMSE_add  | root mean squared additive error                         |
        +------------+----------------------------------------------------------+
        |  RMSE_mult | root mean squared multiplicative error                   |
        +------------+----------------------------------------------------------+
        |  RV_add    | reduction of variance (Brier Score, Nash-Sutcliffe       |
        |            | Efficiency)                                              |
        +------------+----------------------------------------------------------+
        |  RV_mult   | reduction of variance in multiplicative space            |
        +------------+----------------------------------------------------------+
        |  scatter   | half the distance between the 16% and 84% percentiles of |
        |            | the error distribution                                   |
        +------------+----------------------------------------------------------+
        |  binary_mse| binary MSE                                               |
        +------------+----------------------------------------------------------+
        |  FSS       | fractions skill score                                    |
        +------------+----------------------------------------------------------+

        type: ensemble

        +------------+----------------------------------------------------------+
        | Name       | Description                                              |
        +============+==========================================================+
        | ens_skill  | mean ensemble skill                                      |
        +------------+----------------------------------------------------------+
        | ens_spread | mean ensemble spread                                     |
        +------------+----------------------------------------------------------+
        | rankhist   | rank histogram                                           |
        +------------+----------------------------------------------------------+

        type: probabilistic

        +------------+----------------------------------------------------------+
        | Name       | Description                                              |
        +============+==========================================================+
        |  CRPS      | continuous ranked probability score                      |
        +------------+----------------------------------------------------------+
        |  reldiag   | reliability diagram                                      |
        +------------+----------------------------------------------------------+
        |  ROC       | ROC curve                                                |
        +------------+----------------------------------------------------------+

    type : str
        Type of the method. The available options are 'deterministic', 'ensemble'
        and 'probabilistic'.

    """

    if name is None:
        name = 'none'
    if type is None:
        type = 'none'

    name = name.lower()
    type = type.lower()

    if type in ["deterministic"]:

        from .detcatscores import det_cat_fcst
        from .detcontscores import det_cont_fcst
        from .spatialscores import fss, binary_mse

        # categorical
        if name in ["acc", "csi", "fa", "far", "gss", "hk", "hss", "pod", "sedi"]:
            def f(fct, obs, **kwargs):
                return det_cat_fcst(fct, obs, kwargs.pop("thr"), [name])
            return f

        # continuous
        elif name in ["beta", "corr_p", "corr_s", "mae_add", "mae_mult",
                      "me_add", "me_mult", "rmse_add", "rmse_mult", "rv_add",
                      "rv_mult", "scatter"]:
            def f(fct, obs, **kwargs):
                return det_cont_fcst(fct, obs, [name], **kwargs)
            return f

        # spatial
        elif name in ["binary_mse"]:
            return binary_mse
        elif name in ["fss"]:
            return fss
        else:
            raise ValueError("unknown deterministic method %s" % name)

    elif type in ["ensemble"]:

        from .ensscores import ensemble_skill, ensemble_spread, rankhist_init, \
          rankhist_accum, rankhist_compute

        if name in ["ens_skill"]:
            return ensemble_skill
        elif name in ["ens_spread"]:
            return ensemble_spread
        elif name in ["rankhist"]:
            return rankhist_init, rankhist_accum, rankhist_compute
        else:
            raise ValueError("unknown ensemble method %s" % name)

    elif type in ["probabilistic"]:

        from .probscores import CRPS, reldiag_init, reldiag_accum, reldiag_compute, ROC_curve_init, ROC_curve_accum, ROC_curve_compute

        if name in ["crps"]:
            return CRPS
        elif name in ["reldiag"]:
            return reldiag_init, reldiag_accum, reldiag_compute
        elif name in ["roc"]:
            return ROC_curve_init, ROC_curve_accum, ROC_curve_compute
        else:
            raise ValueError("unknown probabilistic method %s" % name)

    else:
        raise ValueError("unknown type %s" % name)
