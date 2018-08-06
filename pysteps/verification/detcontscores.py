"""Forecast evaluation and skill scores for deterministic continuous forecasts."""

import numpy as np
from scipy.stats import spearmanr, pearsonr

def scores_det_cont_fcst(pred, obs, scores, offset=0.01):
    """ Calculate simple and skill scores for deterministic continuous forecasts
    
    Parameters
    ----------
    pred: array-like
        predictions
    obs: array-like
        verifiyinig observations
    scores : list
        a list containing the names of the scores to be computed, the full list 
        is:
    
    +------------+--------------------------------------------------------------+
    | Name       | Description                                                  |
    +============+==============================================================+
    |  beta      | linear regression slope (conditional bias)                   |
    +------------+--------------------------------------------------------------+
    |  corr_p    | pearson's correleation coefficien (linear correlation)       |
    +------------+--------------------------------------------------------------+
    |  corr_s    | spearman's correlation coefficient (rank correlation)        |
    +------------+--------------------------------------------------------------+
    |  ME_add    | mean error or bias of additive residuals                     |
    +------------+--------------------------------------------------------------+
    |  ME_mult   | mean error or bias of multiplicative residuals               |
    +------------+--------------------------------------------------------------+
    |  RMSE_add  | root mean squared additive error                             |
    +------------+--------------------------------------------------------------+
    |  RMSE_mult | root mean squared multiplicative error                       |
    +------------+--------------------------------------------------------------+
    |  RV_add    | reduction of variance (Brier Score, Nash-Sutcliffe Efficiency|
    +------------+--------------------------------------------------------------+
    |  RV_mult   | reduction of variance in multiplicative space                |
    +------------+--------------------------------------------------------------+
    
    offset : float
        an offset that is added to both prediction and observation to avoid 0 division
        when computing multiplicative residuals
    
    Returns
    -------
    result : list
        list containing the verification results
        
    """

    # flatten array if 2D
    pred  = pred.flatten()
    obs   = obs.flatten()
    
    isNaN = np.logical_or(np.isnan(pred), np.isnan(obs))
    pred  = pred[~isNaN]
    obs   = obs[~isNaN]
    
    N     = len(obs)
    s_o   = np.sqrt(1.0/N*sum((obs - obs.mean())**2))
    s_pred = np.sqrt(1.0/N*sum((pred - pred.mean())**2)) # sample standard deviation of prediction
    
    # compute additive and multiplicative residuals
    add_res = pred - obs
    mult_res = 10.0*np.log10((pred + offset)/(obs + offset))
    
    result = []
    for score in scores:
        
        score = score.lower()
        
        # mean error (stm called bias)
        if score == 'me_add':
            ME_add = np.mean(add_res)
            result.append(ME_add)
        
        if score == 'me_mult':
            ME_mult = np.mean(mult_res)
            result.append(ME_mult)
        
        # root mean squared errors
        if score == 'rmse_add':
            RMSE_add = np.sqrt(1.0/N*sum((add_res)**2))
            result.append(RMSE_add)
        
        if score == 'rmse_mult':
            RMSE_mult = np.sqrt(1.0/N*sum((mult_res)**2))
            result.append(RMSE_mult)
        
        # reduction of variance scores
        if score == 'rv_add':
            RV_add = 1.0 - 1.0/N*sum((add_res)**2)/s_o**2
            result.append(RV_add)
        
        if score == 'rv_mult':
            dBo = 10*np.log10(obs+offset)
            s_dBo = np.sqrt(1.0/N*sum((dBo-dBo.mean())**2))
            RV_mult = 1.0-1.0/N*sum((mult_res)**2)/s_dBo**2
            result.append(RV_mult)
        
        # spearman corr (rank correlation)
        if score == 'corr_s':
            corr_s = spearmanr(pred,obs)[0]
            result.append(corr_s)
        
        # pearson corr
        if score == 'corr_p':
            corr_p = pearsonr(pred,obs)[0]
            result.append(corr_p)
            
        # beta (linear regression slope)
        if score == 'beta':
            beta = s_o/s_pred*corr_p
            result.append(beta)
    
    return result