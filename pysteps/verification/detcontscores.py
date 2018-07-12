"""Forecast evaluation and skill scores for deterministic continuous forecasts.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr

def scores_det_cont_fcst(pred, obs, 
                         scores=['corr_p'], 
                         offset=0.01):
                         
    """ Calculate simple and skill scores for deterministic continuous forecasts
    
    Input:
    ------
    pred: array-like
        predictions
    obs: array-like
        verifiyinig observations
    scores : list
        names of the scores to be computed, the full list is:
        ['ME_add', 'RMSE_add', 'RV_add', 'corr_s', 'corr_p', 'beta', 'ME_mult', 
        'RMSE_mult', 'RV_mult']

    Return:
    ------
    result : list
        the verification results
    """

    # flatten array if 2D
    pred = pred.flatten()
    obs = obs.flatten()
    
    isNaN = np.isnan(pred) | np.isnan(obs)
    pred = pred[~isNaN]
    obs = obs[~isNaN]
    
    N = len(obs)
    s_o = np.sqrt(1.0/N*sum((obs-obs.mean())**2))
    s_pred = np.sqrt(1.0/N*sum((pred-pred.mean())**2)) # sample standard deviation of prediction
    
    # compute additive and multiplicative residuals
    add_res = pred-obs
    mult_res = 10.0*np.log10((pred + offset)/(obs + offset))
    
    result = []
    for score in scores:
    
        # mean error (stm called bias... but somehow doesn't add up with multiplicative bias from Christoph Frei's lecture)
        if score.lower() == 'me_add':
            ME_add = np.mean(add_res)
            result.append(ME_add)
        
        if score.lower() == 'me_mult':
            ME_mult = np.mean(mult_res)
            result.append(ME_mult)
        
        # root mean squared errors
        if score.lower() == 'rmse_add':
            RMSE_add = np.sqrt(1.0/N*sum((add_res)**2))
            result.append(RMSE_add)
        
        if score.lower() == 'rmse_mult':
            RMSE_mult = np.sqrt(1.0/N*sum((mult_res)**2))
            result.append(RMSE_mult)
        
        # reduction of variance scores (not sure whether even makes sense in multiplicative space)
        if score.lower() == 'rv_add':
            RV_add = 1.0 - 1.0/N*sum((add_res)**2)/s_o**2
            result.append(RV_add)
        
        if score.lower() == 'rv_mult':
            dBo = 10*np.log10(obs+offset)
            s_dBo = np.sqrt(1.0/N*sum((dBo-dBo.mean())**2))
            RV_mult = 1.0-1.0/N*sum((mult_res)**2)/s_dBo**2
            result.append(RV_mult)
        
        # spearman corr (rank correlation)
        if score.lower() == 'corr_s':
            corr_s = spearmanr(pred,obs)[0]
            result.append(corr_s)
        
        # pearson corr
        if score.lower() == 'corr_p':
            corr_p = pearsonr(pred,obs)[0]
            result.append(corr_p)
            
        # beta (linear regression slope)
        if score.lower() == 'beta':
            beta = s_o/s_pred*corr_p
            result.append(beta)
    
    return result