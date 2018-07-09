"""Forecast evaluation and skill scores for deterministic categorial forecasts.
"""

import numpy as np

def scores_det_cat_fcst(pred, obs, thr,
                        scores=['csi']):
                         
    """ Calculate simple and skill scores for deterministic categorical forecasts
    
    Input:
    ------
    pred: array-like
        predictions
    obs: array-like
        verifiyinig observations
    scores : list
        names of the scores to be computed

    Return:
    ------
    result : list
        the verification results
    """
    
    # flatten array if 2D
    pred = pred.flatten()
    obs = obs.flatten()
    
    # apply threshold
    predb = pred > thr
    obsb  = obs > thr
    
    # calculate hits, misses, false positives, correct rejects   
    H_idx = np.logical_and(predb==1, obsb==1) # correctly predicted precip
    F_idx = np.logical_and(predb==1, obsb==0) # predicted precip even though none there
    M_idx = np.logical_and(predb==0, obsb==1) # predicted no precip even though there was
    R_idx = np.logical_and(predb==0, obsb==0) # correctly predicted no precip

    H = sum(H_idx).astype(float) # hits
    M = sum(M_idx).astype(float) # misses
    F = sum(F_idx).astype(float) # false alarms
    R = sum(R_idx).astype(float) # correct rejections
    tot = H+M+F+R
    
    result = []
    for score in scores:
         
        # simple scores 
        POD = H/(H+M) # probability of detection
        FAR = F/(H+F) # false alarm ratio
        FA = F/(F+R) # false alarm rate = prob of false detection
        s = (H+M)/(H+M+F+R) # base rate = freq of observed events
            
        if score.lower() == 'pod':
            result.append(POD)
        if score.lower() == 'far':
            result.append(FAR)
        if score.lower() == 'fa':
            result.append(FA)
        if score.lower() == 'acc':
            ACC = (H+R)/(H+M+F+R) # accuracy (fraction correct) 
            result.append(ACC)
        if score.lower() == 'csi':
            CSI = H/(H+M+F) # critical success index
            result.append(CSI)
        
        # skill scores
        if score.lower() == 'hss':
            HSS = 2*(H*R-F*M)/((H+M)*(M+R)+(H+F)*(F+R)) # Heidke Skill Score (-1 < HSS < 1) < 0 implies no skill
            result.append(HSS)
        if score.lower() == 'hk':
            HK = POD-FA # Hanssen-Kuipers Discriminant
            result.append(HK)
        if score.lower() == 'gss':
            GSS = (POD-FA)/((1-s*POD)/(1-s)+FA*(1-s)/s) # Gilbert Skill Score
            result.append(GSS)
        if score.lower() == 'sedi':
            # Symmetric extremal dependence index
            SEDI = (np.log(FA)-np.log(POD)+np.log(1-POD)-np.log(1-FA))/(np.log(FA)
                    +np.log(POD)+np.log(1-POD)+np.log(1-FA))
            result.append(SEDI)
            
    return result