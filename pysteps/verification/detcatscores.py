"""Forecast evaluation and skill scores for deterministic categorial forecasts."""

import numpy as np

def det_cat_fcst(pred, obs, thr, scores):
    """Calculate simple and skill scores for deterministic categorical forecasts.

    Parameters
    ----------
    pred : array_like
        predictions
    obs : array_like
        verifying observations
    scores : list
        a list containing the names of the scores to be computed, the full list
        is:

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

    H = sum(H_idx.astype(int)) # hits
    M = sum(M_idx.astype(int)) # misses
    F = sum(F_idx.astype(int)) # false alarms
    R = sum(R_idx.astype(int)) # correct rejections

    result = []
    for score in scores:

        score = score.lower()

        # simple scores
        POD = H/float(H+M)       # probability of detection
        FAR = F/float(H+F)       # false alarm ratio
        FA  = F/float(F+R)        # false alarm rate = prob of false detection
        s   = (H+M)/float(H+M+F+R) # base rate = freq of observed events

        if score == 'pod':
            result.append(POD)
        if score == 'far':
            result.append(FAR)
        if score == 'fa':
            result.append(FA)
        if score == 'acc':
            ACC = (H+R)/(H+M+F+R) # accuracy (fraction correct)
            result.append(ACC)
        if score == 'csi':
            CSI = H/(H+M+F)       # critical success index
            result.append(CSI)
        if score == 'bias': # frequency bias
            B = (H + F) / (H + M)
            result.append(B)

        # skill scores
        if score == 'hss':
            HSS = 2*(H*R-F*M)/((H+M)*(M+R)+(H+F)*(F+R)) # Heidke Skill Score (-1 < HSS < 1) < 0 implies no skill
            result.append(HSS)
        if score == 'hk':
            HK = POD-FA # Hanssen-Kuipers Discriminant
            result.append(HK)
        if score == 'gss':
            GSS = (POD-FA)/((1-s*POD)/(1-s)+FA*(1-s)/s) # Gilbert Skill Score
            result.append(GSS)
        if score == 'sedi':
            # Symmetric extremal dependence index
            SEDI = (np.log(FA)-np.log(POD)+np.log(1-POD)-np.log(1-FA))/(np.log(FA)
                    +np.log(POD)+np.log(1-POD)+np.log(1-FA))
            result.append(SEDI)

    return result
