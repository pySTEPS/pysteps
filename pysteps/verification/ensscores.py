"""Evaluation and skill scores for ensemble forecasts."""

import numpy as np

def rankhist_init(num_ens_members, X_min):
    """Initialize a rank histogram object.
    
    Parameters
    ----------
    num_ens_members : int
      Number ensemble members in the forecasts to accumulate into the rank 
      histogram.
    X_min : float
      Threshold for minimum intensity. Forecast-observation pairs, where all 
      ensemble members and verifying observations are below X_min, are not 
      counted in the rank histogram.
    
    Returns
    -------
    out : dict
      The rank histogram object.
    """
    rankhist = {}
    
    rankhist["num_ens_members"] = num_ens_members
    rankhist["n"] = np.zeros(num_ens_members+1, dtype=int)
    rankhist["X_min"] = X_min
    
    return rankhist

def rankhist_accum(rankhist, X_f, X_o):
    """Accumulate forecast-observation pairs to the given rank histogram.
    
    Parameters
    ----------
    X_f : array-like
      Array of shape (n,m) containing the values from n ensemble forecasts with 
      m members.
    X_o : array_like
      Array of length n containing the observed values corresponding to the 
      forecast.
    """
    if X_f.shape[1] != rankhist["num_ens_members"]:
        raise ValueError("the number of ensemble members in X_f does not match the number of members in the rank histogram (%d!=%d)" % (X_f.shape[1], rankhist["num_ens_members"]))
    
    X_min = rankhist["X_min"]
    
    mask = np.logical_and(np.isfinite(X_o), np.all(np.isfinite(X_f), axis=1))
    X_f = X_f[mask, :].copy()
    X_o = X_o[mask]
    
    mask_nz = np.logical_or(X_o >= X_min, np.all(X_f >= X_min, axis=1))
    
    X_f.sort(axis=1)
    bin_idx = [np.digitize(v, f) for v,f in zip(X_o[mask_nz], X_f[mask_nz, :])]
    
    # handle ties, where the verifying observation lies between ensemble members 
    # having the same value
    # ignore the cases where the verifying observations and all ensemble 
    # members are below the threshold X_min
    for i in np.where(~mask_nz)[0]:
        if np.any(X_f[i, :] >= X_min):
            i_eq = np.where(X_f[i, :] < X_min)[0]
            if len(i_eq) > 1 and X_o[i] < X_min:
                bin_idx.append(np.random.randint(low=np.min(i_eq)+1, 
                                                 high=np.max(i_eq)+1))
    
    for bi in bin_idx:
        rankhist["n"][bi] += 1

def rankhist_compute(rankhist, normalize=True):
    """Return the rank histogram counts and optionally normalize the histogram.
    
    Parameters
    ----------
    rankhist : dict
      A rank histogram object created with rankhist_init.
    
    Returns
    -------
    out : array_like
      The counts for the n+1 bins in the rank histogram, where n is the number 
      of ensemble members.
    """
    if normalize:
        return 1.0*rankhist["n"] / sum(rankhist["n"])
    else:
        return rankhist["n"]
