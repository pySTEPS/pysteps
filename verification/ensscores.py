"""Evaluation and skill scores for ensemble forecasts."""

import numpy as np

def rankhist_init(num_ens_members):
    """Initialize a rank histogram object.
    
    Parameters
    ----------
    num_ens_members : int
      Number ensemble members in the forecasts to accumulate into the rank 
      histogram.
    
    Returns
    -------
    out : dict
      The rank histogram object.
    """
    rankhist = {}
    
    rankhist["num_ens_members"] = num_ens_members
    rankhist["n"] = np.zeros(num_ens_members+1, dtype=int)
    
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
        raise ValueError("the number of ensemble members in X_f does not match the number of members in the rank histogram (%d!=%d)" % (X_f.shape[0], rankhist["num_ens_members"]))
    
    mask = np.logical_and(np.isfinite(X_o), np.all(np.isfinite(X_f), axis=1))
    
    X_f = X_f.copy()
    X_f.sort(axis=1)
    idx = [np.digitize(v.flatten(), f.flatten()) for v,f in \
           zip(X_o[mask], X_f[mask, :])]
    
    X_min = np.min(X_f, axis=1)
    
    # handle ties, where the verifying observation lies between ensemble members 
    # having the same value
    for i in range(len(idx)):
        i_eq = np.where(X_f[i, :] == X_o)[0]
        if len(i_eq) > 1:
            idx[i] = np.random.randint(low=i_eq[0]+1, high=i_eq[-1]+1)
    
    for i in idx:
        rankhist["n"][i] += 1

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
