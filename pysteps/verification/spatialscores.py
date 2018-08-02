"""Skill scores for spatial forecasts"""

import numpy as np
from scipy.ndimage.filters import uniform_filter

def compute_fss(X_f, X_o, threshold=1.0, scale=32):
    """Compute the fractions skill score (FSS, Roberts and Lean 2008) for a 
    deterministic forecast field and the corresponding observation. 
    
    Parameters
    ----------
    X_f : array_like
        Array of shape (n,m) containing the deterministic forecast field.
    X_o : array_like
        Array of shape (n,m) containing the observed field
    threshold : float
        Intensity threshold.
    scale : int
        The spatial scale to verify in px. In practice it represents the size of 
        the moving window that it is used to compute the fraction of pixels above
        the threshold.
    
    Returns
    -------
    out : float
        The fractions skill score between 0 and 1.
    
    References
    ----------
    :cite:`RL2008`, :cite:`EWWM2013`
    
    """
    if X_f.shape != X_o.shape:
        raise ValueError("the shape of X_f does not match the shape of X_o (%d,%d)!=(%d,%d)" 
                            % (X_f.shape[0], X_f.shape[1], X_o.shape[0], X_o.shape[1]))

    # Convert to binary fields with the intensity threshold
    X_f = (X_f >= threshold).astype(float)
    X_o = (X_o >= threshold).astype(float) 
    
    # Compute fractions of pixels above the threshold within a square neighboring 
    # area by applying a 2D moving average to the binary fields
    X_f = uniform_filter(X_f, size=int(scale), mode="constant", cval=0.0)
    X_o = uniform_filter(X_o, size=int(scale), mode="constant", cval=0.0)
    
    n = X_f.size
    # Compute the numerator
    N = 1.0*np.sum((X_o - X_f)**2) / n
    # Compute the denominator
    D = 1.0*(np.sum(X_o**2) + np.nansum(X_f**2)) / n

    return 1 - N / D
