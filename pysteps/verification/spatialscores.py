"""Skill scores for spatial forecasts."""

import numpy as np
from scipy.ndimage.filters import uniform_filter
try:
  import pywt
  pywt_imported = True
except ImportError:
  pywt_imported = False

def fss(X_f, X_o, thr, scale):
    """Compute the fractions skill score (FSS, Roberts and Lean 2008) for a 
    deterministic forecast field and the corresponding observation. 
    
    Parameters
    ----------
    X_f : array_like
        Array of shape (n,m) containing the deterministic forecast field.
    X_o : array_like
        Array of shape (n,m) containing the observed field.
    thr : float
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
    X_f = (X_f >= thr).astype(float)
    X_o = (X_o >= thr).astype(float) 
    
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

def intensity_scale(X_f, X_o, thrs, wavelet="haar"):
    """Compute intensity-scale verification (Casati et al. 2004). This method 
    uses PyWavelets for decomposing the error field between the forecasts and 
    observations into multiple spatial scales.
    
    Parameters
    ----------
    X_f : array_like
        Array of shape (n,m) containing the forecast field.
    X_o : array_like
        Array of shape (n,m) containing the verification observation field.
    thrs : sequence
        A sequence of intensity thresholds for which to compute the verification.
    wavelet : str
        The name of the wavelet function to use. Defaults to the Haar wavelet, 
        as described in Casati et al. 2004. See the documentation of PyWavelets 
        for a list of available options.
    
    Returns
    -------
    out : dict
        A dictionary with the following key-value pairs:
        
        +--------------+------------------------------------------------------+
        |       Key    |                Value                                 |
        +==============+======================================================+
        |  SS          | two-dimensional array containing the intensity-scale |
        |              | skill scores for each spatial scale and intensity    |
        |              | threshold                                            |
        +--------------+------------------------------------------------------+
        |  scales      | the spatial scales from largest to smallest,         |
        |              | corresponds to the first index of SS                 |
        +--------------+------------------------------------------------------+
        |  thrs        | the used intensity thresholds in increasing order,   |
        |              | corresponds to the second index of SS                |
        +--------------+------------------------------------------------------+
    
    """
    if len(X_f.shape) != 2 or len(X_o.shape) != 2 or X_f.shape != X_o.shape:
        raise ValueError("X_f and X_o must be two-dimensional arrays having the same shape")
    
    thr_min = np.min(thrs)
    X_f = X_f.copy()
    X_f[~np.isfinite(X_f)] = thr_min - 1
    X_o = X_o.copy()
    X_o[~np.isfinite(X_o)] = thr_min - 1
    
    w = pywt.Wavelet(wavelet)
    
    SS = None
    n_thrs = len(thrs)
    for i in range(n_thrs):
        I_f = (X_f >= thrs[i]).astype(float)
        I_o = (X_o >= thrs[i]).astype(float)
        
        E_decomp = _wavelet_decomp(I_f - I_o, w)
        n_scales = len(E_decomp)
        
        eps = 1.0*np.sum((X_o >= thrs[i]).astype(int)) / np.size(X_o)
        
        if SS is None:
            SS = np.empty((n_scales, n_thrs))
        
        for j in range(n_scales):
            mse = np.mean(E_decomp[j]**2)
            SS[j, i] = 1 - mse / (2*eps*(1-eps) / n_scales)
    
    out = {}
    out["SS"]     = SS
    out["scales"] = pow(2, np.arange(SS.shape[0]))[::-1]
    out["thrs"]   = thrs[:]
    
    return out

def _wavelet_decomp(X, w):
    c = pywt.wavedec2(X, w)
    
    X_out = []
    for k in range(len(c)):
        c_ = c[:]
        for k_ in set(range(len(c))).difference([k]):
            c_[k_] = tuple([np.zeros_like(v) for v in c[k_]])
        X_k = pywt.waverec2(c_, w)
        X_out.append(X_k)
    
    return X_out
