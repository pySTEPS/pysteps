''' Methods for probability matching'''

import numpy as np
from scipy import interpolate as sip

def compute_empirical_cdf(bin_edges, hist):
    cdf = []
    xs = 0.0
    
    for x,h in zip(zip(bin_edges[:-1], bin_edges[1:]), hist):
        cdf.append(xs)
        xs += (x[1] - x[0]) * h
    
    cdf.append(xs)
    cdf = np.array(cdf) / xs
    
    return cdf

def nonparam_match_empirical_cdf(R, R_trg):
    """Matches the empirical CDF of the initial array with the empirical CDF
    of a target array. 
    Initial ranks are conserved, but empirical distribution matches the target one.
    Zero-pixels in initial array are conserved.
    
    Parameters: 
    ----------
    R : array-like
        The initial array whose CDF is to be changed.
    R_trg : 
        The target array whose CDF is to be matched.
        
    Returns
    -------
    outputarray : array-like
        The new array. 
    """
    if R.size != R_trg.size:
        raise ValueError("the input arrays must have the same size")
    if np.any(~np.isfinite(R)) or np.any(~np.isfinite(R_trg)):
      raise ValueError("input contains non-finite values")
      
    
    # zeros in initial image
    zvalue = R.min()
    idxzeros = R == zvalue
    
    # zeros in target image
    zvalue_trg = R_trg.min()
    idxzeros_trg = R_trg == zvalue
    
    if np.sum(R_trg > zvalue_trg) > np.sum(R > zvalue):
        # adjust the fraction of rain in target distribution if the number of zeros
        # is greater than in the initial array
        # TODO: this needs more testing
        war = np.sum(R > zvalue)/R.size
        p = np.percentile(R_trg, 100*(1 - war))     
        R_trg[R_trg < p] = zvalue_trg
        
    # flatten the arrays
    arrayshape = R.shape
    R_trg = R_trg.flatten()
    R = R.flatten()
    
    # rank target values
    order = R_trg.argsort()
    ranked = R_trg[order]

    # rank initial values order
    orderin = R.argsort()
    ranks = np.empty(len(R), int)
    ranks[orderin] = np.arange(len(R))

    # get ranked values from target and rearrange with inital order
    R = ranked[ranks]

    # reshape as original array
    R = R.reshape(arrayshape)
    
    # readding original zeros
    R[idxzeros] = zvalue_trg
    
    return R

def nonparam_match_empirical_cdf_masked():
    pass

def pmm_init(bin_edges_1, cdf_1, bin_edges_2, cdf_2):
    pmm = {}
    
    pmm["bin_edges_1"]      = bin_edges_1.copy()
    pmm["cdf_1"]            = cdf_1.copy()
    pmm["bin_edges_2"]      = bin_edges_2.copy()
    pmm["cdf_2"]            = cdf_2.copy()
    pmm["cdf_interpolator"] = sip.interp1d(bin_edges_1, cdf_1, kind="linear")
    
    return pmm

def pmm_compute(pmm, x):
    mask = np.logical_and(x >= pmm["bin_edges_1"][0], x <= pmm["bin_edges_1"][-1])
    p = pmm["cdf_interpolator"](x[mask])
    
    result = np.ones(len(mask)) * np.nan
    result[mask] = _invfunc(p, pmm["bin_edges_2"], pmm["cdf_2"])
    
    return result

def _invfunc(y, fx, fy):
  if len(y) == 0:
      return np.array([])
  
  b = np.digitize(y, fy)
  mask = np.logical_and(b > 0, b < len(fy))
  c = (y[mask] - fy[b[mask]-1]) / (fy[b[mask]] - fy[b[mask]-1])
  
  result = np.ones(len(y)) * np.nan
  result[mask] = c * fx[b[mask]] + (1.0-c) * fx[b[mask]-1]
  
  return result
