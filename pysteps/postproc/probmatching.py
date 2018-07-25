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

def nonparam_match_empirical_cdf(initialarray, targetarray):
    """Matches the empirical CDF of the initial array with the empirical CDF
    of a target array. 
    Initial ranks are conserved, but empirical distribution matches the target one.
    Zero-pixels in initial array are conserved, too.
    
    Parameters: 
    ----------
    initialarray : array-like
        The initial array whose CDF is to be changed.
    targetarray : 
        The target array whose CDF is to be matched.
        
    Returns
    -------
    outputarray : array-like
        The new array. 
    """
    if initialarray.size != targetarray.size:
        raise ValueError("the input arrays must have the same size")
    if np.any(~np.isfinite(initialarray)) or np.any(~np.isfinite(targetarray)):
      raise ValueError("input contains non-finite values")
    
    # zeros in initial image
    zvalue = initialarray.min()
    idxzeros = initialarray == zvalue

    # flatten the arrays
    arrayshape = initialarray.shape
    target = targetarray.flatten()
    array = initialarray.flatten()
    
    # rank target values
    order = target.argsort()
    ranked = target[order]

    # rank initial values order
    orderin = array.argsort()
    ranks = np.empty(len(array), int)
    ranks[orderin] = np.arange(len(array))

    # get ranked values from target and rearrange with inital order
    outputarray = ranked[ranks]

    # reshape as original array
    outputarray = outputarray.reshape(arrayshape)
    
    # readding original zeros
    outputarray[idxzeros] = zvalue

    return outputarray

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
