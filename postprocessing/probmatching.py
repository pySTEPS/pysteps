''' Methods for probability matching'''

import numpy as np

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