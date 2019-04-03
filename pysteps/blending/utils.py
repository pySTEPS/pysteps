"""
pysteps.blending.utils
======================

Module with common utilities used by blending methods.

.. autosummary::
    :toctree: ../generated/

    stack_cascades
    recompose_cascade
"""

import numpy as np


def stack_cascades(R_d, donorm=True):
    """Stack the given cascades into a larger array.
    
    Parameters
    ----------
    R_d : list
      List of cascades obtained by calling a method implemented in
      pysteps.cascade.decomposition.
    donorm : bool
      If True, normalize the cascade levels before stacking.
    
    Returns
    -------
    out : tuple
      A three-element tuple containing a four-dimensional array of stacked
      cascade levels and arrays of mean values and standard deviations for each
      cascade level.
    """
    R_c = []
    mu_c = []
    sigma_c = []
   
    for cascade in R_d:
        R_ = []
        R_i = cascade["cascade_levels"]
        n_levels = R_i.shape[0]
        mu_ = np.ones(n_levels)* 0.0
        sigma_ = np.ones(n_levels) *1.0
        if donorm:
            mu_ = np.asarray(cascade["means"])
            sigma_ = np.asarray(cascade["stds"])
        for j in range(n_levels):
            R__ = (R_i[j, :, :] - mu_[j]) / sigma_[j]
            R_.append(R__)
        R_c.append(np.stack(R_))
        mu_c.append(mu_)
        sigma_c.append(sigma_)    
    return np.stack(R_c),np.stack(mu_c),np.stack(sigma_c)
