# -*- coding: utf-8 -*-
"""
pysteps.blending.skill_scores
==============================

Methods for computing skill scores, needed for the blending weights, of two-
dimensional model fields with the latest observation field.

.. autosummary::
    :toctree: ../generated/

    spatial_correlation
"""

import numpy as np


def spatial_correlation(obs, mod):
    """Determine the spatial correlation between the cascade of the latest 
    available observed (radar) rainfall field and a time-synchronous cascade 
    derived from a model (generally NWP) field. Both fields are assumed to use 
    the same grid.
    

    Parameters
    ----------
    obs : array-like
        Array of shape [cascade_level, y, x] with per cascade_level the
        normalized cascade of the observed (radar) rainfall field.
    mod : array-like
        Array of shape [cascade_level, y, x] with per cascade_level the
        normalized cascade of the model field.

    Returns
    -------
    rho : array-like
        Array of shape [n_cascade_levels] containing per cascade_level the
        correlation between the normalized cascade of the observed (radar) 
        rainfall field and the normalized cascade of the model field.
        
    References
    ----------
    :cite:`BPS2006`
    :cite:`SPN2013`

    """
    rho = []
    # Fill rho per cascade level, so loop through the cascade levels
    for cascade_level in range(0,obs.shape[0]):
        # Flatten both arrays
        obs_1d = obs[cascade_level, :, :].flatten()
        mod_1d = mod[cascade_level, :, :].flatten()
        # Calculate the correlation between the two
        cov = np.sum((mod_1d - np.mean(mod_1d))*(obs_1d - np.mean(obs_1d)))  # Without 1/n, as this cancels out (same for stdx and -y)
        std_obs = np.sqrt(np.sum((obs_1d - np.mean(obs_1d))**2.0))
        std_mod = np.sqrt(np.sum((mod_1d - np.mean(mod_1d))**2.0))
        rho.append(cov / (std_mod * std_obs))
    
    return rho

