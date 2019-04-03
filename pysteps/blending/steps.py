"""
pysteps.blending.steps
======================

Implementation of the STEPS stochastic blending method as described in  
:cite:`BPS2006`.

.. autosummary::
    :toctree: ../generated/
    
    calculate_ratios
    calculate_weights
    blender

"""

import numpy as np


def calculate_ratios(correlations):
    """Calculate explain variance ratios from correlation.
    
    Parameters
    ----------
    Array of shape [component, scale_level, ...]
      containing correlation (skills) for each component, scale level,
      and optionally along [y, x] dimensions.

    Returns
    -------
    out : numpy array
      An array containing the ratios of explain variance for each
      component, scale level, ...
    """
    # correlations: [component, scale, ...]
    square_corrs = np.square(correlations)
    out = square_corrs / (1 - square_corrs)
    # out: [component, scale, ...]
    return out


def calculate_weights(correlations):
    """Calculate blending weights for STEPS blending from correlation.
    
    Parameters
    ----------
    correlations : array-like
      Array of shape [component, scale_level, ...]
      containing correlation (skills) for each component, scale level,
      and optionally along [y, x] dimensions.
    
    Returns
    -------
    weights : array-like
      Array of shape [component+1, scale_level, ...]
      containing the weights to be used in STEPS blending for
      each original component plus noise, scale level, ...
    """
    # correlations: [component, scale, ...]
    # calculate weights for each source     
    ratios = calculate_ratios(correlations)
    # ratios: [component, scale, ...]
    total_ratios = np.sum(ratios, axis=0)
    # total_ratios: [scale, ...]
    weights = correlations * np.sqrt(ratios/total_ratios)
    # weights: [component, scale, ...]
    # calculate noise weights
    total_square_weights = np.sum(np.square(weights),axis=0)
    # total_square_weights: [scale,...]
    noise_weight = np.sqrt(1.0 - total_square_weights)
    weights = np.concatenate((weights, noise_weight[None,...]), axis=0)
    return weights


def blend_cascades(cascades_norm,weights):
    """Calculate blended cascades using STEPS weights.
    
    Parameters
    ----------
    cascades_norm : array-like
      Array of shape [number_components + 1, scale_level, ...]
      with normalized cascades components for each component, scale level,
      and optionally [y, x ] dimensions,
      obtained by calling a method implemented in
      pysteps.blending.utils.stack_cascades

    weights : array-like
      An array of shape [number_components + 1, scale_level, ...] 
      containing the weights to be used in this rutine
      for each component plus noise, scale level, ...
      obtained by calling a method implemented in
      pysteps.blending.steps.calculate_weights
    
    Returns
    -------
    combined_cascade : array-like
      An array of shape [scale_level, ...] 
      containing the weighted combination of cascades from multiple 
      componented to be used in STEPS blending.
    """

    #cascade_norm component, scales, ....
    #weights component, scales, ....
    all_c_wn = weights * cascades_norm
    combined_cascade = np.sum(all_c_wn,axis=0)
    print(combined_cascade.shape)
    # combined_cascade [scale, ...]
    return combined_cascade



