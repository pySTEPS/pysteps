"""
pysteps.blending.steps
======================

Implementation of the STEPS stochastic blending method as described in  
:cite:`BPS2006`.

.. autosummary::
    :toctree: ../generated/
    
    calculate_ratios
    calculate_weights
    blend_cascades
    blend_means_sigmas
    recompose_cascade

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


def blend_means_sigmas(means,sigmas,weights):
    #means component, scales, ....
    #sigmas component, scales, ....
    #weights component, scales, ....
    total_weights = np.sum(weights,axis=0)
    factors = weights / total_weights
    diff_dims = factors.ndim - means.ndim
    if diff_dims:
        for i in range(diff_dims):
            means = np.expand_dims(means, axis=means.ndim)
    diff_dims = factors.ndim - sigmas.ndim
    if diff_dims:
        for i in range(diff_dims):
            sigmas = np.expand_dims(sigmas, axis=sigmas.ndim)
    weighted_means = np.sum((factors * means),axis=0)
    weighted_sigmas = np.sum((factors * sigmas),axis=0)

    # to do: substract covarainces to weigthed sigmas
  
    return weighted_means, weighted_sigmas

    
def recompose_cascade(w_cascade, w_mean,w_sigma):
    renorm = (w_cascade * w_sigma) + w_mean
    # print(renorm.shape)
    out = np.sum(renorm,axis=0)
    # print(out.shape)
    return out




