# -*- coding: utf-8 -*-
"""
pysteps.blending.steps
======================

Implementation of the STEPS stochastic blending method as described in
:cite:`BPS2006`. The method assumes the presence of one NWP model or ensemble
member to be blended with one nowcast. More models, such as in :cite:`SPN2013`
is possible with this code, but we recommend the use of just two models.

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
    """Calculate explained variance ratios from correlation.

    Parameters
    ----------
    Array of shape [component, scale_level, ...]
      containing correlation (skills) for each component (NWP and nowcast), 
      scale level, and optionally along [y, x] dimensions.

    Returns
    -------
    out : numpy array
      An array containing the ratios of explain variance for each
      component, scale level, ...
    """
    # correlations: [component, scale, ...]
    square_corrs = np.square(correlations)
    # Calculate the ratio of the explained variance to the unexplained 
    # variance of the nowcast and NWP model components
    out = square_corrs / (1 - square_corrs)
    # out: [component, scale, ...]
    return out


def calculate_weights(correlations):
    """Calculate blending weights for STEPS blending from correlation.

    Parameters
    ----------
    correlations : array-like
      Array of shape [component, scale_level, ...]
      containing correlation (skills) for each component (NWP and nowcast), 
      scale level, and optionally along [y, x] dimensions.

    Returns
    -------
    weights : array-like
      Array of shape [component+1, scale_level, ...]
      containing the weights to be used in STEPS blending for
      each original component plus an addtional noise component, scale level, 
      and optionally along [y, x] dimensions.
    """
    # correlations: [component, scale, ...]
    # Calculate weights for each source
    ratios = calculate_ratios(correlations)
    # ratios: [component, scale, ...]
    total_ratios = np.sum(ratios, axis=0)
    # total_ratios: [scale, ...] - the denominator of eq. 11 & 12 in BPS2006 
    weights = correlations * np.sqrt(ratios/total_ratios)
    # weights: [component, scale, ...]
    # Calculate the weight of the noise component. This is different from
    # BPS 2006, as the sum of the weights should equal 1 here.
    noise_weight = 1.0 - np.sum(weights, axis=0)
    # noise_weight: [scale, ...]
    
    # Original BPS2006 method in the following two lines
    # total_square_weights = np.sum(np.square(weights), axis=0)
    # noise_weight = np.sqrt(1.0 - total_square_weights) 
    #TODO: determine the weights method and/or add different functions
    
    # Finally, add the noise_weights to the weights variable. 
    weights = np.concatenate((weights, noise_weight[None, ...]), axis=0)
    return weights


def blend_cascades(cascades_norm, weights):
    """Calculate blended normalized cascades using STEPS weights following eq. 
    10 in :cite:`BPS2006`.
    
    Parameters
    ----------
    cascades_norm : array-like
      Array of shape [number_components + 1, scale_level, ...]
      with normalized cascades components for each component 
      (NWP, nowcasts, noise) and scale level, obtained by calling a method 
      implemented in pysteps.blending.utils.stack_cascades

    weights : array-like
      An array of shape [number_components + 1, scale_level, ...]
      containing the weights to be used in this routine
      for each component plus noise, scale level, and optionally [y, x] 
      dimensions, obtained by calling a method implemented in
      pysteps.blending.steps.calculate_weights

    Returns
    -------
    combined_cascade : array-like
      An array of shape [scale_level, y, x] 
      containing per scale level (cascade) the weighted combination of 
      cascades from multiple components (NWP, nowcasts and noise) to be used 
      in STEPS blending.
    """
    # cascade_norm component, scales, y, x
    # weights component, scales, ....
    all_c_wn = weights * cascades_norm
    combined_cascade = np.sum(all_c_wn, axis=0)
    # combined_cascade [scale, ...]
    return combined_cascade


def blend_means_sigmas(means, sigmas, weights):
    """Calculate the blended means and sigmas, the normalization parameters 
    needed to recompose the cascade. This procedure is similar to the
    blending of the normalized cascades and uses the same weights.
    

    Parameters
    ----------
    means : array-like
      Array of shape [number_components + 1, scale_level, ...]
      with the mean for each component (NWP, nowcasts, noise).
    sigmas : array-like
      Array of shape [number_components + 1, scale_level, ...]
      with the standard deviation for each component.
    weights : array-like
      An array of shape [number_components + 1, scale_level, ...]
      containing the weights to be used in this routine
      for each component plus noise, scale level, and optionally [y, x] 
      dimensions, obtained by calling a method implemented in
      pysteps.blending.steps.calculate_weights

    Returns
    -------
    combined_means : array-like
      An array of shape [scale_level, ...] 
      containing per scale level (cascade) the weighted combination of 
      means from multiple components (NWP, nowcasts and noise).
    combined_sigmas : array-like
      An array of shape [scale_level, ...] 
      similar to combined_means, but containing the standard deviations.

    """
    # Check if the dimensions are the same
    diff_dims = weights.ndim - means.ndim
    if diff_dims:
        for i in range(diff_dims):
            means = np.expand_dims(means, axis=means.ndim)
    diff_dims = weights.ndim - sigmas.ndim
    if diff_dims:
        for i in range(diff_dims):
            sigmas = np.expand_dims(sigmas, axis=sigmas.ndim)
    # Combine (blend) the means and sigmas
    combined_means = np.sum((weights * means), axis=0)
    combined_sigmas = np.sum((weights * sigmas), axis=0)
    #TODO: substract covarainces to weigthed sigmas

    return combined_means, combined_sigmas


def recompose_cascade(combined_cascade, combined_mean, combined_sigma):
    """ Recompose the cascades into a transformed rain rate field. 
    

    Parameters
    ----------
    combined_cascade : array-like
      An array of shape [scale_level, y, x] 
      containing per scale level (cascade) the weighted combination of 
      cascades from multiple components (NWP, nowcasts and noise) to be used 
      in STEPS blending.
    combined_mean : array-like
      An array of shape [scale_level, ...] 
      similar to combined_cascade, but containing the normalization parameter
      mean.
    combined_sigma : array-like
      An array of shape [scale_level, ...] 
      similar to combined_cascade, but containing the normalization parameter
      standard deviation.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """
    renorm = (combined_cascade * combined_sigma) + combined_mean
    # print(renorm.shape)
    out = np.sum(renorm, axis=0)
    # print(out.shape)
    return out
