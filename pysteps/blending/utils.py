# -*- coding: utf-8 -*-
"""
pysteps.blending.utils
======================

Module with common utilities used by blending methods.

.. autosummary::
    :toctree: ../generated/

    stack_cascades
    normalize_cascade
    blend_cascades
    recompose_cascade
    blend_optical_flows
"""

import numpy as np


def stack_cascades(R_d, donorm=True):
    """Stack the given cascades into a larger array.

    Parameters
    ----------
    R_d : dict
      Dictionary containing a list of cascades obtained by calling a method
      implemented in pysteps.cascade.decomposition.
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
        mu_ = np.ones(n_levels) * 0.0
        sigma_ = np.ones(n_levels) * 1.0
        if donorm:
            mu_ = np.asarray(cascade["means"])
            sigma_ = np.asarray(cascade["stds"])
        for j in range(n_levels):
            R__ = (R_i[j, :, :] - mu_[j]) / sigma_[j]
            R_.append(R__)
        R_c.append(np.stack(R_))
        mu_c.append(mu_)
        sigma_c.append(sigma_)
    return np.stack(R_c), np.stack(mu_c), np.stack(sigma_c)


def normalize_cascade(cascade):
    """Normalizes a cascade (again).

    Parameters
    ----------
    cascade : array-like
      An array of shape [scale_level, y, x]
      containing per scale level a cascade that has to be normalized (again).

    Returns
    -------
    out : array-like
        The normalized cascade with the same shape as cascade.

    """
    # Determine the mean and standard dev. of the combined cascade
    mu = np.mean(cascade, axis=(1, 2))
    sigma = np.std(cascade, axis=(1, 2))
    # Normalize the cascade
    out = [(cascade[i] - mu[i]) / sigma[i] for i in range(cascade.shape[0])]
    out = np.stack(out)

    return out


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
    # check inputs
    if isinstance(cascades_norm, (list, tuple)):
        cascades_norm = np.stack(cascades_norm)

    if isinstance(weights, (list, tuple)):
        weights = np.asarray(weights)

    # check weights dimensions match number of sources
    num_sources = cascades_norm.shape[0]
    num_sources_klevels = cascades_norm.shape[1]
    num_weights = weights.shape[0]
    num_weights_klevels = weights.shape[1]

    if num_weights != num_sources:
        raise ValueError(
            "dimension mismatch between cascades and weights.\n"
            "weights dimension must match the number of components in cascades.\n"
            f"number of models={num_sources}, number of weights={num_weights}"
        )
    if num_weights_klevels != num_sources_klevels:
        raise ValueError(
            "dimension mismatch between cascades and weights.\n"
            "weights cascade levels dimension must match the number of cascades in cascades_norm.\n"
            f"number of cascade levels={num_sources_klevels}, number of weights={num_weights_klevels}"
        )

    # cascade_norm component, scales, y, x
    # weights component, scales, ....
    # Reshape weights to make the calculation possible with numpy
    all_c_wn = weights.reshape(num_weights, num_weights_klevels, 1, 1) * cascades_norm
    combined_cascade = np.sum(all_c_wn, axis=0)
    # combined_cascade [scale, ...]
    return combined_cascade


def recompose_cascade(combined_cascade, combined_mean, combined_sigma):
    """Recompose the cascades into a transformed rain rate field.


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
    out: array-like
        A two-dimensional array containing the recomposed cascade.

    Notes
    -----
    The combined_cascade is made with weights that do not have to sum up to
    1.0. Therefore, the combined_cascade is first normalized again using
    normalize_cascade.
    """
    # First, normalize the combined_cascade again
    combined_cascade = normalize_cascade(combined_cascade)
    # Now, renormalize it with the blended sigma and mean values
    renorm = (
        combined_cascade * combined_sigma.reshape(combined_cascade.shape[0], 1, 1)
    ) + combined_mean.reshape(combined_mean.shape[0], 1, 1)
    # print(renorm.shape)
    out = np.sum(renorm, axis=0)
    # print(out.shape)
    return out


def blend_optical_flows(flows, weights):
    """Combine advection fields using given weights. Following :cite:`BPS2006`
    the second level of the cascade is used for the weights

    Parameters
    ----------
    flows : array-like
      A stack of multiple advenction fields having shape
      (S, 2, m, n), where flows[N, :, :, :] contains the motion vectors
      for source N.
      Advection fields for each source can be obtanined by
      calling any of the methods implemented in
      pysteps.motion and then stack all together
    weights : array-like
      An array of shape [number_sources]
      containing the weights to be used to combine
      the advection fields of each source.
      weights are modified to make their sum equal to one.
    Returns
    -------
    out: ndarray_
        Return the blended advection field having shape
        (2, m, n), where out[0, :, :] contains the x-components of
        the blended motion vectors and out[1, :, :] contains the y-components.
        The velocities are in units of pixels / timestep.
    """

    # check inputs
    if isinstance(flows, (list, tuple)):
        flows = np.stack(flows)

    if isinstance(weights, (list, tuple)):
        weights = np.asarray(weights)

    # check weights dimensions match number of sources
    num_sources = flows.shape[0]
    num_weights = weights.shape[0]

    if num_weights != num_sources:
        raise ValueError(
            "dimension mismatch between flows and weights.\n"
            "weights dimension must match the number of flows.\n"
            f"number of flows={num_sources}, number of weights={num_weights}"
        )
    # normalize weigths
    weights = weights / np.sum(weights)

    # flows dimension sources, 2, m, n
    # weights dimension sources
    # move source axis to last to allow broadcasting
    all_c_wn = weights * np.moveaxis(flows, 0, -1)
    # sum uses last axis
    combined_flows = np.sum(all_c_wn, axis=-1)
    # combined_flows [2, m, n]
    return combined_flows
