# -*- coding: utf-8 -*-
"""
pysteps.blending.utils
======================

Module with common utilities used by blending methods.

.. autosummary::
    :toctree: ../generated/

    stack_cascades
    normalize_cascade
    blend_optical_flows
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
