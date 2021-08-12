# -*- coding: utf-8 -*-
"""
pysteps.blending.skill_scores
==============================

Methods for computing skill scores, needed for the blending weights, of two-
dimensional model fields with the latest observation field.

.. autosummary::
    :toctree: ../generated/

    spatial_correlation
    lt_dependent_cor_nwp
    lt_dependent_cor_extrapolation
    clim_regr_values
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
    for cascade_level in range(0, obs.shape[0]):
        # Flatten both arrays
        obs_1d = obs[cascade_level, :, :].flatten()
        mod_1d = mod[cascade_level, :, :].flatten()
        # Calculate the correlation between the two
        cov = np.sum(
            (mod_1d - np.mean(mod_1d)) * (obs_1d - np.mean(obs_1d))
        )  # Without 1/n, as this cancels out (same for stdx and -y)
        std_obs = np.sqrt(np.sum((obs_1d - np.mean(obs_1d)) ** 2.0))
        std_mod = np.sqrt(np.sum((mod_1d - np.mean(mod_1d)) ** 2.0))
        rho.append(cov / (std_mod * std_obs))

    return rho


def lt_dependent_cor_nwp(lt, correlations, clim_regr_file=None):
    """Determine the correlation of a model field for lead time lt and
    cascade k, by assuming that the correlation determined at t=0 regresses
    towards the climatological values.


    Parameters
    ----------
    lt : int
        The lead time of the forecast in minutes.
    correlations : array-like
        Array of shape [n_cascade_levels] containing per cascade_level the
        correlation between the normalized cascade of the observed (radar)
        rainfall field and the normalized cascade of the model field.
    clim_regr_file : str, optional
        The location of the file with the climatological correlation values
        and regression parameters.

    Returns
    -------
    rho : array-like
        Array of shape [n_cascade_levels] containing, for lead time lt, per
        cascade_level the correlation between the normalized cascade of the
        observed (radar) rainfall field and the normalized cascade of the
        model field.

    References
    ----------
    :cite:`BPS2004`
    :cite:`BPS2006`
    """
    # Obtain the climatological values towards which the correlations will
    # regress
    clim_cor_values, regr_pars = clim_regr_values(len(correlations), clim_regr_file)
    # Determine the speed of the regression (eq. 24 in BPS2004)
    qm = np.exp(-lt / regr_pars[0, :]) * (2 - np.exp(-lt / regr_pars[1, :]))
    # Determine the correlation for lead time lt
    rho = qm * correlations + (1 - qm) * clim_cor_values

    return rho


def lt_dependent_cor_extrapolation(PHI, correlations=None, correlations_prev=None):
    """Determine the correlation of the extrapolation (nowcast) component for
    lead time lt and cascade k, by assuming that the correlation determined at
    t=0 regresses towards the climatological values.


    Parameters
    ----------
    PHI : array-like
        Array of shape [n_cascade_levels, ar_order + 1] containing per
        cascade level the autoregression parameters.
    correlations : array-like, optional
        Array of shape [n_cascade_levels] containing per cascade_level the
        latest available correlation from the extrapolation component that can
        be found from the AR-2 model.
    correlations_prev : array-like, optional
        Similar to correlations, but from the timestep before that.

    Returns
    -------
    rho : array-like
        Array of shape [n_cascade_levels] containing, for lead time lt, per
        cascade_level the correlation of the extrapolation component.

    References
    ----------
    :cite:`BPS2004`
    :cite:`BPS2006`

    """
    # Check if correlations_prev exists, if not, we set it to 1.0
    if correlations_prev is None:
        correlations_prev = np.repeat(1.0, PHI.shape[0])
    # Same for correlations at first time step, we set it to
    # phi1 / (1 - phi2), see BPS2004
    if correlations is None:
        correlations = PHI[:, 0] / (1.0 - PHI[:, 1])

    # Calculate the correlation for lead time lt
    rho = PHI[:, 0] * correlations + PHI[:, 1] * correlations_prev

    # Finally, set the current correlations array as the previous one for the
    # next time step
    rho_prev = correlations

    return rho, rho_prev


# TODO: Make sure the initial values also work for n_cascade_levels != 8.
def clim_regr_values(n_cascade_levels, clim_regr_file=None):
    """Obtains the climatological correlation values and regression parameters
    from a file called ... in ... If this file is not present yet, the values
    from :cite:`BPS2004`.


    Parameters
    ----------
    n_cascade_levels : int
        The number of cascade levels to use.
    clim_regr_file : str, optional
        Location of the file with the climatological correlation values
        and regression parameters.

    Returns
    -------
    clim_cor_values : array-like
        Array of shape [n_cascade_levels] containing the
        climatological values of the lag 1 and lag 2 auto-correlation
        coefficients, obtained by calling a method implemented in
        pysteps.blending.skill_scores.get_clim_skill_scores.
    regr_pars : array-like
        Array of shape [2, n_cascade_levels] containing the regression
        parameters for the auto-correlation coefficients, obtained by calling
        a method implemented in
        pysteps.blending.skill_scores.get_clim_skill_scores.

    """
    if clim_regr_file is not None:
        # TODO: Finalize this function when the I/O skill file system has been
        # set up (work in progress by Lesley).

        # 1. Open the file

        # 2. Get the climatological correlation values and the regression
        # parameters
        clim_cor_values = ...
        regr_pars = ...

    else:
        # Get the values from BPS2004
        clim_cor_values = np.array(
            [0.848, 0.537, 0.237, 0.065, 0.020, 0.0044, 0.0052, 0.0040]
        )
        regr_pars = np.array(
            [
                [130.0, 165.0, 120.0, 55.0, 50.0, 15.0, 15.0, 10.0],
                [155.0, 220.0, 200.0, 75.0, 10e4, 10e4, 10e4, 10e4],
            ]
        )

    return clim_cor_values, regr_pars
