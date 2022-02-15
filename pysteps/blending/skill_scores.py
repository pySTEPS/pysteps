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
from pysteps.blending import clim


def spatial_correlation(obs, mod, domain_mask):
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
    domain_mask : array-like
        Boolean array of shape [y, x] indicating which cells fall outside the
        radar domain.

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
        # Only calculate the skill for the pixels within the radar domain
        # (as that is where there are observations)
        obs_cascade_level = obs[cascade_level, :, :]
        mod_cascade_level = mod[cascade_level, :, :]
        obs_cascade_level[domain_mask] = np.nan
        mod_cascade_level[domain_mask] = np.nan

        # Flatten both arrays
        obs_1d = obs_cascade_level.flatten()
        mod_1d = mod_cascade_level.flatten()
        # Calculate the correlation between the two
        cov = np.nansum(
            (mod_1d - np.nanmean(mod_1d)) * (obs_1d - np.nanmean(obs_1d))
        )  # Without 1/n, as this cancels out (same for stdx and -y)
        std_obs = np.sqrt(np.nansum((obs_1d - np.nanmean(obs_1d)) ** 2.0))
        std_mod = np.sqrt(np.nansum((mod_1d - np.nanmean(mod_1d)) ** 2.0))
        rho.append(cov / (std_mod * std_obs))

    # Make sure rho is always a (finite) number
    rho = np.nan_to_num(rho, copy=True, nan=10e-5, posinf=10e-5, neginf=10e-5)

    return rho


def lt_dependent_cor_nwp(lt, correlations, outdir_path, n_model=0, skill_kwargs=None):
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
    outdir_path: string
      Path to folder where the historical skill are stored. Defaults to
      path_workdir from rcparams.
    n_model: int, optional
        The index number of the (NWP) model when the climatological skill of
        multiple (NWP) models is stored. For calculations with one model, or
        when n_model is not provided, n_model = 0.
    skill_kwargs : dict, optional
        Dictionary containing e.g. the outdir_path, nmodels and window_length
        parameters.

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

    if skill_kwargs is None:
        skill_kwargs = dict()

    # Obtain the climatological values towards which the correlations will
    # regress
    clim_cor_values, regr_pars = clim_regr_values(
        n_cascade_levels=len(correlations),
        outdir_path=outdir_path,
        n_model=n_model,
        skill_kwargs=skill_kwargs,
    )
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


def clim_regr_values(n_cascade_levels, outdir_path, n_model=0, skill_kwargs=None):
    """Obtains the climatological correlation values and regression parameters
    from a file called NWP_weights_window.bin in the outdir_path. If this file
    is not present yet, the values from :cite:`BPS2004` are used.


    Parameters
    ----------
    n_cascade_levels : int
        The number of cascade levels to use.
    outdir_path: string
      Path to folder where the historical skill are stored. Defaults to
      path_workdir from rcparams.
    n_model: int, optional
        The index number of the (NWP) model when the climatological skill of
        multiple (NWP) models is stored. For calculations with one model, or
        when n_model is not provided, n_model = 0.
    skill_kwargs : dict, optional
        Dictionary containing e.g. the outdir_path, nmodels and window_length
        parameters.

    Returns
    -------
    clim_cor_values : array-like
        Array of shape [n_cascade_levels] containing the
        climatological values of the lag 1 and lag 2 auto-correlation
        coefficients, obtained by calling a method implemented in
        pysteps.blending.skill_scores.get_clim_skill_scores.
    regr_pars : array-like
        Array of shape [2, n_cascade_levels] containing the regression
        parameters. These are fixed values that should be hard-coded in this
        function. Unless changed by the user, the standard values from
        `BPS2004` are used.

    Notes
    -----
    The literature climatological values assume 8 cascade levels. In case
    less or more cascade levels are used, the script will handle this by taking
    the first n values or extending the array with a small value. This is not
    ideal, but will be fixed once the clim_regr_file is made. Hence, this
    requires a warm-up period of the model.
    In addition, the regression parameter values (eq. 24 in BPS2004) are hard-
    coded and can only be optimized by the user after (re)fitting of the
    equation.

    """

    if skill_kwargs is None:
        skill_kwargs = dict()

    # First, obtain climatological skill values
    try:
        clim_cor_values = clim.calc_clim_skill(
            outdir_path=outdir_path, n_cascade_levels=n_cascade_levels, **skill_kwargs
        )
    except FileNotFoundError:
        # The climatological skill values file does not exist yet, so we'll
        # use the default values from BPS2004.
        clim_cor_values = np.array(
            [[0.848, 0.537, 0.237, 0.065, 0.020, 0.0044, 0.0052, 0.0040]]
        )

    clim_cor_values = clim_cor_values[n_model, :]

    # Check if clim_cor_values has only one model, otherwise it has
    # returned the skill values for multiple models
    if clim_cor_values.ndim != 1:
        raise IndexError(
            "The climatological cor. values of multiple models were returned, but only one model should be specified. Make sure to pass the argument nmodels in the function"
        )

    # Also check whether the number of cascade_levels is correct
    if clim_cor_values.shape[0] > n_cascade_levels:
        clim_cor_values = clim_cor_values[0:n_cascade_levels]
    elif clim_cor_values.shape[0] < n_cascade_levels:
        # Get the number of cascade levels that is missing
        n_extra_lev = n_cascade_levels - clim_cor_values.shape[0]
        # Append the array with correlation values of 10e-4
        clim_cor_values = np.append(clim_cor_values, np.repeat(1e-4, n_extra_lev))

    # Get the regression parameters (L in eq. 24 in BPS2004) - Hard coded,
    # change to own values when present.
    regr_pars = np.array(
        [
            [130.0, 165.0, 120.0, 55.0, 50.0, 15.0, 15.0, 10.0],
            [155.0, 220.0, 200.0, 75.0, 10e4, 10e4, 10e4, 10e4],
        ]
    )

    # Check whether the number of cascade_levels is correct
    if regr_pars.shape[1] > n_cascade_levels:
        regr_pars = regr_pars[:, 0:n_cascade_levels]
    elif regr_pars.shape[1] < n_cascade_levels:
        # Get the number of cascade levels that is missing
        n_extra_lev = n_cascade_levels - regr_pars.shape[1]
        # Append the array with correlation values of 10e-4
        regr_pars = np.append(
            regr_pars,
            [np.repeat(10.0, n_extra_lev), np.repeat(10e4, n_extra_lev)],
            axis=1,
        )

    return clim_cor_values, regr_pars
