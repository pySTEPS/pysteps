# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.blending.skill_scores import (
    spatial_correlation,
    lt_dependent_cor_nwp,
    lt_dependent_cor_extrapolation,
    clim_regr_values,
)


# Set the climatological correlations values
clim_cor_values_8lev = np.array(
    [0.848, 0.537, 0.237, 0.065, 0.020, 0.0044, 0.0052, 0.0040]
)
clim_cor_values_6lev = np.array([0.848, 0.537, 0.237, 0.065, 0.020, 0.0044])
clim_cor_values_9lev = np.array(
    [0.848, 0.537, 0.237, 0.065, 0.020, 0.0044, 0.0052, 0.0040, 1e-4]
)

# Set the regression values
regr_pars_8lev = np.array(
    [
        [130.0, 165.0, 120.0, 55.0, 50.0, 15.0, 15.0, 10.0],
        [155.0, 220.0, 200.0, 75.0, 10e4, 10e4, 10e4, 10e4],
    ]
)
regr_pars_6lev = np.array(
    [
        [130.0, 165.0, 120.0, 55.0, 50.0, 15.0],
        [155.0, 220.0, 200.0, 75.0, 10e4, 10e4],
    ]
)
regr_pars_9lev = np.array(
    [
        [130.0, 165.0, 120.0, 55.0, 50.0, 15.0, 15.0, 10.0, 10.0],
        [155.0, 220.0, 200.0, 75.0, 10e4, 10e4, 10e4, 10e4, 10e4],
    ]
)

# Set the dummy observation and model values
dummy_2d_array = np.array([[1.0, 2.0], [3.0, 4.0]])
obs_8lev = np.repeat(dummy_2d_array[None, :, :], 8, axis=0)
obs_6lev = np.repeat(dummy_2d_array[None, :, :], 6, axis=0)
obs_9lev = np.repeat(dummy_2d_array[None, :, :], 9, axis=0)
mod_8lev = np.repeat(dummy_2d_array[None, :, :], 8, axis=0)
mod_6lev = np.repeat(dummy_2d_array[None, :, :], 6, axis=0)
mod_9lev = np.repeat(dummy_2d_array[None, :, :], 9, axis=0)

# Gives some dummy values to PHI
dummy_phi = np.array([0.472650, 0.523825, 0.103454])
PHI_8lev = np.repeat(dummy_phi[None, :], 8, axis=0)
PHI_6lev = np.repeat(dummy_phi[None, :], 6, axis=0)
PHI_9lev = np.repeat(dummy_phi[None, :], 9, axis=0)

# Test function arguments
skill_scores_arg_names = (
    "obs",
    "mod",
    "lt",
    "PHI",
    "cor_prev",
    "clim_cor_values",
    "regr_pars",
    "n_cascade_levels",
    "expected_cor_t0",
    "expected_cor_nwp_lt",
    "expected_cor_nowcast_lt",
)

# Test function values
skill_scores_arg_values = [
    (
        obs_8lev,
        mod_8lev,
        60,
        PHI_8lev,
        None,
        clim_cor_values_8lev,
        regr_pars_8lev,
        8,
        np.repeat(1.0, 8),
        np.array(
            [
                0.97455941,
                0.9356775,
                0.81972779,
                0.55202975,
                0.31534738,
                0.02264599,
                0.02343133,
                0.00647032,
            ]
        ),
        np.array(
            [
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
            ]
        ),
    ),
    (
        obs_6lev,
        mod_6lev,
        60,
        PHI_6lev,
        None,
        clim_cor_values_6lev,
        regr_pars_6lev,
        6,
        np.repeat(1.0, 6),
        np.array(
            [0.97455941, 0.9356775, 0.81972779, 0.55202975, 0.31534738, 0.02264599]
        ),
        np.array([0.996475, 0.996475, 0.996475, 0.996475, 0.996475, 0.996475]),
    ),
    (
        obs_9lev,
        mod_9lev,
        60,
        PHI_9lev,
        None,
        clim_cor_values_9lev,
        regr_pars_9lev,
        9,
        np.repeat(1.0, 9),
        np.array(
            [
                0.97455941,
                0.9356775,
                0.81972779,
                0.55202975,
                0.31534738,
                0.02264599,
                0.02343133,
                0.00647032,
                0.00347776,
            ]
        ),
        np.array(
            [
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
            ]
        ),
    ),
    (
        obs_8lev,
        mod_8lev,
        0,
        PHI_8lev,
        None,
        clim_cor_values_8lev,
        regr_pars_8lev,
        8,
        np.repeat(1.0, 8),
        np.repeat(1.0, 8),
        np.array(
            [
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
                0.996475,
            ]
        ),
    ),
]


# The test
@pytest.mark.parametrize(skill_scores_arg_names, skill_scores_arg_values)

# The test function to be used
def test_blending_skill_scores(
    obs,
    mod,
    lt,
    PHI,
    cor_prev,
    clim_cor_values,
    regr_pars,
    n_cascade_levels,
    expected_cor_t0,
    expected_cor_nwp_lt,
    expected_cor_nowcast_lt,
):
    """Tests if the skill_score functions behave correctly. A dummy gridded
    model and observation field should be given for n_cascade_levels, which
    leads to a given spatial correlation per cascade level. Then, the function
    tests if the correlation regresses towards the climatological values given
    lead time lt for the NWP fields or given the PHI-values for the
    extrapolation field.

    """
    domain_mask = np.full(obs[0, :, :].shape, False, dtype=bool)

    # Calculate the spatial correlation of the given model field
    correlations_t0 = np.array(spatial_correlation(obs, mod, domain_mask))

    # Check if the field has the same number of cascade levels as the model
    # field and as the given n_cascade_levels
    assert (
        correlations_t0.shape[0] == mod.shape[0]
    ), "Number of cascade levels should be the same as in the model field"
    assert (
        correlations_t0.shape[0] == n_cascade_levels
    ), "Number of cascade levels should be the same as n_cascade_levels"

    # Check if the returned values are as expected
    assert_array_almost_equal(
        correlations_t0,
        expected_cor_t0,
        decimal=3,
        err_msg="Returned spatial correlation is not the same as the expected value",
    )

    # Test if the NWP correlation regresses towards the correct value given
    # a lead time in minutes
    # First, check if the climatological values are returned correctly
    correlations_clim, regr_clim = clim_regr_values(
        n_cascade_levels=n_cascade_levels, outdir_path="./tmp/"
    )
    assert (
        correlations_clim.shape[0] == n_cascade_levels
    ), "Number of cascade levels should be the same as n_cascade_levels"
    assert_array_almost_equal(
        correlations_clim,
        clim_cor_values,
        decimal=3,
        err_msg="Not the correct climatological correlations were returned",
    )
    assert_array_almost_equal(
        regr_clim,
        regr_pars,
        decimal=3,
        err_msg="Not the correct regression parameters were returned",
    )

    # Then, check the regression of the correlation values
    correlations_nwp_lt = lt_dependent_cor_nwp(
        lt=lt, correlations=correlations_t0, outdir_path="./tmp/"
    )
    assert (
        correlations_nwp_lt.shape[0] == mod.shape[0]
    ), "Number of cascade levels should be the same as in the model field"
    assert (
        correlations_nwp_lt.shape[0] == n_cascade_levels
    ), "Number of cascade levels should be the same as n_cascade_levels"
    assert_array_almost_equal(
        correlations_nwp_lt,
        expected_cor_nwp_lt,
        decimal=3,
        err_msg="Correlations of NWP not equal to the expected correlations",
    )

    # Finally, make sure nowcast correlation regresses towards the correct
    # value given some PHI-values.
    correlations_nowcast_lt, __ = lt_dependent_cor_extrapolation(
        PHI, correlations_t0, cor_prev
    )

    print(correlations_nowcast_lt)
    assert (
        correlations_nowcast_lt.shape[0] == mod.shape[0]
    ), "Number of cascade levels should be the same as in the model field"
    assert (
        correlations_nowcast_lt.shape[0] == n_cascade_levels
    ), "Number of cascade levels should be the same as n_cascade_levels"
    assert_array_almost_equal(
        correlations_nowcast_lt,
        expected_cor_nowcast_lt,
        decimal=3,
        err_msg="Correlations of nowcast not equal to the expected correlations",
    )
