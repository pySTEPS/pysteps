# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.extrapolation.semilagrangian import extrapolate


def test_semilagrangian():
    """Test semilagrangian extrapolation with number of timesteps."""
    # inputs
    precip = np.zeros((8, 8))
    precip[0, 0] = 1
    v = np.ones((8, 8))
    velocity = np.stack([v, v])
    num_timesteps = 1
    # expected
    expected = np.zeros((1, 8, 8))
    expected[:, :, 0] = np.nan
    expected[:, 0, :] = np.nan
    expected[:, 1, 1] = 1
    # result
    result = extrapolate(precip, velocity, num_timesteps)
    assert_array_almost_equal(result, expected)


def test_wrong_input_dimensions():
    p_1d = np.ones(8)
    p_2d = np.ones((8, 8))
    p_3d = np.ones((8, 8, 2))
    v_2d = np.ones((8, 8))
    v_3d = np.stack([v_2d, v_2d])

    num_timesteps = 1

    invalid_inputs = [
        (p_1d, v_3d),
        (p_2d, v_2d),
        (p_3d, v_2d),
        (p_3d, v_3d),
    ]
    for precip, velocity in invalid_inputs:
        with pytest.raises(ValueError):
            extrapolate(precip, velocity, num_timesteps)


def test_ascending_time_step():
    precip = np.ones((8, 8))
    v = np.ones((8, 8))
    velocity = np.stack([v, v])

    not_ascending_timesteps = [1, 2, 3, 5, 4, 6, 7]
    with pytest.raises(ValueError):
        extrapolate(precip, velocity, not_ascending_timesteps)


def test_semilagrangian_timesteps():
    """Test semilagrangian extrapolation with list of timesteps."""
    # inputs
    precip = np.zeros((8, 8))
    precip[0, 0] = 1
    v = np.ones((8, 8)) * 10
    velocity = np.stack([v, v])
    timesteps = [0.1]
    # expected
    expected = np.zeros((1, 8, 8))
    expected[:, :, 0] = np.nan
    expected[:, 0, :] = np.nan
    expected[:, 1, 1] = 1
    # result
    result = extrapolate(precip, velocity, timesteps)
    assert_array_almost_equal(result, expected)
