# -*- coding: utf-8 -*-
import numpy as np
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
