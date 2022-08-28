# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_almost_equal

from pysteps.extrapolation.semilagrangian import extrapolate


def test_semilagrangian():
    """Tests semilagrangian extrapolation."""
    # inputs
    precip = np.ones((8, 8))
    v = np.ones((8, 8))
    velocity = np.stack([v, v])
    num_timesteps = 1
    # expected
    expected = np.ones((1, 8, 8))
    expected[:, :, 0] = np.nan
    expected[:, 0, :] = np.nan
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
