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

def test_ascending_time_step():
    precip = np.ones((8, 8))
    v = np.ones((8, 8))
    velocity = np.stack([v, v])

    not_ascending_timesteps = [1,2,3,5,4,6,7]
    with pytest.raises(ValueError):
        extrapolate(precip, velocity, not_ascending_timesteps)
