# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.noise.motion import generate_bps
from pysteps.noise.motion import get_default_params_bps_par
from pysteps.noise.motion import get_default_params_bps_perp
from pysteps.noise.motion import initialize_bps


def test_noise_motion_get_default_params_bps_par():
    """Tests default BPS velocity parameters."""
    expected = (10.88, 0.23, -7.68)
    result = get_default_params_bps_par()
    assert_array_almost_equal(result, expected)


def test_noise_motion_get_default_params_bps_perp():
    """Tests default BPS velocity perturbation."""
    expected = (5.76, 0.31, -2.72)
    result = get_default_params_bps_perp()
    assert_array_almost_equal(result, expected)


vv = np.ones((8, 8)) * np.sqrt(2) * 0.5
test_init_bps_vars = [
    ("vsf", 60),
    ("eps_par", -0.2042896366299448),
    ("eps_perp", 1.6383482042624593),
    ("p_par", (10.88, 0.23, -7.68)),
    ("p_perp", (5.76, 0.31, -2.72)),
    ("V_par", np.stack([vv, vv])),
    ("V_perp", np.stack([-vv, vv])),
]


@pytest.mark.parametrize("variable, expected", test_init_bps_vars)
def test_initialize_bps(variable, expected):
    """Tests initialation BPS velocity perturbation method."""
    seed = 42
    timestep = 1
    pixelsperkm = 1
    v = np.ones((8, 8))
    velocity = np.stack([v, v])
    perturbator = initialize_bps(velocity, pixelsperkm, timestep, seed=seed)
    assert_array_almost_equal(perturbator[variable], expected)


def test_generate_bps():
    """Tests generation BPS velocity perturbation method."""
    seed = 42
    timestep = 1
    pixelsperkm = 1
    v = np.ones((8, 8))
    velocity = np.stack([v, v])
    perturbator = initialize_bps(velocity, pixelsperkm, timestep, seed=seed)
    new_vv = generate_bps(perturbator, timestep)
    expected = np.stack([v * -0.066401, v * 0.050992])
    assert_array_almost_equal(new_vv, expected)
