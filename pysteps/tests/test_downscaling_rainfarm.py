# -*- coding: utf-8 -*-

import pytest

from pysteps import downscaling
from pysteps.tests.helpers import get_precipitation_fields

# load and preprocess input field
PRECIP = get_precipitation_fields(
    source="bom",
    filled=True,
    convert_to="mm/h",
)

rainfarm_arg_names = ("alpha", "ds_factor", "threshold", "return_alpha")


rainfarm_arg_values = [(1.0, 1, 0, False), (1, 2, 0, False), (1, 4, 0, False)]


@pytest.mark.parametrize(rainfarm_arg_names, rainfarm_arg_values)
def test_rainfarm_shape(alpha, ds_factor, threshold, return_alpha):
    """Test that the output of rainfarm is consistent with the downscaling factor."""

    precip_lr = PRECIP.coarsen(x=ds_factor, y=ds_factor).mean()

    rainfarm = downscaling.get_method("rainfarm")

    precip_hr = rainfarm(precip_lr, alpha, ds_factor, threshold, return_alpha)

    assert precip_hr.ndim == PRECIP.ndim
    assert precip_hr.shape[0] == PRECIP.shape[0]
    assert precip_hr.shape[1] == PRECIP.shape[1]


rainfarm_arg_values = [(1.0, 2, 0, True), (None, 2, 0, True)]


@pytest.mark.parametrize(rainfarm_arg_names, rainfarm_arg_values)
def test_rainfarm_alpha(alpha, ds_factor, threshold, return_alpha):
    """Test that rainfarm computes and returns alpha."""

    precip_lr = PRECIP.coarsen(x=ds_factor, y=ds_factor).mean()
    rainfarm = downscaling.get_method("rainfarm")

    precip_hr = rainfarm(precip_lr, alpha, ds_factor, threshold, return_alpha)

    assert len(precip_hr) == 2
    if alpha is None:
        assert not precip_hr[1] == alpha
    else:
        assert precip_hr[1] == alpha
