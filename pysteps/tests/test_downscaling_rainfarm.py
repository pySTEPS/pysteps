# -*- coding: utf-8 -*-

import pytest

from pysteps import downscaling
from pysteps.tests.helpers import get_precipitation_fields
from pysteps.utils import aggregate_fields_space, square_domain


# load and preprocess input field
precip, metadata = get_precipitation_fields(
    num_prev_files=0, num_next_files=0, return_raw=False, metadata=True
)
precip = precip.filled()
precip, metadata = square_domain(precip, metadata, "crop")


rainfarm_arg_names = ("alpha", "ds_factor", "threshold", "return_alpha")


rainfarm_arg_values = [(1.0, 1, 0, False), (1, 2, 0, False), (1, 4, 0, False)]


@pytest.mark.parametrize(rainfarm_arg_names, rainfarm_arg_values)
def test_rainfarm_shape(alpha, ds_factor, threshold, return_alpha):
    """Test that the output of rainfarm is consistent with the downscalnig factor."""

    window = metadata["xpixelsize"] * ds_factor
    precip_lr, __ = aggregate_fields_space(precip, metadata, window)

    rainfarm = downscaling.get_method("rainfarm")

    precip_hr = rainfarm(precip_lr, alpha, ds_factor, threshold, return_alpha)

    assert precip_hr.ndim == precip.ndim
    assert precip_hr.shape[0] == precip.shape[0]
    assert precip_hr.shape[1] == precip.shape[1]


rainfarm_arg_values = [(1.0, 2, 0, True), (None, 2, 0, True)]


@pytest.mark.parametrize(rainfarm_arg_names, rainfarm_arg_values)
def test_rainfarm_alpha(alpha, ds_factor, threshold, return_alpha):
    """Test that rainfarm computes and returns alpha."""

    window = metadata["xpixelsize"] * ds_factor
    precip_lr, __ = aggregate_fields_space(precip, metadata, window)

    rainfarm = downscaling.get_method("rainfarm")

    precip_hr = rainfarm(precip_lr, alpha, ds_factor, threshold, return_alpha)

    assert len(precip_hr) == 2
    if alpha is None:
        assert not precip_hr[1] == alpha
    else:
        assert precip_hr[1] == alpha
