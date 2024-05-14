# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pysteps import downscaling
from pysteps.tests.helpers import get_precipitation_fields
from pysteps.utils import aggregate_fields_space, square_domain, aggregate_fields


@pytest.fixture(scope="module")
def data():
    precip, metadata = get_precipitation_fields(
        num_prev_files=0, num_next_files=0, return_raw=False, metadata=True
    )
    precip = precip.filled()
    precip, metadata = square_domain(precip, metadata, "crop")
    return precip, metadata


rainfarm_arg_names = (
    "alpha",
    "ds_factor",
    "threshold",
    "return_alpha",
    "spectral_fusion",
    "kernel_type",
)
rainfarm_arg_values = [
    (1.0, 1, 0, False, False, None),
    (1, 2, 0, False, False, "gaussian"),
    (1, 4, 0, False, False, "tophat"),
    (1, 4, 0, False, True, "uniform"),
]


@pytest.mark.parametrize(rainfarm_arg_names, rainfarm_arg_values)
def test_rainfarm_shape(
    data,
    alpha,
    ds_factor,
    threshold,
    return_alpha,
    spectral_fusion,
    kernel_type,
):
    """Test that the output of rainfarm is consistent with the downscaling factor."""
    precip, metadata = data
    window = metadata["xpixelsize"] * ds_factor
    precip_lr, __ = aggregate_fields_space(precip, metadata, window)

    rainfarm = downscaling.get_method("rainfarm")
    precip_hr = rainfarm(
        precip_lr,
        alpha=alpha,
        ds_factor=ds_factor,
        threshold=threshold,
        return_alpha=return_alpha,
        spectral_fusion=spectral_fusion,
        kernel_type=kernel_type,
    )

    assert precip_hr.ndim == precip.ndim
    assert precip_hr.shape[0] == precip.shape[0]
    assert precip_hr.shape[1] == precip.shape[1]


rainfarm_arg_values = [
    (1.0, 1, 0, False, False, None),
    (1, 2, 0, False, False, None),
    (1, 4, 0, False, False, None),
    (1, 4, 0, False, True, None),
]


@pytest.mark.parametrize(rainfarm_arg_names, rainfarm_arg_values)
def test_rainfarm_aggregate(
    data,
    alpha,
    ds_factor,
    threshold,
    return_alpha,
    spectral_fusion,
    kernel_type,
):
    """Test that the output of rainfarm is equal to original when aggregated."""
    precip, metadata = data
    window = metadata["xpixelsize"] * ds_factor
    precip_lr, __ = aggregate_fields_space(precip, metadata, window)

    rainfarm = downscaling.get_method("rainfarm")
    precip_hr = rainfarm(
        precip_lr,
        alpha=alpha,
        ds_factor=ds_factor,
        threshold=threshold,
        return_alpha=return_alpha,
        spectral_fusion=spectral_fusion,
        kernel_type=kernel_type,
    )
    precip_low = aggregate_fields(precip_hr, ds_factor, axis=(0, 1))
    precip_lr[precip_lr < threshold] = 0.0

    np.testing.assert_array_almost_equal(precip_lr, precip_low)


rainfarm_arg_values = [(1.0, 2, 0, True, False, None), (None, 2, 0, True, True, None)]


@pytest.mark.parametrize(rainfarm_arg_names, rainfarm_arg_values)
def test_rainfarm_alpha(
    data,
    alpha,
    ds_factor,
    threshold,
    return_alpha,
    spectral_fusion,
    kernel_type,
):
    """Test that rainfarm computes and returns alpha."""
    precip, metadata = data
    window = metadata["xpixelsize"] * ds_factor
    precip_lr, __ = aggregate_fields_space(precip, metadata, window)

    rainfarm = downscaling.get_method("rainfarm")
    precip_hr = rainfarm(
        precip_lr,
        alpha=alpha,
        ds_factor=ds_factor,
        threshold=threshold,
        return_alpha=return_alpha,
        spectral_fusion=spectral_fusion,
        kernel_type=kernel_type,
    )

    assert len(precip_hr) == 2
    if alpha is None:
        assert not precip_hr[1] == alpha
    else:
        assert precip_hr[1] == alpha
