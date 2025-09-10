# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pysteps import downscaling
from pysteps.tests.helpers import get_precipitation_fields
from pysteps.utils import aggregate_fields_space, square_domain, aggregate_fields


@pytest.fixture(scope="module")
def dataset():
    precip_dataset = get_precipitation_fields(
        num_prev_files=0, num_next_files=0, return_raw=False, metadata=True
    )
    precip_dataset = square_domain(precip_dataset, "crop")
    return precip_dataset


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
    dataset,
    alpha,
    ds_factor,
    threshold,
    return_alpha,
    spectral_fusion,
    kernel_type,
):
    """Test that the output of rainfarm is consistent with the downscaling factor."""
    precip_var = dataset.attrs["precip_var"]
    window = dataset.x.attrs["stepsize"] * ds_factor
    precip_lr_dataset = aggregate_fields_space(dataset, window)

    rainfarm = downscaling.get_method("rainfarm")
    precip_hr_dataset = rainfarm(
        precip_lr_dataset,
        alpha=alpha,
        ds_factor=ds_factor,
        threshold=threshold,
        return_alpha=return_alpha,
        spectral_fusion=spectral_fusion,
        kernel_type=kernel_type,
    )

    assert precip_hr_dataset[precip_var].values.ndim == dataset[precip_var].values.ndim
    assert (
        precip_hr_dataset[precip_var].values.shape[0]
        == dataset[precip_var].values.shape[0]
    )
    assert (
        precip_hr_dataset[precip_var].values.shape[1]
        == dataset[precip_var].values.shape[1]
    )


rainfarm_arg_values = [
    (1.0, 1, 0, False, False, None),
    (1, 2, 0, False, False, None),
    (1, 4, 0, False, False, None),
    (1, 4, 0, False, True, None),
]


@pytest.mark.parametrize(rainfarm_arg_names, rainfarm_arg_values)
def test_rainfarm_aggregate(
    dataset,
    alpha,
    ds_factor,
    threshold,
    return_alpha,
    spectral_fusion,
    kernel_type,
):
    """Test that the output of rainfarm is equal to original when aggregated."""
    precip_var = dataset.attrs["precip_var"]
    window = dataset.x.attrs["stepsize"] * ds_factor
    precip_lr_dataset = aggregate_fields_space(dataset, window)

    rainfarm = downscaling.get_method("rainfarm")
    precip_hr_dataset = rainfarm(
        precip_lr_dataset,
        alpha=alpha,
        ds_factor=ds_factor,
        threshold=threshold,
        return_alpha=return_alpha,
        spectral_fusion=spectral_fusion,
        kernel_type=kernel_type,
    )
    precip_low_dataset = aggregate_fields(precip_hr_dataset, ds_factor, dim=("y", "x"))
    precip_lr = precip_lr_dataset[precip_var].values
    precip_lr[precip_lr < threshold] = 0.0
    precip_low = precip_low_dataset[precip_var].values

    np.testing.assert_array_almost_equal(precip_lr, precip_low)


rainfarm_arg_values = [(1.0, 2, 0, True, False, None), (None, 2, 0, True, True, None)]


@pytest.mark.parametrize(rainfarm_arg_names, rainfarm_arg_values)
def test_rainfarm_alpha(
    dataset,
    alpha,
    ds_factor,
    threshold,
    return_alpha,
    spectral_fusion,
    kernel_type,
):
    """Test that rainfarm computes and returns alpha."""
    window = dataset.x.attrs["stepsize"] * ds_factor
    precip_lr_dataset = aggregate_fields_space(dataset, window)

    rainfarm = downscaling.get_method("rainfarm")
    precip_hr_dataset = rainfarm(
        precip_lr_dataset,
        alpha=alpha,
        ds_factor=ds_factor,
        threshold=threshold,
        return_alpha=return_alpha,
        spectral_fusion=spectral_fusion,
        kernel_type=kernel_type,
    )

    assert len(precip_hr_dataset) == 2
    if alpha is None:
        assert not precip_hr_dataset[1] == alpha
    else:
        assert precip_hr_dataset[1] == alpha
