# -*- coding: utf-8 -*-
import numpy as np
import pytest
import xarray as xr

from pysteps.tests.helpers import assert_dataset_equivalent
from pysteps.utils import transformation

# boxcox_transform
test_data_boxcox_transform = [
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": np.e,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        None,
        None,
        None,
        False,
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([0.0]),
                    {
                        "units": "mm/h",
                        "transform": "BoxCox",
                        "accutime": 5,
                        "threshold": 1,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm/h",
                        "transform": "BoxCox",
                        "accutime": 5,
                        "threshold": 1,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        None,
        None,
        None,
        True,
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([np.exp(1.0)]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": np.e,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": np.e,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        1.0,
        None,
        None,
        False,
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([np.e - 2]),
                    {
                        "units": "mm/h",
                        "transform": "BoxCox",
                        "accutime": 5,
                        "threshold": np.e - 1,
                        "zerovalue": np.e - 2,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([np.e - 2]),
                    {
                        "units": "mm/h",
                        "transform": "BoxCox",
                        "accutime": 5,
                        "threshold": np.e - 1,
                        "zerovalue": np.e - 2,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        1.0,
        None,
        None,
        True,
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([0.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": np.e,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
]


@pytest.mark.parametrize(
    "dataset, Lambda, threshold, zerovalue, inverse, expected",
    test_data_boxcox_transform,
)
def test_boxcox_transform(dataset, Lambda, threshold, zerovalue, inverse, expected):
    """Test the boxcox_transform."""
    actual = transformation.boxcox_transform(
        dataset, Lambda, threshold, zerovalue, inverse
    )
    assert_dataset_equivalent(actual, expected)


# dB_transform
test_data_dB_transform = [
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 1,
                        "zerovalue": 1,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        None,
        None,
        False,
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([0.0]),
                    {
                        "units": "mm/h",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 0,
                        "zerovalue": -5,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([0.0]),
                    {
                        "units": "mm/h",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 0,
                        "zerovalue": -5,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        None,
        None,
        True,
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 1,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
]


@pytest.mark.parametrize(
    "dataset, threshold, zerovalue, inverse, expected", test_data_dB_transform
)
def test_dB_transform(dataset, threshold, zerovalue, inverse, expected):
    """Test the dB_transform."""
    actual = transformation.dB_transform(dataset, threshold, zerovalue, inverse)
    assert_dataset_equivalent(actual, expected)


# NQ_transform
test_data_NQ_transform = [
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0, 2.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 0,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        False,
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([-0.4307273, 0.4307273]),
                    {
                        "units": "mm/h",
                        "transform": "NQT",
                        "accutime": 5,
                        "threshold": 0.4307273,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
]


@pytest.mark.parametrize("dataset, inverse, expected", test_data_NQ_transform)
def test_NQ_transform(dataset, inverse, expected):
    """Test the NQ_transform."""
    actual = transformation.NQ_transform(dataset, inverse)
    assert_dataset_equivalent(actual, expected)


# sqrt_transform
test_data_sqrt_transform = [
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0, 4.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 4,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        False,
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0, 2.0]),
                    {
                        "units": "mm/h",
                        "transform": "sqrt",
                        "accutime": 5,
                        "threshold": 2,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0, 2.0]),
                    {
                        "units": "mm/h",
                        "transform": "sqrt",
                        "accutime": 5,
                        "threshold": 2,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        True,
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0, 4.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 4,
                        "zerovalue": 0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
]


@pytest.mark.parametrize("dataset, inverse, expected", test_data_sqrt_transform)
def test_sqrt_transform(dataset, inverse, expected):
    """Test the sqrt_transform."""
    actual = transformation.sqrt_transform(dataset, inverse)
    assert_dataset_equivalent(actual, expected)
