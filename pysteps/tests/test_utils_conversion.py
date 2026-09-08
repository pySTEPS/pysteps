# -*- coding: utf-8 -*-
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

from pysteps.tests.helpers import assert_dataset_equivalent
from pysteps.utils import conversion

# to_rainrate
test_data_to_rainrate = [
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([12.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 12.0,
                        "zerovalue": 12.0,
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
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.25892541]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 1.25892541,
                        "zerovalue": 0.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([15.10710494]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 15.10710494,
                        "zerovalue": 0.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
        ),
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([0.04210719]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 0.04210719,
                        "zerovalue": 0.0,
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
                        "transform": "log",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([2.71828183]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 2.71828183,
                        "zerovalue": 0.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "transform": "log",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([32.61938194]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 32.61938194,
                        "zerovalue": 0.0,
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
                        "transform": "sqrt",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "transform": "sqrt",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([12.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 12.0,
                        "zerovalue": 12.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
    ),
]


@pytest.mark.parametrize("dataset, expected", test_data_to_rainrate)
def test_to_rainrate(dataset, expected):
    """Test the to_rainrate."""
    actual = conversion.to_rainrate(dataset)
    assert_dataset_equivalent(actual, expected)


# to_raindepth
test_data_to_raindepth = [
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([0.08333333]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 0.08333333,
                        "zerovalue": 0.08333333,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
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
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([0.10491045]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 0.10491045,
                        "zerovalue": 0.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.25892541]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 1.25892541,
                        "zerovalue": 0.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
        ),
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([0.00350893]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 0.00350893,
                        "zerovalue": 0.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
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
                        "transform": "log",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([0.22652349]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 0.22652349,
                        "zerovalue": 0.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
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
                        "transform": "sqrt",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([0.08333333]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 0.08333333,
                        "zerovalue": 0.08333333,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "transform": "sqrt",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
    ),
]


@pytest.mark.parametrize("dataset, expected", test_data_to_raindepth)
def test_to_raindepth(dataset, expected):
    """Test the to_raindepth."""
    actual = conversion.to_raindepth(dataset)
    assert_dataset_equivalent(actual, expected)


# to_reflectivity
test_data_to_reflectivity = [
    (
        xr.Dataset(
            data_vars={
                "precip_intensity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm/h",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([23.01029996]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 23.01029996,
                        "zerovalue": 18.01029996,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([40.27719989]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 40.27719989,
                        "zerovalue": 35.27719989,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
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
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([24.61029996]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 24.61029996,
                        "zerovalue": 19.61029996,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([41.87719989]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 41.87719989,
                        "zerovalue": 36.87719989,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
        ),
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": -4.0,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
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
                        "transform": "log",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([29.95901167]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 29.95901167,
                        "zerovalue": 24.95901167,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "transform": "log",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([47.2259116]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 47.2259116,
                        "zerovalue": 42.2259116,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
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
                        "transform": "sqrt",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_intensity"},
        ),
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([23.01029996]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 23.01029996,
                        "zerovalue": 18.01029996,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
        ),
    ),
    (
        xr.Dataset(
            data_vars={
                "precip_accum": (
                    ["x"],
                    np.array([1.0]),
                    {
                        "units": "mm",
                        "transform": "sqrt",
                        "accutime": 5,
                        "threshold": 1.0,
                        "zerovalue": 1.0,
                    },
                )
            },
            attrs={"precip_var": "precip_accum"},
        ),
        xr.Dataset(
            data_vars={
                "reflectivity": (
                    ["x"],
                    np.array([40.27719989]),
                    {
                        "units": "dBZ",
                        "transform": "dB",
                        "accutime": 5,
                        "threshold": 40.27719989,
                        "zerovalue": 35.27719989,
                    },
                )
            },
            attrs={"precip_var": "reflectivity"},
        ),
    ),
]


@pytest.mark.parametrize("dataset, expected", test_data_to_reflectivity)
def test_to_reflectivity(dataset, expected):
    """Test the to_reflectivity."""
    actual = conversion.to_reflectivity(dataset)
    assert_dataset_equivalent(actual, expected)
