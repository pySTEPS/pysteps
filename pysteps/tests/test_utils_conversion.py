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
                        "transform": None,
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
                        "transform": None,
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
                        "transform": None,
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
                        "transform": None,
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
                        "transform": None,
                        "accutime": 5,
                        "threshold": 1.25892541,
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
                        "transform": None,
                        "accutime": 5,
                        "threshold": 15.10710494,
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
                        "transform": None,
                        "accutime": 5,
                        "threshold": 0.04210719,
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
                        "transform": None,
                        "accutime": 5,
                        "threshold": 2.71828183,
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
                        "transform": None,
                        "accutime": 5,
                        "threshold": 32.61938194,
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
                        "transform": None,
                        "accutime": 5,
                        "threshold": 1,
                        "zerovalue": 1,
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
                        "transform": None,
                        "accutime": 5,
                        "threshold": 12,
                        "zerovalue": 12,
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
test_data = [
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([0.08333333]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([1]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "dB",
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([0.10491045]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "dB",
            "unit": "mm",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([1.25892541]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "dB",
            "unit": "dBZ",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([0.00350893]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "log",
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([0.22652349]),
    ),
    (
        np.array([1.0]),
        {
            "accutime": 5,
            "transform": "log",
            "unit": "mm",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([2.71828183]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "sqrt",
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([0.08333333]),
    ),
    (
        np.array([1.0]),
        {
            "accutime": 5,
            "transform": "sqrt",
            "unit": "mm",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([1.0]),
    ),
]


@pytest.mark.parametrize("R, metadata, expected", test_data)
def test_to_raindepth(R, metadata, expected):
    """Test the to_raindepth."""
    assert_array_almost_equal(conversion.to_raindepth(R, metadata)[0], expected)


# to_reflectivity
test_data = [
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([23.01029996]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([40.27719989]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "dB",
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([24.61029996]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "dB",
            "unit": "mm",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([41.87719989]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "dB",
            "unit": "dBZ",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([1]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "log",
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([29.95901167]),
    ),
    (
        np.array([1.0]),
        {
            "accutime": 5,
            "transform": "log",
            "unit": "mm",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([47.2259116]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "sqrt",
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([23.01029996]),
    ),
    (
        np.array([1.0]),
        {
            "accutime": 5,
            "transform": "sqrt",
            "unit": "mm",
            "threshold": 0,
            "zerovalue": 0,
        },
        np.array([40.27719989]),
    ),
]


@pytest.mark.parametrize("R, metadata, expected", test_data)
def test_to_reflectivity(R, metadata, expected):
    """Test the to_reflectivity."""
    assert_array_almost_equal(conversion.to_reflectivity(R, metadata)[0], expected)
