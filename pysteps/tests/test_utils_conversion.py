# -*- coding: utf-8 -*-

import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

# to_rainrate
test_data = [
    (
        1.0,
        {
            "accutime": 5,
            "unit": "mm/h",
        },
        1.0,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm",
        },
        12.0,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "dB",
            "offset": 0.0,
            "unit": "mm/h",
        },
        1.2589,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "dB",
            "offset": 0.0,
            "unit": "mm",
        },
        15.1071,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "dB",
            "offset": 0.0,
            "unit": "dBZ",
        },
        0.0421,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "log",
            "offset": 0.0,
            "unit": "mm/h",
        },
        2.7183,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "log",
            "offset": 0.0,
            "unit": "mm",
        },
        32.6194,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "sqrt",
            "unit": "mm/h",
        },
        1.0,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "sqrt",
            "unit": "mm",
        },
        12.0,
    ),
]


@pytest.mark.parametrize("data, attrs, expected", test_data)
def test_to_rainrate(data, attrs, expected):
    """Test the to_rainrate."""
    data_array = xr.DataArray([data], dims="x", attrs=attrs)
    output = data_array.pysteps.to_rainrate()
    assert output.attrs.get("unit") == "mm/h"
    assert_array_almost_equal(output.values, [expected], decimal=4)


# to_raindepth
test_data = [
    (
        1.0,
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm/h",
        },
        0.0833,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm",
        },
        1.0,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "dB",
            "offset": 0.0,
            "unit": "mm/h",
        },
        0.1049,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "dB",
            "offset": 0.0,
            "unit": "mm",
        },
        1.2589,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "dB",
            "offset": 0.0,
            "unit": "dBZ",
        },
        0.00353,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "log",
            "offset": 0.0,
            "unit": "mm/h",
        },
        0.2265,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "log",
            "offset": 0.0,
            "unit": "mm",
        },
        2.7183,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "sqrt",
            "unit": "mm/h",
        },
        0.0833,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "sqrt",
            "unit": "mm",
        },
        1.0,
    ),
]


@pytest.mark.parametrize("data, attrs, expected", test_data)
def test_to_raindepth(data, attrs, expected):
    """Test the to_raindepth."""
    data_array = xr.DataArray([data], dims="x", attrs=attrs)
    output = data_array.pysteps.to_raindepth()
    assert output.attrs.get("unit") == "mm"
    assert_array_almost_equal(output.values, [expected], decimal=4)


# to_reflectivity
test_data = [
    (
        1.0,
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm/h",
        },
        23.0103,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm",
        },
        40.2772,
    ),
    (
        1,
        {
            "accutime": 5,
            "transform": "dB",
            "offset": 0.0,
            "unit": "mm/h",
        },
        24.6103,
    ),
    (
        1,
        {
            "accutime": 5,
            "transform": "dB",
            "offset": 0.0,
            "unit": "mm",
        },
        41.8772,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "dB",
            "offset": 0.0,
            "unit": "dBZ",
        },
        1.0,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "log",
            "offset": 0.0,
            "unit": "mm/h",
        },
        29.9590,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "log",
            "offset": 0.0,
            "unit": "mm",
        },
        47.2259,
    ),
    (
        1,
        {
            "accutime": 5,
            "transform": "sqrt",
            "unit": "mm/h",
        },
        23.0103,
    ),
    (
        1.0,
        {
            "accutime": 5,
            "transform": "sqrt",
            "unit": "mm",
        },
        40.2772,
    ),
]


@pytest.mark.parametrize("data, attrs, expected", test_data)
def test_to_reflectivity(data, attrs, expected):
    """Test the to_reflectivity."""
    data_array = xr.DataArray([data], dims="x", attrs=attrs)
    output = data_array.pysteps.to_reflectivity(offset=0.0, to_decibels=True)
    assert output.attrs.get("unit") == "dBZ"
    assert_array_almost_equal(output.values, [expected], decimal=4)
    output = data_array.pysteps.to_reflectivity(offset=0.0, to_decibels=False)
    assert output.attrs.get("unit") == "Z"
    assert_array_almost_equal(output.values, [10.0 ** (expected / 10.0)], decimal=0)
