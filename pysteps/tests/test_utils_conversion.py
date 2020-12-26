# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.utils import conversion

# to_rainrate
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
        np.array([1]),
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
        np.array([12]),
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
        np.array([1.25892541]),
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
        np.array([15.10710494]),
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
        np.array([0.04210719]),
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
        np.array([2.71828183]),
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
        np.array([32.61938194]),
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
        np.array([1]),
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
        np.array([12.0]),
    ),
]


@pytest.mark.parametrize("R, metadata, expected", test_data)
def test_to_rainrate(R, metadata, expected):
    """Test the to_rainrate."""
    assert_array_almost_equal(conversion.to_rainrate(R, metadata)[0], expected)


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
