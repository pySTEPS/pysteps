# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.utils import transformation

# boxcox_transform
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
        None,
        None,
        None,
        False,
        np.array([0]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "BoxCox",
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        None,
        None,
        None,
        True,
        np.array([np.exp(1)]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        1.0,
        None,
        None,
        False,
        np.array([0]),
    ),
    (
        np.array([1]),
        {
            "accutime": 5,
            "transform": "BoxCox",
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        1.0,
        None,
        None,
        True,
        np.array([2.0]),
    ),
]


@pytest.mark.parametrize(
    "R, metadata, Lambda, threshold, zerovalue, inverse, expected", test_data
)
def test_boxcox_transform(R, metadata, Lambda, threshold, zerovalue, inverse, expected):
    """Test the boxcox_transform."""
    assert_array_almost_equal(
        transformation.boxcox_transform(
            R, metadata, Lambda, threshold, zerovalue, inverse
        )[0],
        expected,
    )


# dB_transform
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
        None,
        None,
        False,
        np.array([0]),
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
        None,
        None,
        True,
        np.array([1.25892541]),
    ),
]


@pytest.mark.parametrize(
    "R, metadata, threshold, zerovalue, inverse, expected", test_data
)
def test_dB_transform(R, metadata, threshold, zerovalue, inverse, expected):
    """Test the dB_transform."""
    assert_array_almost_equal(
        transformation.dB_transform(R, metadata, threshold, zerovalue, inverse)[0],
        expected,
    )


# NQ_transform
test_data = [
    (
        np.array([1, 2]),
        {
            "accutime": 5,
            "transform": None,
            "unit": "mm/h",
            "threshold": 0,
            "zerovalue": 0,
        },
        False,
        np.array([-0.4307273, 0.4307273]),
    )
]


@pytest.mark.parametrize("R, metadata, inverse, expected", test_data)
def test_NQ_transform(R, metadata, inverse, expected):
    """Test the NQ_transform."""
    assert_array_almost_equal(
        transformation.NQ_transform(R, metadata, inverse)[0], expected
    )


# sqrt_transform
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
        False,
        np.array([1]),
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
        True,
        np.array([1]),
    ),
]


@pytest.mark.parametrize("R, metadata, inverse, expected", test_data)
def test_sqrt_transform(R, metadata, inverse, expected):
    """Test the sqrt_transform."""
    assert_array_almost_equal(
        transformation.sqrt_transform(R, metadata, inverse)[0], expected
    )
