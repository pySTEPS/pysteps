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


@pytest.mark.parametrize("Lambda", [0.0, 0.5, 1.0])
def test_boxcox_transform_roundtrip_with_zeros(Lambda):
    """Test that the inverse boxcox_transform maps the zero value back to zero."""
    R = np.array([0.0, 0.05, 0.1, 1.0, 10.0])
    metadata = {
        "accutime": 5,
        "transform": None,
        "unit": "mm/h",
        "threshold": 0.1,
        "zerovalue": 0.0,
    }
    R_t, metadata_t = transformation.boxcox_transform(R.copy(), metadata, Lambda)
    R_b, metadata_b = transformation.boxcox_transform(R_t, metadata_t, inverse=True)
    assert_array_almost_equal(R_b, [0.0, 0.0, 0.1, 1.0, 10.0])
    assert metadata_b["transform"] is None
    assert metadata_b["threshold"] == pytest.approx(0.1)


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
