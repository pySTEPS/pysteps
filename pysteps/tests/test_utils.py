# -*- coding: utf-8 -*-

import datetime as dt
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from pysteps.utils import arrays, conversion, dimension, transformation

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# arrays

# compute_centred_coord_array
test_data = [
    (2, 2, [np.array([[-1, 0]]).T, np.array([[-1, 0]])]),
    (3, 3, [np.array([[-1, 0, 1]]).T, np.array([[-1, 0, 1]])]),
    (3, 2, [np.array([[-1, 0, 1]]).T, np.array([[-1, 0]])]),
    (2, 3, [np.array([[-1, 0]]).T, np.array([[-1, 0, 1]])]),
]


@pytest.mark.parametrize("M, N, expected", test_data)
def test_compute_centred_coord_array(M, N, expected):
    """Test the compute_centred_coord_array."""
    assert_array_equal(arrays.compute_centred_coord_array(M, N)[0], expected[0])
    assert_array_equal(arrays.compute_centred_coord_array(M, N)[1], expected[1])


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# conversion

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


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# dimension

# aggregate_fields_time
timestamps = [dt.datetime.now() + dt.timedelta(minutes=t) for t in range(10)]
test_data = [
    (
        np.ones((10, 1, 1)),
        {"unit": "mm/h", "timestamps": timestamps},
        2,
        False,
        np.ones((5, 1, 1)),
    ),
    (
        np.ones((10, 1, 1)),
        {"unit": "mm", "timestamps": timestamps},
        2,
        False,
        2 * np.ones((5, 1, 1)),
    ),
]


@pytest.mark.parametrize(
    "R, metadata, time_window_min, ignore_nan, expected", test_data
)
def test_aggregate_fields_time(R, metadata, time_window_min, ignore_nan, expected):
    """Test the aggregate_fields_time."""
    assert_array_equal(
        dimension.aggregate_fields_time(R, metadata, time_window_min, ignore_nan)[0],
        expected,
    )


# aggregate_fields_space
test_data = [
    (
        np.ones((1, 10, 10)),
        {"unit": "mm/h", "xpixelsize": 1, "ypixelsize": 1},
        2,
        False,
        np.ones((1, 5, 5)),
    ),
    (
        np.ones((1, 10, 10)),
        {"unit": "mm", "xpixelsize": 1, "ypixelsize": 1},
        2,
        False,
        4 * np.ones((1, 5, 5)),
    ),
]


@pytest.mark.parametrize("R, metadata, space_window, ignore_nan, expected", test_data)
def test_aggregate_fields_space(R, metadata, space_window, ignore_nan, expected):
    """Test the aggregate_fields_space."""
    assert_array_equal(
        dimension.aggregate_fields_space(R, metadata, space_window, ignore_nan)[0],
        expected,
    )


# clip_domain
R = np.zeros((4, 4))
R[:2, :] = 1
test_data = [
    (
        R,
        {
            "x1": 0,
            "x2": 4,
            "y1": 0,
            "y2": 4,
            "xpixelsize": 1,
            "ypixelsize": 1,
            "zerovalue": 0,
            "yorigin": "upper",
        },
        None,
        R,
    ),
    (
        R,
        {
            "x1": 0,
            "x2": 4,
            "y1": 0,
            "y2": 4,
            "xpixelsize": 1,
            "ypixelsize": 1,
            "zerovalue": 0,
            "yorigin": "lower",
        },
        (2, 4, 2, 4),
        np.zeros((2, 2)),
    ),
    (
        R,
        {
            "x1": 0,
            "x2": 4,
            "y1": 0,
            "y2": 4,
            "xpixelsize": 1,
            "ypixelsize": 1,
            "zerovalue": 0,
            "yorigin": "upper",
        },
        (2, 4, 2, 4),
        np.ones((2, 2)),
    ),
]


@pytest.mark.parametrize("R, metadata, extent, expected", test_data)
def test_clip_domain(R, metadata, extent, expected):
    """Test the clip_domain."""
    assert_array_equal(dimension.clip_domain(R, metadata, extent)[0], expected)


# square_domain
R = np.zeros((4, 2))
test_data = [
    # square by padding
    (
        R,
        {"x1": 0, "x2": 2, "y1": 0, "y2": 4, "xpixelsize": 1, "ypixelsize": 1},
        "pad",
        False,
        np.zeros((4, 4)),
    ),
    # square by cropping
    (
        R,
        {"x1": 0, "x2": 2, "y1": 0, "y2": 4, "xpixelsize": 1, "ypixelsize": 1},
        "crop",
        False,
        np.zeros((2, 2)),
    ),
    # inverse square by padding
    (
        np.zeros((4, 4)),
        {
            "x1": -1,
            "x2": 3,
            "y1": 0,
            "y2": 4,
            "xpixelsize": 1,
            "ypixelsize": 1,
            "orig_domain": (4, 2),
            "square_method": "pad",
        },
        "pad",
        True,
        R,
    ),
    # inverse square by cropping
    (
        np.zeros((2, 2)),
        {
            "x1": 0,
            "x2": 2,
            "y1": 1,
            "y2": 3,
            "xpixelsize": 1,
            "ypixelsize": 1,
            "orig_domain": (4, 2),
            "square_method": "crop",
        },
        "crop",
        True,
        R,
    ),
]


@pytest.mark.parametrize("R, metadata, method, inverse, expected", test_data)
def test_square_domain(R, metadata, method, inverse, expected):
    """Test the square_domain."""
    assert_array_equal(
        dimension.square_domain(R, metadata, method, inverse)[0], expected
    )


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# transformation

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
