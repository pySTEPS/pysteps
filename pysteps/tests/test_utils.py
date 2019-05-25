# -*- coding: utf-8 -*-

import datetime as dt
import pytest
import numpy as np
from pysteps.utils import arrays, conversion, dimension
from numpy.testing import assert_array_equal, assert_array_almost_equal

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
        np.array([0.02513093]),
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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# conversion


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
        np.array([0.00209424]),
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
        np.array([24.99687083]),
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
        np.array([41.18458952]),
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
        np.array([26.49687083]),
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
        np.array([42.68458952]),
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
        np.array([31.51128805]),
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
        np.array([47.69900675]),
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
        np.array([24.99687083]),
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
        np.array([41.18458952]),
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
timestamps = [dt.datetime.now() + dt.timedelta(minutes=t) for t in range(10)]
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


@pytest.mark.parametrize("R, metadata, space_window_m, ignore_nan, expected", test_data)
def test_aggregate_fields_space(R, metadata, space_window_m, ignore_nan, expected):
    """Test the aggregate_fields_space."""
    assert_array_equal(
        dimension.aggregate_fields_space(R, metadata, space_window_m, ignore_nan)[0],
        expected,
    )
