# -*- coding: utf-8 -*-

import datetime as dt

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import raises

from pysteps.utils import dimension

test_data_not_trim = (
    # "data, window_size, axis, method, expected"
    (np.arange(6), 2, 0, "mean", np.array([0.5, 2.5, 4.5])),
    (
        np.arange(4 * 6).reshape(4, 6),
        (2, 3),
        (0, 1),
        "sum",
        np.array([[24, 42], [96, 114]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        (2, 2),
        (0, 1),
        "sum",
        np.array([[14, 22, 30], [62, 70, 78]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        2,
        (0, 1),
        "sum",
        np.array([[14, 22, 30], [62, 70, 78]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        (2, 3),
        (0, 1),
        "mean",
        np.array([[4.0, 7.0], [16.0, 19.0]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        (2, 2),
        (0, 1),
        "mean",
        np.array([[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        2,
        (0, 1),
        "mean",
        np.array([[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]]),
    ),
)


@pytest.mark.parametrize(
    "data, window_size, axis, method, expected", test_data_not_trim
)
def test_aggregate_fields(data, window_size, axis, method, expected):
    """
    Test the aggregate_fields function.
    The windows size must divide exactly the data dimensions.
    Internally, additional test are generated for situations where the
    windows size does not divide the data dimensions.
    The length of each dimension should be larger than 2.
    """

    assert_array_equal(
        dimension.aggregate_fields(data, window_size, axis=axis, method=method),
        expected,
    )

    # Test the trimming capabilities.
    data = np.pad(data, (0, 1))
    assert_array_equal(
        dimension.aggregate_fields(
            data, window_size, axis=axis, method=method, trim=True
        ),
        expected,
    )

    with raises(ValueError):
        dimension.aggregate_fields(data, window_size, axis=axis, method=method)


def test_aggregate_fields_errors():
    """
    Test that the errors are correctly captured in the aggregate_fields
    function.
    """
    data = np.arange(4 * 6).reshape(4, 6)

    with raises(ValueError):
        dimension.aggregate_fields(data, -1, axis=0)
    with raises(ValueError):
        dimension.aggregate_fields(data, 0, axis=0)
    with raises(ValueError):
        dimension.aggregate_fields(data, 1, method="invalid")

    with raises(TypeError):
        dimension.aggregate_fields(data, (1, 1), axis=0)


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
