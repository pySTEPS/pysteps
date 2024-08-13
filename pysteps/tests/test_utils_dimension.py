# -*- coding: utf-8 -*-

import datetime as dt

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pytest import raises

from pysteps.utils import dimension
from pysteps.xarray_helpers import convert_input_to_xarray_dataset

fillvalues_metadata = {
    "x1": 0,
    "x2": 4,
    "y1": 0,
    "y2": 4,
    "zerovalue": 0,
    "yorigin": "lower",
    "unit": "mm/h",
    "transform": None,
    "accutime": 5,
    "threshold": 1.0,
    "projection": "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0",
    "zr_a": 200,
    "zr_b": 1.6,
    "cartesian_unit": "km",
    "institution": "KNMI",
}

test_data_not_trim = (
    (
        np.arange(12).reshape(2, 6),
        2,
        "x",
        "mean",
        np.array([[0.5, 2.5, 4.5], [6.5, 8.5, 10.5]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        (2, 3),
        ("y", "x"),
        "sum",
        np.array([[24, 42], [96, 114]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        (2, 2),
        ("y", "x"),
        "sum",
        np.array([[14, 22, 30], [62, 70, 78]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        2,
        ("y", "x"),
        "sum",
        np.array([[14, 22, 30], [62, 70, 78]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        (2, 3),
        ("y", "x"),
        "mean",
        np.array([[4.0, 7.0], [16.0, 19.0]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        (2, 2),
        ("y", "x"),
        "mean",
        np.array([[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        2,
        ("y", "x"),
        "mean",
        np.array([[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]]),
    ),
)


@pytest.mark.parametrize("data, window_size, dim, method, expected", test_data_not_trim)
def test_aggregate_fields(data, window_size, dim, method, expected):
    """
    Test the aggregate_fields function.
    The windows size must divide exactly the data dimensions.
    Internally, additional test are generated for situations where the
    windows size does not divide the data dimensions.
    The length of each dimension should be larger than 2.
    """
    dataset = convert_input_to_xarray_dataset(data, None, fillvalues_metadata)

    actual = dimension.aggregate_fields(dataset, window_size, dim=dim, method=method)
    assert_array_equal(actual["precip_intensity"].values, expected)

    # Test the trimming capabilities.
    if np.ndim(window_size) == 0:
        data = np.pad(data, ((0, 0), (0, 1)))
    else:
        data = np.pad(data, (0, 1))
    dataset = convert_input_to_xarray_dataset(data, None, fillvalues_metadata)

    actual = dimension.aggregate_fields(
        dataset, window_size, dim=dim, method=method, trim=True
    )
    assert_array_equal(actual["precip_intensity"].values, expected)

    with raises(ValueError):
        dimension.aggregate_fields(dataset, window_size, dim=dim, method=method)


test_data_agg_w_velocity = (
    (
        np.arange(12).reshape(2, 6),
        np.arange(12).reshape(2, 6),
        np.arange(12).reshape(2, 6),
        np.arange(0, 1.2, 0.1).reshape(2, 6),
        2,
        "x",
        "mean",
        "mean",
        np.array([[0.5, 2.5, 4.5], [6.5, 8.5, 10.5]]),
        np.array([[0.5, 2.5, 4.5], [6.5, 8.5, 10.5]]),
        np.array([[0, 0.2, 0.4], [0.6, 0.8, 1]]),
    ),
    (
        np.arange(4 * 6).reshape(4, 6),
        np.arange(4 * 6).reshape(4, 6),
        np.arange(4 * 6).reshape(4, 6),
        np.arange(0, 1.2, 0.05).reshape(4, 6),
        (2, 3),
        ("y", "x"),
        "mean",
        "sum",
        np.array([[4, 7], [16, 19]]),
        np.array([[24, 42], [96, 114]]),
        np.array([[0, 0.15], [0.6, 0.75]]),
    ),
)


@pytest.mark.parametrize(
    "data, data_vx, data_vy, data_qual, window_size, dim, method, velocity_method, expected, expected_v, expected_qual",
    test_data_agg_w_velocity,
)
def test_aggregate_fields_w_velocity(
    data,
    data_vx,
    data_vy,
    data_qual,
    window_size,
    dim,
    method,
    velocity_method,
    expected,
    expected_v,
    expected_qual,
):
    """
    Test the aggregate_fields function for dataset with velocity information.
    The windows size must divide exactly the data dimensions.
    Internally, additional test are generated for situations where the
    windows size does not divide the data dimensions.
    The length of each dimension should be larger than 2.
    """
    dataset = convert_input_to_xarray_dataset(data, None, fillvalues_metadata)
    dataset = dataset.assign(
        {
            "velocity_x": (("y", "x"), data_vx),
            "velocity_y": (("y", "x"), data_vy),
            "quality": (("y", "x"), data_qual),
        }
    )

    actual = dimension.aggregate_fields(
        dataset, window_size, dim=dim, method=method, velocity_method=velocity_method
    )
    assert_array_equal(actual["precip_intensity"].values, expected)
    assert_array_equal(actual["velocity_x"].values, expected_v)
    assert_array_equal(actual["velocity_y"].values, expected_v)
    assert_array_almost_equal(actual["quality"].values, expected_qual)


def test_aggregate_fields_errors():
    """
    Test that the errors are correctly captured in the aggregate_fields
    function.
    """
    data = np.arange(4 * 6).reshape(4, 6)
    dataset = convert_input_to_xarray_dataset(data, None, fillvalues_metadata)

    with raises(ValueError):
        dimension.aggregate_fields(dataset, -1, dim="y")
    with raises(ValueError):
        dimension.aggregate_fields(dataset, 0, dim="y")
    with raises(ValueError):
        dimension.aggregate_fields(dataset, 1, method="invalid")

    with raises(TypeError):
        dimension.aggregate_fields(dataset, (1, 1), dim="y")


# aggregate_fields_time
now = dt.datetime.now()
timestamps = [now + dt.timedelta(minutes=t) for t in range(10)]
test_data_time = [
    (
        np.ones((2, 2)),
        {"unit": "mm/h", "timestamps": timestamps},
        2,
        False,
        np.ones((5, 2, 2)),
    ),
    (
        np.ones((2, 2)),
        {"unit": "mm", "timestamps": timestamps},
        2,
        False,
        2 * np.ones((5, 2, 2)),
    ),
]


@pytest.mark.parametrize(
    "data, metadata, time_window_min, ignore_nan, expected", test_data_time
)
def test_aggregate_fields_time(data, metadata, time_window_min, ignore_nan, expected):
    """Test the aggregate_fields_time."""
    dataset_ref = convert_input_to_xarray_dataset(
        data, None, {**fillvalues_metadata, **metadata}
    )
    datasets = []
    for timestamp in metadata["timestamps"]:
        dataset_ = dataset_ref.copy(deep=True)
        dataset_ = dataset_.expand_dims(dim="time", axis=0)
        dataset_ = dataset_.assign_coords(time=("time", [timestamp]))
        datasets.append(dataset_)

    dataset = xr.concat(datasets, dim="time")
    assert_array_equal(
        dimension.aggregate_fields_time(dataset, time_window_min, ignore_nan)[
            "precip_intensity" if metadata["unit"] == "mm/h" else "precip_accum"
        ].values,
        expected,
    )


# aggregate_fields_space
test_data_space = [
    (
        np.ones((10, 10)),
        {
            "unit": "mm/h",
            "x1": 0,
            "x2": 10,
            "y1": 0,
            "y2": 10,
            "xpixelsize": 1,
            "ypixelsize": 1,
        },
        2,
        False,
        np.ones((5, 5)),
    ),
    (
        np.ones((10, 10)),
        {
            "unit": "mm",
            "x1": 0,
            "x2": 10,
            "y1": 0,
            "y2": 10,
            "xpixelsize": 1,
            "ypixelsize": 1,
        },
        2,
        False,
        np.ones((5, 5)),
    ),
    (
        np.ones((10, 10)),
        {
            "unit": "mm/h",
            "x1": 0,
            "x2": 10,
            "y1": 0,
            "y2": 20,
            "xpixelsize": 1,
            "ypixelsize": 2,
        },
        (4, 2),
        False,
        np.ones((5, 5)),
    ),
]


@pytest.mark.parametrize(
    "data, metadata, space_window, ignore_nan, expected", test_data_space
)
def test_aggregate_fields_space(data, metadata, space_window, ignore_nan, expected):
    """Test the aggregate_fields_space."""
    dataset = convert_input_to_xarray_dataset(
        data, None, {**fillvalues_metadata, **metadata}
    )
    assert_array_equal(
        dimension.aggregate_fields_space(dataset, space_window, ignore_nan)[
            "precip_intensity" if metadata["unit"] == "mm/h" else "precip_accum"
        ].values,
        expected,
    )


# clip_domain
R = np.zeros((4, 4))
R[:2, :] = 1
test_data_clip_domain = [
    (
        R,
        {"yorigin": "lower"},
        None,
        R,
    ),
    (
        R,
        {"yorigin": "lower"},
        (2, 4, 2, 4),
        np.zeros((2, 2)),
    ),
    (
        R,
        {"yorigin": "upper"},
        (2, 4, 2, 4),
        np.ones((2, 2)),
    ),
]


@pytest.mark.parametrize("R, metadata, extent, expected", test_data_clip_domain)
def test_clip_domain(R, metadata, extent, expected):
    """Test the clip_domain."""
    dataset = convert_input_to_xarray_dataset(
        R, None, {**fillvalues_metadata, **metadata}
    )
    assert_array_equal(
        dimension.clip_domain(dataset, extent)["precip_intensity"].values, expected
    )


# square_domain
R = np.zeros((4, 2))
test_data_square = [
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
            "orig_domain": (np.array([0.5, 1.5, 2.5, 3.5]), np.array([0.5, 1.5])),
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
            "orig_domain": (np.array([0.5, 1.5, 2.5, 3.5]), np.array([0.5, 1.5])),
            "square_method": "crop",
        },
        "crop",
        True,
        R,
    ),
]


@pytest.mark.parametrize("data, metadata, method, inverse, expected", test_data_square)
def test_square_domain(data, metadata, method, inverse, expected):
    """Test the square_domain."""
    dataset = convert_input_to_xarray_dataset(
        data, None, {**fillvalues_metadata, **metadata}
    )
    if "square_method" in metadata:
        dataset.attrs["square_method"] = metadata["square_method"]
    if "orig_domain" in metadata:
        dataset.attrs["orig_domain"] = metadata["orig_domain"]
    assert_array_equal(
        dimension.square_domain(dataset, method, inverse)["precip_intensity"].values,
        expected,
    )


# square_domain
R = np.ones((4, 2))
test_data_square_w_velocity = [
    # square by padding
    (
        R,
        {"x1": 0, "x2": 2, "y1": 0, "y2": 4, "xpixelsize": 1, "ypixelsize": 1},
        "pad",
        False,
        np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
        np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]),
    )
]


@pytest.mark.parametrize(
    "data, metadata, method, inverse, expected, expected_velqual",
    test_data_square_w_velocity,
)
def test_square_w_velocity(data, metadata, method, inverse, expected, expected_velqual):
    """Test the square_domain."""
    dataset = convert_input_to_xarray_dataset(
        data, None, {**fillvalues_metadata, **metadata}
    )
    dataset = dataset.assign(
        {
            "velocity_x": (("y", "x"), data),
            "velocity_y": (("y", "x"), data),
            "quality": (("y", "x"), data),
        }
    )
    if "square_method" in metadata:
        dataset.attrs["square_method"] = metadata["square_method"]
    if "orig_domain" in metadata:
        dataset.attrs["orig_domain"] = metadata["orig_domain"]
    assert_array_equal(
        dimension.square_domain(dataset, method, inverse)["precip_intensity"].values,
        expected,
    )
    assert_array_equal(
        dimension.square_domain(dataset, method, inverse)["velocity_x"].values,
        expected_velqual,
    )
    assert_array_equal(
        dimension.square_domain(dataset, method, inverse)["velocity_y"].values,
        expected_velqual,
    )
    assert_array_equal(
        dimension.square_domain(dataset, method, inverse)["quality"].values,
        expected_velqual,
    )
