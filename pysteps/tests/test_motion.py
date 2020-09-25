# coding: utf-8

"""
Test the convergence of the optical flow methods available in
pySTEPS using idealized motion fields.

To test the convergence, using an example precipitation field we will:

- Read precipitation field from a file
- Morph the precipitation field using a given motion field (linear or rotor) to
  generate a sequence of moving precipitation patterns.
- Using the available optical flow methods, retrieve the motion field from the
  precipitation time sequence (synthetic precipitation observations).

This tests check that the retrieved motion fields are within reasonable values.
Also, they will fail if any modification on the code decrease the quality of
the retrieval.
"""

from contextlib import contextmanager

import numpy as np
import pytest
from functools import partial
from scipy.ndimage import uniform_filter

import pysteps as stp
from pysteps import motion
from pysteps.motion.vet import morph
from pysteps.tests.helpers import get_precipitation_fields


@contextmanager
def not_raises(_exception):
    try:
        yield
    except _exception:
        raise pytest.fail("DID RAISE {0}".format(_exception))


reference_field = get_precipitation_fields(num_prev_files=0)


def _create_motion_field(input_precip, motion_type):
    """
    Create idealized motion fields to be applied to the reference image.

    Parameters
    ----------

    input_precip: numpy array (lat, lon)

    motion_type : str
        The supported motion fields are:

            - linear_x: (u=2, v=0)
            - linear_y: (u=0, v=2)

    Returns
    -------
    ideal_motion : numpy array (u, v)
    """

    # Create an imaginary grid on the image and create a motion field to be
    # applied to the image.
    ny, nx = input_precip.shape

    ideal_motion = np.zeros((2, nx, ny))

    if motion_type == "linear_x":
        ideal_motion[0, :] = 2  # Motion along x
    elif motion_type == "linear_y":
        ideal_motion[1, :] = 2  # Motion along y
    else:
        raise ValueError("motion_type not supported.")

    # We need to swap the axes because the optical flow methods expect
    # (lat, lon) or (y,x) indexing convention.
    ideal_motion = ideal_motion.swapaxes(1, 2)
    return ideal_motion


def _create_observations(input_precip, motion_type, num_times=9):
    """
    Create synthetic precipitation observations by displacing the input field
    using an ideal motion field.

    Parameters
    ----------

    input_precip: numpy array (lat, lon)
        Input precipitation field.

    motion_type : str
        The supported motion fields are:

            - linear_x: (u=2, v=0)
            - linear_y: (u=0, v=2)

    num_times: int, optional
        Length of the observations sequence.


    Returns
    -------
    synthetic_observations : numpy array
        Sequence of observations
    """

    ideal_motion = _create_motion_field(input_precip, motion_type)

    # The morph function expects (lon, lat) or (x, y) dimensions.
    # Hence, we need to swap the lat,lon axes.

    # NOTE: The motion field passed to the morph function can't have any NaNs.
    # Otherwise, it can produce a segmentation fault.
    morphed_field, mask = morph(
        input_precip.swapaxes(0, 1), ideal_motion.swapaxes(1, 2)
    )

    mask = np.array(mask, dtype=bool)

    synthetic_observations = np.ma.MaskedArray(morphed_field, mask=mask)
    synthetic_observations = synthetic_observations[np.newaxis, :]

    for t in range(1, num_times):
        morphed_field, mask = morph(
            synthetic_observations[t - 1], ideal_motion.swapaxes(1, 2)
        )
        mask = np.array(mask, dtype=bool)

        morphed_field = np.ma.MaskedArray(
            morphed_field[np.newaxis, :], mask=mask[np.newaxis, :]
        )

        synthetic_observations = np.ma.concatenate(
            [synthetic_observations, morphed_field], axis=0
        )

    # Swap  back to (lat, lon)
    synthetic_observations = synthetic_observations.swapaxes(1, 2)

    synthetic_observations = np.ma.masked_invalid(synthetic_observations)

    synthetic_observations.data[np.ma.getmaskarray(synthetic_observations)] = 0

    return ideal_motion, synthetic_observations


convergence_arg_names = (
    "input_precip, optflow_method_name, motion_type, " "num_times, max_rel_rmse"
)

convergence_arg_values = [
    (reference_field, "lk", "linear_x", 2, 0.1),
    (reference_field, "lk", "linear_y", 2, 0.1),
    (reference_field, "lk", "linear_x", 3, 0.1),
    (reference_field, "lk", "linear_y", 3, 0.1),
    (reference_field, "vet", "linear_x", 2, 0.1),
    # (reference_field, 'vet', 'linear_x', 3, 9),
    # (reference_field, 'vet', 'linear_y', 2, 9),
    (reference_field, "vet", "linear_y", 3, 0.1),
    (reference_field, "proesmans", "linear_x", 2, 0.45),
    (reference_field, "proesmans", "linear_y", 2, 0.45),
    (reference_field, "darts", "linear_x", 9, 20),
    (reference_field, "darts", "linear_y", 9, 20),
]


@pytest.mark.parametrize(convergence_arg_names, convergence_arg_values)
def test_optflow_method_convergence(
    input_precip, optflow_method_name, motion_type, num_times, max_rel_rmse
):
    """
    Test the convergence to the actual solution of the optical flow method used.

    We measure the error in the retrieved field by using the
    Relative RMSE = Rel_RMSE = sqrt(Relative MSE)

        - Rel_RMSE = 0%: no error
        - Rel_RMSE = 100%: The retrieved motion field has an average error
          equal in magnitude to the motion field.

    Relative RMSE is computed only un a region surrounding the precipitation
    field, were we have enough information to retrieve the motion field.
    The precipitation region includes the precipitation pattern plus a margin
    of approximately 20 grid points.


    Parameters
    ----------

    input_precip: numpy array (lat, lon)
        Input precipitation field.

    optflow_method_name: str
        Optical flow method name

    motion_type : str
        The supported motion fields are:

            - linear_x: (u=2, v=0)
            - linear_y: (u=0, v=2)
    """
    if optflow_method_name == "lk":
        pytest.importorskip("cv2")

    ideal_motion, precip_obs = _create_observations(
        input_precip.copy(), motion_type, num_times=num_times
    )

    oflow_method = motion.get_method(optflow_method_name)

    if optflow_method_name == "vet":
        # By default, the maximum number of iteration in the VET minimization
        # is maxiter=100.
        # To increase the stability of the tests to we increase this value to
        # maxiter=150.
        retrieved_motion = oflow_method(
            precip_obs, verbose=False, options=dict(maxiter=150)
        )
    elif optflow_method_name == "proesmans":
        retrieved_motion = oflow_method(precip_obs)
    else:

        retrieved_motion = oflow_method(precip_obs, verbose=False)

    precip_data, _ = stp.utils.dB_transform(precip_obs.max(axis=0), inverse=True)
    precip_data.data[precip_data.mask] = 0

    precip_mask = (uniform_filter(precip_data, size=20) > 0.1) & ~precip_obs.mask.any(
        axis=0
    )

    # To evaluate the accuracy of the computed_motion vectors, we will use
    # a relative RMSE measure.
    # Relative MSE = < (expected_motion - computed_motion)^2 > / <expected_motion^2 >
    # Relative RMSE = sqrt(Relative MSE)

    mse = ((ideal_motion - retrieved_motion)[:, precip_mask] ** 2).mean()

    rel_mse = mse / (ideal_motion[:, precip_mask] ** 2).mean()
    rel_rmse = np.sqrt(rel_mse) * 100
    print(
        f"method:{optflow_method_name} ; "
        f"motion:{motion_type} ; times: {num_times} ; "
        f"rel_rmse:{rel_rmse:.2f}%"
    )
    assert rel_rmse < max_rel_rmse


no_precip_args_names = "optflow_method_name, num_times"
no_precip_args_values = [
    ("lk", 2),
    ("lk", 3),
    ("vet", 2),
    ("vet", 3),
    ("darts", 9),
    ("proesmans", 2),
]


@pytest.mark.parametrize(no_precip_args_names, no_precip_args_values)
def test_no_precipitation(optflow_method_name, num_times):
    """
    Test that the motion methods work well with a zero precipitation in the
    domain.

    The expected result is a zero motion vector.

    Parameters
    ----------

    optflow_method_name: str
        Optical flow method name

    num_times : int
        Number of precipitation frames (times) used as input for the optical
        flow methods.
    """
    if optflow_method_name == "lk":
        pytest.importorskip("cv2")
    zero_precip = np.zeros((num_times,) + reference_field.shape)
    motion_method = motion.get_method(optflow_method_name)
    uv_motion = motion_method(zero_precip, verbose=False)

    assert np.abs(uv_motion).max() < 0.01


input_tests_args_names = (
    "optflow_method_name",
    "minimum_input_frames",
    "maximum_input_frames",
)
input_tests_args_values = [
    ("lk", 2, np.inf),
    ("vet", 2, 3),
    ("darts", 9, 9),
    ("proesmans", 2, 2),
]


@pytest.mark.parametrize(input_tests_args_names, input_tests_args_values)
def test_input_shape_checks(
    optflow_method_name, minimum_input_frames, maximum_input_frames
):
    if optflow_method_name == "lk":
        pytest.importorskip("cv2")
    image_size = 100
    motion_method = motion.get_method(optflow_method_name)

    if maximum_input_frames == np.inf:
        maximum_input_frames = minimum_input_frames + 10

    with not_raises(Exception):
        for frames in range(minimum_input_frames, maximum_input_frames + 1):
            motion_method(np.zeros((frames, image_size, image_size)), verbose=False)

    with pytest.raises(ValueError):
        motion_method(np.zeros((2,)))
        motion_method(np.zeros((2, 2)))
        for frames in range(minimum_input_frames):
            motion_method(np.zeros((frames, image_size, image_size)), verbose=False)
        for frames in range(maximum_input_frames + 1, maximum_input_frames + 4):
            motion_method(np.zeros((frames, image_size, image_size)), verbose=False)


def test_vet_padding():
    """
    Test that the padding functionality in vet works correctly with ndarrays and
    masked arrays.
    """

    _, precip_obs = _create_observations(
        reference_field.copy(), "linear_y", num_times=2
    )

    # Use a small region to speed up the test
    precip_obs = precip_obs[:, 200:427, 250:456]
    # precip_obs.shape == (227 , 206)
    # 227 is a prime number ; 206 = 2*103
    # Using this shape will force vet to internally pad the input array for the sector's
    # blocks to divide exactly the input shape.
    # NOTE: This "internal padding" is different from the padding keyword being test next.

    for padding in [0, 3, 10]:
        # No padding
        vet_method = partial(
            motion.get_method("vet"),
            verbose=False,
            sectors=((16, 4, 2), (16, 4, 2)),
            options=dict(maxiter=5),
            padding=padding
            # We use only a few iterations since
            # we don't care about convergence in this test
        )

        assert precip_obs.shape == vet_method(precip_obs).shape
        assert precip_obs.shape == vet_method(np.ma.masked_invalid(precip_obs)).shape


def test_vet_cost_function():
    """
    Test that the vet cost_function computation gives always the same result
    with the same input.

    Useful to test if the parallelization in VET produce undesired results.
    """

    from pysteps.motion import vet

    ideal_motion, precip_obs = _create_observations(
        reference_field.copy(), "linear_y", num_times=2
    )

    mask_2d = np.ma.getmaskarray(precip_obs).any(axis=0).astype("int8")

    returned_values = np.zeros(20)

    for i in range(20):
        returned_values[i] = vet.vet_cost_function(
            ideal_motion.ravel(),  # sector_displacement_1d
            precip_obs.data,  # input_images
            ideal_motion.shape[1:],  # blocks_shape (same as 2D grid)
            mask_2d,  # Mask
            1e6,  # smooth_gain
            debug=False,
        )

    tolerance = 1e-12
    errors = np.abs(returned_values - returned_values[0])
    # errors should contain all zeros
    assert (errors < tolerance).any()
    assert (returned_values[0] - 1548250.87627097) < 0.001


def test_lk_masked_array():
    """
    Passing a ndarray with NaNs or a masked array should produce the same results.
    """
    pytest.importorskip("cv2")

    __, precip_obs = _create_observations(
        reference_field.copy(), "linear_y", num_times=2
    )
    motion_method = motion.get_method("LK")

    # ndarray with nans
    np.ma.set_fill_value(precip_obs, -15)
    ndarray = precip_obs.filled()
    ndarray[ndarray == -15] = np.nan
    uv_ndarray = motion_method(ndarray, fd_kwargs={"buffer_mask": 20}, verbose=False)

    # masked array
    mdarray = np.ma.masked_invalid(ndarray)
    mdarray.data[mdarray.mask] = -15
    uv_mdarray = motion_method(mdarray, fd_kwargs={"buffer_mask": 20}, verbose=False)

    assert np.abs(uv_mdarray - uv_ndarray).max() < 0.01
