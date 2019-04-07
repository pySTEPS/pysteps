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

from datetime import datetime

import numpy as np
import pytest
from scipy.ndimage import uniform_filter

import pysteps as stp
from pysteps import motion, io, rcparams
from pysteps.motion.vet import morph


################################################################################
# Synthetic precipitation observations
# ------------------------------------
#
# Here we create a series of precipitation fields by applying the ideal
# motion field to the reference precipitation field "n" times.
#
# To evaluate the accuracy of the computed_motion vectors, we will use
# a relative RMSE measure.
# Relative MSE = <(expected_motion - computed_motion)^2> / <expected_motion^2>

# Relative RMSE = Rel_RMSE = sqrt(Relative MSE)
#
# - Rel_RMSE = 0%: no error
# - Rel_RMSE = 100%: The retrieved motion field has an average error equal in
#   magnitude to the motion field.
#
# Relative RMSE is computed only un a region surrounding the precipitation
# field, were we have enough information to retrieve the motion field.
# The precipitation region includes the precipitation pattern plus a margin of
# approximately 20 grid points.
from pysteps.tests.helpers import get_precipitation_fields


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
            - rotor: rotor field

    Returns
    -------
    ideal_motion : numpy array (u, v)
    """

    # Create an imaginary grid on the image and create a motion field to be
    # applied to the image.
    ny, nx = input_precip.shape

    x_pos = np.arange(nx)
    y_pos = np.arange(ny)
    x, y = np.meshgrid(x_pos, y_pos, indexing='ij')

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
    morphed_field, mask = morph(input_precip.swapaxes(0, 1),
                                ideal_motion.swapaxes(1, 2))

    mask = np.array(mask, dtype=bool)

    synthetic_observations = np.ma.MaskedArray(morphed_field, mask=mask)
    synthetic_observations = synthetic_observations[np.newaxis, :]

    for t in range(1, num_times):
        morphed_field, mask = morph(synthetic_observations[t - 1],
                                    ideal_motion.swapaxes(1, 2))
        mask = np.array(mask, dtype=bool)

        morphed_field = np.ma.MaskedArray(morphed_field[np.newaxis, :],
                                          mask=mask[np.newaxis, :])

        synthetic_observations = np.ma.concatenate([synthetic_observations, morphed_field],
                                                   axis=0)

    # Swap  back to (lat, lon)
    synthetic_observations = synthetic_observations.swapaxes(1, 2)

    synthetic_observations = np.ma.masked_invalid(synthetic_observations)

    synthetic_observations.data[np.ma.getmaskarray(synthetic_observations)] = 0

    return ideal_motion, synthetic_observations


reference_field = get_precipitation_fields(num_prev_files=0)
arg_values = [(reference_field, 'lk', 'linear_x', 2, 0.1),
              (reference_field, 'lk', 'linear_y', 2, 0.1),
              (reference_field, 'lk', 'linear_x', 3, 0.1),
              (reference_field, 'lk', 'linear_y', 3, 0.1),
              (reference_field, 'vet', 'linear_x', 2, 6),
              (reference_field, 'vet', 'linear_y', 2, 6),
              (reference_field, 'vet', 'linear_x', 3, 5),
              (reference_field, 'vet', 'linear_y', 3, 5),
              (reference_field, 'darts', 'linear_x', 9, 25),
              (reference_field, 'darts', 'linear_y', 9, 25)]

arg_names = ("input_precip, optflow_method_name, "
             "motion_type, num_times, max_rel_rmse")


@pytest.mark.parametrize(arg_names, arg_values)
def test_optflow_method_convergence(input_precip,
                                    optflow_method_name,
                                    motion_type,
                                    num_times,
                                    max_rel_rmse):
    """
    Test the convergence to the actual solution of the optical flow method used.

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

    ideal_motion, precip_obs = _create_observations(input_precip.copy(),
                                                    motion_type,
                                                    num_times=num_times)

    oflow_method = motion.get_method(optflow_method_name)

    computed_motion = oflow_method(precip_obs, verbose=False)

    precip_data, _ = stp.utils.dB_transform(precip_obs.max(axis=0),
                                            inverse=True)
    precip_data.data[precip_data.mask] = 0

    precip_mask = ((uniform_filter(precip_data, size=20) > 0.1)
                   & ~precip_obs.mask.any(axis=0))

    # To evaluate the accuracy of the computed_motion vectors, we will use
    # a relative RMSE measure.
    # Relative MSE = < (expected_motion - computed_motion)^2 > / <expected_motion^2 >
    # Relative RMSE = sqrt(Relative MSE)

    mse = ((ideal_motion - computed_motion)[:, precip_mask] ** 2).mean()

    rel_mse = mse / (ideal_motion[:, precip_mask] ** 2).mean()
    rel_rmse = np.sqrt(rel_mse) * 100
    print(f"method:{optflow_method_name} ; "
          f"motion:{motion_type} ; times: {num_times} ; "
          f"rel_rmse:{rel_rmse}%")
    assert rel_rmse < max_rel_rmse
