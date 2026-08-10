# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.blending.linear_blending import _get_ranked_salience, _get_ws, forecast
from pysteps.utils import transformation

# Test function arguments
linear_arg_values = [
    (5, 30, 60, 20, 45, "eulerian", None, 1, False, True, False),
    (5, 30, 60, 20, 45, "eulerian", None, 2, False, False, False),
    (5, 30, 60, 20, 45, "eulerian", None, 0, False, False, False),
    (4, 23, 33, 9, 28, "eulerian", None, 1, False, False, False),
    (3, 18, 36, 13, 27, "eulerian", None, 1, False, False, False),
    (7, 30, 68, 11, 49, "eulerian", None, 1, False, False, False),
    (7, 30, 68, 11, 49, "eulerian", None, 1, False, False, True),
    (10, 100, 160, 25, 130, "eulerian", None, 1, False, False, False),
    (6, 60, 180, 22, 120, "eulerian", None, 1, False, False, False),
    (5, 100, 200, 40, 150, "eulerian", None, 1, False, False, False),
    (
        5,
        30,
        60,
        20,
        45,
        "extrapolation",
        np.zeros((2, 200, 200)),
        1,
        False,
        False,
        False,
    ),
    (
        4,
        23,
        33,
        9,
        28,
        "extrapolation",
        np.zeros((2, 200, 200)),
        1,
        False,
        False,
        False,
    ),
    (
        3,
        18,
        36,
        13,
        27,
        "extrapolation",
        np.zeros((2, 200, 200)),
        1,
        False,
        False,
        False,
    ),
    (
        7,
        30,
        68,
        11,
        49,
        "extrapolation",
        np.zeros((2, 200, 200)),
        1,
        False,
        False,
        False,
    ),
    (
        10,
        100,
        160,
        25,
        130,
        "extrapolation",
        np.zeros((2, 200, 200)),
        1,
        False,
        False,
        False,
    ),
    (
        6,
        60,
        180,
        22,
        120,
        "extrapolation",
        np.zeros((2, 200, 200)),
        1,
        False,
        False,
        False,
    ),
    (
        5,
        100,
        200,
        40,
        150,
        "extrapolation",
        np.zeros((2, 200, 200)),
        1,
        False,
        False,
        False,
    ),
    (
        5,
        100,
        200,
        40,
        150,
        "extrapolation",
        np.zeros((2, 200, 200)),
        1,
        False,
        False,
        True,
    ),
    (5, 30, 60, 20, 45, "eulerian", None, 1, True, True, False),
    (5, 30, 60, 20, 45, "eulerian", None, 2, True, False, False),
    (5, 30, 60, 20, 45, "eulerian", None, 0, True, False, False),
    (
        5,
        30,
        60,
        20,
        45,
        "extrapolation",
        np.zeros((2, 200, 200)),
        1,
        True,
        False,
        False,
    ),
    (4, 23, 33, 9, 28, "extrapolation", np.zeros((2, 200, 200)), 1, True, False, False),
    (
        3,
        18,
        36,
        13,
        27,
        "extrapolation",
        np.zeros((2, 200, 200)),
        1,
        True,
        False,
        False,
    ),
]


@pytest.mark.parametrize(
    "timestep, start_blending, end_blending, n_timesteps, controltime, nowcast_method, V, n_models, salient_blending, squeeze_nwp_array, fill_nwp",
    linear_arg_values,
)
def test_linear_blending(
    timestep,
    start_blending,
    end_blending,
    n_timesteps,
    controltime,
    nowcast_method,
    V,
    n_models,
    salient_blending,
    squeeze_nwp_array,
    fill_nwp,
):
    """Tests if the linear blending function is correct. For the nowcast data a precipitation field
    which is constant over time is taken. One half of the field has no rain and the other half
    has a set value. For the NWP data a similar field is taken, the only difference
    being that now the other half of the field is zero. The blended field should have a
    constant value over the entire field at the timestep right in the middle between the start
    of the blending and the end of the blending. This assertion is checked to see if the
    linear blending function works well."""

    # The argument controltime gives the timestep at which the field is assumed to be
    # entirely constant

    # Assert that the control time step is in the range of the forecasted time steps
    assert controltime <= (
        n_timesteps * timestep
    ), "Control time needs to be within reach of forecasts, controltime = {} and n_timesteps = {}".format(
        controltime, n_timesteps
    )

    # Assert that the start time of the blending comes before the end time of the blending
    assert (
        start_blending < end_blending
    ), "Start time of blending needs to be smaller than end time of blending"

    # Assert that the control time is a multiple of the time step
    assert (
        not controltime % timestep
    ), "Control time needs to be a multiple of the time step"

    # Initialise dummy NWP data
    if n_models == 0:
        r_nwp = None
    else:
        r_nwp = np.zeros((n_models, n_timesteps, 200, 200))

        for i in range(100):
            r_nwp[:, :, i, :] = 11.0

        if squeeze_nwp_array:
            r_nwp = np.squeeze(r_nwp)

    # Define nowcast input data (alternate between 2D and 3D arrays for testing)
    if timestep % 2 == 0:
        r_input = np.zeros((4, 200, 200))
        for i in range(100, 200):
            r_input[:, i, :] = 11.0
    else:
        r_input = np.zeros((200, 200))
        for i in range(100, 200):
            r_input[i, :] = 11.0

    # Transform from mm/h to dB
    r_input, _ = transformation.dB_transform(
        r_input, None, threshold=0.1, zerovalue=-15.0
    )
    if len(r_input.shape) == 3:
        r_input = (
            r_input[-1]
            if nowcast_method
            in [
                "extrapolation",
                "eulerian",
                "lagrangian",
                "lagrangian_probability",
                "probability",
            ]
            else r_input
        )

    # Calculate the blended field
    r_blended = forecast(
        r_input,
        dict({"unit": "mm/h", "transform": "dB"}),
        V,
        n_timesteps,
        timestep,
        nowcast_method,
        r_nwp,
        dict({"unit": "mm/h", "transform": None}),
        start_blending=start_blending,
        end_blending=end_blending,
        fill_nwp=fill_nwp,
        saliency=salient_blending,
    )

    # Assert that the blended field has the expected dimension
    if n_models > 1:
        assert r_blended.shape == (
            n_models,
            n_timesteps,
            200,
            200,
        ), "The shape of the blended array does not have the expected value. The shape is {}".format(
            r_blended.shape
        )
    else:
        assert r_blended.shape == (
            n_timesteps,
            200,
            200,
        ), "The shape of the blended array does not have the expected value. The shape is {}".format(
            r_blended.shape
        )

    # Assert that the blended field at the control time step is equal to
    # a constant field with the expected value.
    if salient_blending == False:
        if n_models > 1:
            assert_array_almost_equal(
                r_blended[0, controltime // timestep - 1],
                np.ones((200, 200)) * 5.5,
                err_msg="The blended array does not have the expected value",
            )
        elif n_models > 0:
            assert_array_almost_equal(
                r_blended[controltime // timestep - 1],
                np.ones((200, 200)) * 5.5,
                err_msg="The blended array does not have the expected value",
            )


def test_linear_blending_ar_nowcast_3d_precip():
    """Regression test: AR-based nowcast methods (e.g. sprog) require several past
    precipitation fields (shape (ar_order + 1, m, n)) rather than a single 2D field.
    forecast() used to always collapse a 3D precip input down to its last frame,
    which broke these methods. This test checks that a 3D precip stack is passed
    through to the nowcast method unmodified."""

    np.random.seed(42)

    timestep = 5
    n_timesteps = 5

    r_input = np.random.rand(3, 32, 32) * 5.0
    r_input, _ = transformation.dB_transform(
        r_input, None, threshold=0.1, zerovalue=-15.0
    )
    velocity = np.zeros((2, 32, 32))
    r_nwp = np.random.rand(n_timesteps, 32, 32) * 5.0

    r_blended = forecast(
        r_input,
        dict({"unit": "mm/h", "transform": "dB"}),
        velocity,
        n_timesteps,
        timestep,
        "sprog",
        r_nwp,
        dict({"unit": "mm/h", "transform": None}),
        start_blending=10,
        end_blending=20,
        nowcast_kwargs=dict(n_cascade_levels=4, precip_thr=-10.0),
    )

    assert r_blended.shape == (
        n_timesteps,
        32,
        32,
    ), "The shape of the blended array does not have the expected value. The shape is {}".format(
        r_blended.shape
    )
    assert np.all(
        np.isfinite(r_blended)
    ), "The blended array contains non-finite values"


def test_linear_blending_timesteps_nowcast_clamp():
    """Regression test: when end_blending / timestep exceeds the requested number of
    timesteps, the nowcast used to be computed for more timesteps than precip_nwp
    (and thus precip_blended) has. Combined with fill_nwp and NaNs in the nowcast,
    this caused a shape mismatch when filling in the NWP data. timesteps_nowcast is
    now clamped to timesteps to prevent this."""

    timestep = 5
    n_timesteps = 5

    r_input = np.random.rand(32, 32) * 5.0
    r_input, _ = transformation.dB_transform(
        r_input, None, threshold=0.1, zerovalue=-15.0
    )
    r_input[0, 0] = np.nan
    velocity = np.zeros((2, 32, 32))
    r_nwp = np.random.rand(n_timesteps, 32, 32) * 5.0

    # end_blending / timestep = 10, which is larger than n_timesteps = 5
    r_blended = forecast(
        r_input,
        dict({"unit": "mm/h", "transform": "dB"}),
        velocity,
        n_timesteps,
        timestep,
        "eulerian",
        r_nwp,
        dict({"unit": "mm/h", "transform": None}),
        start_blending=10,
        end_blending=50,
        fill_nwp=True,
    )

    assert r_blended.shape == (
        n_timesteps,
        32,
        32,
    ), "The shape of the blended array does not have the expected value. The shape is {}".format(
        r_blended.shape
    )
    assert np.all(
        np.isfinite(r_blended)
    ), "The blended array contains non-finite values after filling with NWP data"


ranked_salience_values = [
    (np.ones((200, 200)), np.ones((200, 200)), 0.9),
    (np.zeros((200, 200)), np.random.rand(200, 200), 0.7),
    (np.random.rand(200, 200), np.random.rand(200, 200), 0.5),
]


@pytest.mark.parametrize(
    "nowcast, nwp, weight_nowcast",
    ranked_salience_values,
)
def test_salient_weight(
    nowcast,
    nwp,
    weight_nowcast,
):
    ranked_salience = _get_ranked_salience(nowcast, nwp)
    ws = _get_ws(weight_nowcast, ranked_salience)

    assert np.min(ws) >= 0, "Negative value for the ranked saliency output"
    assert np.max(ws) <= 1, "Too large value for the ranked saliency output"

    assert ws.shape == (
        200,
        200,
    ), "The shape of the ranked salience array does not have the expected value. The shape is {}".format(
        ws.shape
    )
