# -*- coding: utf-8 -*-
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.nowcasts.lagrangian_probability import forecast
from pysteps.tests.helpers import get_precipitation_fields


def test_numerical_example():
    """"""
    precip = np.zeros((20, 20))
    precip[5:10, 5:10] = 1
    velocity = np.zeros((2, *precip.shape))
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    dataset_input = xr.Dataset(
        data_vars={
            "precip_intensity": (["time", "y", "x"], [precip]),
            "velocity_x": (["y", "x"], velocity[0]),
            "velocity_y": (["y", "x"], velocity[1]),
        },
        coords={"time": (["time"], [now], {"stepsize": 300})},
        attrs={"precip_var": "precip_intensity"},
    )
    timesteps = 4
    thr = 0.5
    slope = 1  # pixels / timestep

    # compute probability forecast
    dataset_forecast = forecast(dataset_input, timesteps, thr, slope=slope)
    fct = dataset_forecast["precip_intensity"].values

    assert fct.ndim == 3
    assert fct.shape[0] == timesteps
    assert fct.shape[1:] == precip.shape
    assert fct.max() <= 1.0
    assert fct.min() >= 0.0

    # slope = 0 should return a binary field
    dataset_forecast = forecast(dataset_input, timesteps, thr, slope=0)
    fct = dataset_forecast["precip_intensity"].values
    ref = (np.repeat(precip[None, ...], timesteps, axis=0) >= thr).astype(float)
    assert np.allclose(fct, fct.astype(bool))
    assert np.allclose(fct, ref)


def test_numerical_example_with_float_slope_and_float_list_timesteps():
    """"""
    precip = np.zeros((20, 20))
    precip[5:10, 5:10] = 1
    velocity = np.zeros((2, *precip.shape))
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    dataset_input = xr.Dataset(
        data_vars={
            "precip_intensity": (["time", "y", "x"], [precip]),
            "velocity_x": (["y", "x"], velocity[0]),
            "velocity_y": (["y", "x"], velocity[1]),
        },
        coords={"time": (["time"], [now], {"stepsize": 300})},
        attrs={"precip_var": "precip_intensity"},
    )
    timesteps = [1.0, 2.0, 5.0, 12.0]
    thr = 0.5
    slope = 1.0  # pixels / timestep

    # compute probability forecast
    dataset_forecast = forecast(dataset_input, timesteps, thr, slope=slope)
    fct = dataset_forecast["precip_intensity"].values

    assert fct.ndim == 3
    assert fct.shape[0] == len(timesteps)
    assert fct.shape[1:] == precip.shape
    assert fct.max() <= 1.0
    assert fct.min() >= 0.0


def test_real_case():
    """"""
    pytest.importorskip("cv2")

    # inputs
    dataset_input = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        metadata=True,
        upscale=2000,
    )
    precip_var = dataset_input.attrs["precip_var"]
    metadata = dataset_input[precip_var].attrs

    # motion
    dataset_w_motion = dense_lucaskanade(dataset_input)

    # parameters
    timesteps = [1, 2, 3]
    thr = 1  # mm / h
    slope = 1 * metadata["accutime"]  # min-1

    # compute probability forecast
    extrap_kwargs = dict(allow_nonfinite_values=True)
    dataset_forecast = forecast(
        dataset_w_motion.isel(time=slice(-1, None, None)),
        timesteps,
        thr,
        slope=slope,
        extrap_kwargs=extrap_kwargs,
    )
    fct = dataset_forecast["precip_intensity"].values

    assert fct.ndim == 3
    assert fct.shape[0] == len(timesteps)
    assert fct.shape[1:] == dataset_input[precip_var].values.shape[1:]
    assert np.nanmax(fct) <= 1.0
    assert np.nanmin(fct) >= 0.0


def test_wrong_inputs():
    # dummy inputs
    precip = np.zeros((3, 3))
    velocity = np.zeros((2, *precip.shape))
    dataset_input = xr.Dataset(
        data_vars={
            "precip_intensity": (["y", "x"], precip),
            "velocity_x": (["y", "x"], velocity[0]),
            "velocity_y": (["y", "x"], velocity[1]),
        },
        attrs={"precip_var": "precip_intensity"},
    )

    # timesteps must be > 0
    with pytest.raises(ValueError):
        forecast(dataset_input, 0, 1)

    # timesteps must be a sorted list
    with pytest.raises(ValueError):
        forecast(dataset_input, [2, 1], 1)
