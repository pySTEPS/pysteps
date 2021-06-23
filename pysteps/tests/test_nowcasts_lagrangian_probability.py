# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np

from pysteps.nowcasts.lagrangian_probability import forecast
from pysteps.tests.helpers import get_precipitation_fields
from pysteps.motion.lucaskanade import dense_lucaskanade


def test_numerical_example():
    """"""
    precip = np.zeros((20, 20))
    precip[5:10, 5:10] = 1
    velocity = np.zeros((2, *precip.shape))
    timesteps = 4
    thr = 0.5
    slope = 1  # pixels / timestep

    # compute probability forecast
    fct = forecast(precip, velocity, timesteps, thr, slope=slope)

    assert fct.ndim == 3
    assert fct.shape[0] == timesteps
    assert fct.shape[1:] == precip.shape
    assert fct.max() <= 1.0
    assert fct.min() >= 0.0


def test_real_case():
    """"""
    # inputs
    precip, metadata = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        metadata=True,
        upscale=2000,
    )

    precip_obs = get_precipitation_fields(
        num_prev_files=0, num_next_files=3, return_raw=False, upscale=2000
    )[1:, :, :]

    # motion
    motion = dense_lucaskanade(precip)

    # parameters
    nleadtimes = 6
    timesteps = [1, 2, 3]
    thr = 1  # mm / h
    slope = 1 * metadata["accutime"]  # min-1

    # compute probability forecast
    extrap_kwargs = dict(allow_nonfinite_values=True)
    fct = forecast(
        precip[-1], motion, timesteps, thr, slope=slope, extrap_kwargs=extrap_kwargs
    )

    assert fct.ndim == 3
    assert fct.shape[0] == len(timesteps)
    assert fct.shape[1:] == precip.shape[1:]
    assert np.nanmax(fct) <= 1.0
    assert np.nanmin(fct) >= 0.0
