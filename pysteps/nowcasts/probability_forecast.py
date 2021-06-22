# -*- coding: utf-8 -*-
"""

pysteps.nowcasts.probability_forecasts
======================================

Implementation of probability nowcasting methods.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import numpy as np
from scipy import signal

from pysteps.nowcasts import extrapolation


def forecast(
    precip,
    velocity,
    timesteps,
    threshold,
    extrap_method="semilagrangian",
    extrap_kwargs=None,
    neighborhood_slope=5,
):
    """Generate a probability nowcast by a local lagrangian approach.

    Parameters
    ----------
    precip: array-like
      Two-dimensional array of shape (m,n) containing the input precipitation
      field.
    velocity: array-like
      Array of shape (2,m,n) containing the x- and y-components of the
      advection field. The velocities are assumed to represent one time step
      between the inputs.
    timesteps: int or list of floats
      Number of time steps to forecast or a list of time steps for which the
      forecasts are computed (relative to the input time step). The elements of
      the list are required to be in ascending order.
    threshold: float
      Intensity threshold for which the exceedance probabilities are computed.
   neighborhood_slope: float, optional
      Slope (pixels / timestep) specifying the spatial scale of a function
      of lead time.

    Returns
    -------
    out: ndarray_
      Three-dimensional array of shape (num_timesteps, m, n) containing a time
      series of nowcast exceedence probabilities. The time series starts from
      t0 + timestep, where timestep is taken from the advection field velocity.
    """
    if isinstance(timesteps, int):
        timesteps = np.arange(1, timesteps + 1)
    precip_forecast = extrapolation.forecast(
        precip,
        velocity,
        timesteps,
        extrap_method,
        extrap_kwargs,
    )
    nanmask = np.isnan(precip_forecast)
    precip_forecast[nanmask] = np.nanmin(precip_forecast)
    precip_forecast = (precip_forecast > threshold).astype(float)
    valid_pixels = (~nanmask).astype(float)
    for i, timestep in enumerate(timesteps):
        scale = timestep * neighborhood_slope
        if scale == 0: continue
        kernel = _get_kernel(scale)
        kernel_sum = signal.convolve(valid_pixels[i, ], kernel, mode="same")
        precip_forecast[i, ] = signal.convolve(precip_forecast[i, ], kernel, mode="same")
        precip_forecast[i, ] /= kernel_sum
    precip_forecast[nanmask] = np.nan
    return precip_forecast


def _get_kernel(size):
    """Generate a circular kernel.

    Parameters
    ----------
    size : int
        Size of the circular kernel (its diameter). For size < 5, the kernel is
        a square instead of a circle.

    Returns
    -------
    2-D array with kernel values
    """
    middle = max((int(size / 2), 1))
    if size < 5:
        return np.ones((size, size), dtype=np.float32)
    else:
        xx, yy = np.mgrid[:size, :size]
        circle = (xx - middle) ** 2 + (yy - middle) ** 2
        return np.asarray(circle <= (middle ** 2), dtype=np.float32)

