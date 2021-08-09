# -*- coding: utf-8 -*-
"""
pysteps.nowcasts.lagrangian_probability
======================================

Implementation of the local Lagrangian probability nowcasting technique
described in :cite:`GZ2004`.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import numpy as np
from scipy.signal import convolve

from pysteps.nowcasts import extrapolation


def forecast(
    precip,
    velocity,
    timesteps,
    threshold,
    extrap_method="semilagrangian",
    extrap_kwargs=None,
    slope=5,
):
    """
    Generate a probability nowcast by a local lagrangian approach. The ouput is
    the probability of exceeding a given intensity threshold, i.e.
    P(precip>=threshold).

    Parameters
    ----------
    precip: array_like
       Two-dimensional array of shape (m,n) containing the input precipitation
       field.
    velocity: array_like
       Array of shape (2,m,n) containing the x- and y-components of the
       advection field. The velocities are assumed to represent one time step
       between the inputs.
    timesteps: int or list of floats
       Number of time steps to forecast or a sorted list of time steps for which
       the forecasts are computed (relative to the input time step).
       The number of time steps has to be a positive integer.
       The elements of the list are required to be in ascending order.
    threshold: float
       Intensity threshold for which the exceedance probabilities are computed.
    slope: float, optional
       The slope of the relationship between optimum scale and lead time in
       pixels / timestep. Germann and Zawadzki (2004) found the optimal slope
       to be equal to 1 km / minute.

    Returns
    -------
    out: ndarray
        Three-dimensional array of shape (num_timesteps, m, n) containing a time
        series of nowcast exceedence probabilities. The time series starts from
        t0 + timestep, where timestep is taken from the advection field velocity.

    References
    ----------
    Germann, U. and I. Zawadzki, 2004:
    Scale Dependence of the Predictability of Precipitation from Continental
    Radar Images. Part II: Probability Forecasts.
    Journal of Applied Meteorology, 43(1), 74-89.
    """
    # Compute deterministic extrapolation forecast
    if isinstance(timesteps, int) and timesteps > 0:
        timesteps = np.arange(1, timesteps + 1)
    elif not isinstance(timesteps, list):
        raise ValueError(f"invalid value for argument 'timesteps': {timesteps}")
    precip_forecast = extrapolation.forecast(
        precip,
        velocity,
        timesteps,
        extrap_method,
        extrap_kwargs,
    )

    # Ignore missing values
    nanmask = np.isnan(precip_forecast)
    precip_forecast[nanmask] = threshold - 1
    valid_pixels = (~nanmask).astype(float)

    # Compute exceedance probabilities using a neighborhood approach
    precip_forecast = (precip_forecast >= threshold).astype(float)
    for i, timestep in enumerate(timesteps):
        scale = timestep * slope
        if scale == 0:
            continue
        kernel = _get_kernel(scale)
        kernel_sum = convolve(
            valid_pixels[i, ...],
            kernel,
            mode="same",
        )
        precip_forecast[i, ...] = convolve(
            precip_forecast[i, ...],
            kernel,
            mode="same",
        )
        precip_forecast[i, ...] /= kernel_sum
    precip_forecast = np.clip(precip_forecast, 0, 1)
    precip_forecast[nanmask] = np.nan
    return precip_forecast


def _get_kernel(size):
    """
    Generate a circular kernel.

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
