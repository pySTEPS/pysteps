# -*- coding: utf-8 -*-
"""
pysteps.nowcasts.extrapolation
==============================

Implementation of extrapolation-based nowcasting methods.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import time

from pysteps import extrapolation


def forecast(
    precip,
    velocity,
    timesteps,
    extrap_method="semilagrangian",
    extrap_kwargs=None,
    measure_time=False,
):
    """
    Generate a nowcast by applying a simple advection-based extrapolation to
    the given precipitation field.

    .. _ndarray: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

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
    extrap_method: str, optional
      Name of the extrapolation method to use. See the documentation of
      pysteps.extrapolation.interface.
    extrap_kwargs: dict, optional
      Optional dictionary that is expanded into keyword arguments for the
      extrapolation method.
    measure_time: bool, optional
      If True, measure, print, and return the computation time.

    Returns
    -------
    out: ndarray_
      Three-dimensional array of shape (num_timesteps, m, n) containing a time
      series of nowcast precipitation fields. The time series starts from
      t0 + timestep, where timestep is taken from the advection field velocity.
      If *measure_time* is True, the return value is a two-element tuple
      containing this array and the computation time (seconds).

    See also
    --------
    pysteps.extrapolation.interface
    """

    _check_inputs(precip, velocity, timesteps)

    if extrap_kwargs is None:
        extrap_kwargs = dict()
    else:
        extrap_kwargs = extrap_kwargs.copy()

    if measure_time:
        print(
            "Computing extrapolation nowcast from a "
            f"{precip.shape[0]:d}x{precip.shape[1]:d} input grid... ",
            end="",
        )

    if measure_time:
        start_time = time.time()

    extrapolation_method = extrapolation.get_method(extrap_method)

    precip_forecast = extrapolation_method(precip, velocity, timesteps, **extrap_kwargs)

    if measure_time:
        computation_time = time.time() - start_time
        print(f"{computation_time:.2f} seconds.")

    if measure_time:
        return precip_forecast, computation_time
    else:
        return precip_forecast


def _check_inputs(precip, velocity, timesteps):
    if precip.ndim != 2:
        raise ValueError("The input precipitation must be a " "two-dimensional array")
    if velocity.ndim != 3:
        raise ValueError("Input velocity must be a three-dimensional array")
    if precip.shape != velocity.shape[1:3]:
        raise ValueError(
            "Dimension mismatch between "
            "input precipitation and velocity: "
            + "shape(precip)=%s, shape(velocity)=%s"
            % (str(precip.shape), str(velocity.shape))
        )
    if isinstance(timesteps, list) and not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")
