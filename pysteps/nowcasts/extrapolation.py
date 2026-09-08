"""
pysteps.nowcasts.extrapolation
==============================

Implementation of extrapolation-based nowcasting methods.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import time

import numpy as np
import xarray as xr

from pysteps import extrapolation
from pysteps.xarray_helpers import convert_output_to_xarray_dataset


def forecast(
    dataset: xr.Dataset,
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
    dataset: xarray.Dataset
        Input dataset as described in the documentation of
        :py:mod:`pysteps.io.importers`. It has to contain the ``velocity_x`` and
        ``velocity_y`` data variables, as well as any pecipitation data variable.
        It should contain a time dimension of size 1.
    timesteps: int or list of floats
        Number of time steps to forecast or a list of time steps for which the
        forecasts are computed (relative to the input time step). The elements
        of the list are required to be in ascending order.
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
    out: xarray.Dataset
        If return_output is True, a dataset as described in the documentation of
        :py:mod:`pysteps.io.importers` is returned containing a time series of forecast
        precipitation fields. Otherwise, a None value
        is returned. The time series starts from t0+timestep, where timestep is
        taken from the metadata of the time coordinate. If measure_time is True, the
        return value is a three-element tuple containing the nowcast dataset, the
        initialization time of the nowcast generator and the time used in the
        main loop (seconds).

    See also
    --------
    pysteps.extrapolation.interface
    """

    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    precip = dataset[precip_var].values[-1]
    velocity = np.stack([dataset["velocity_x"], dataset["velocity_y"]])
    _check_inputs(precip, velocity, timesteps)

    if extrap_kwargs is None:
        extrap_kwargs = dict()
    else:
        extrap_kwargs = extrap_kwargs.copy()

    extrap_kwargs["allow_nonfinite_values"] = (
        True if np.any(~np.isfinite(precip)) else False
    )

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

    output_dataset = convert_output_to_xarray_dataset(
        dataset, timesteps, precip_forecast
    )
    if measure_time:
        return output_dataset, computation_time
    else:
        return output_dataset


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
