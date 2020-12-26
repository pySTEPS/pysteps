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

from pysteps.extrapolation.semilagrangian import extrapolate
from pysteps.nowcasts.extrapolation import _check_inputs
from pysteps.postprocessing.ensemblestats import excprob

def forecast(
    precip,
    velocity,
    timesteps,
    slope,
    nsamples,
    prob_thr,
    method,
):
    """Generate a probability nowcast by ...

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
    slope: float
      Slope (pixels / timestep) specifying the optimum spatial scale as a function of lead time.
    nsamples: int
      Number of realizations.
    prob_thr: float
        Intensity threshold for which the exceedance probabilities are computed.
    method: {"unit_circle", "gaussian_mv"}

    Returns
    -------
    out: ndarray_
      Three-dimensional array of shape (num_timesteps, m, n) containing a time
      series of nowcast exceedence probabilities. The time series starts from
      t0 + timestep, where timestep is taken from the advection field velocity.

    See also
    --------
    pysteps.extrapolation.interface

    """

    _check_inputs(precip, velocity, timesteps)

    if isinstance(timesteps, int):
        timesteps = np.arange(1, timesteps + 1)
    elif np.any(np.diff(timesteps) <= 0.0):
        raise ValueError("the given timestep sequence is not monotonously increasing")

    x_values, y_values = np.meshgrid(
        np.arange(precip.shape[1]), np.arange(precip.shape[0])
    )
    xy_coords = np.stack([x_values, y_values])

    prob_out = np.zeros((len(timesteps), *precip.shape))
    for sample in range(nsamples):
        precip_extrap = []
        for n, timestep in enumerate(timesteps):
            if method == "unit_circle":
                displacement = _sample_unit_circle(*precip.shape) * timestep * slope
            else:
                displacement = _sample_mv_gaussian(*precip.shape) * timestep * slope
            xy_coords_ = xy_coords + displacement
            precip_extrap.append(
                extrapolate(
                    precip,
                    velocity,
                    [timestep],
                    outval="min",
                    xy_coords=xy_coords_,
                )[0]
            )
        prob_out += np.stack(precip_extrap) >= prob_thr
    return prob_out / nsamples


def _sample_unit_circle(m, n):
    """
    Source
    ------
    https://stackoverflow.com/questions/46996866/sampling-uniformly-within-the-unit-circle

    Parameters
    ----------
    m: float
        Number of rows.
    n: float
        Number of columns.
    """
    length = np.sqrt(np.random.uniform(0, 1, size=(m, n)))
    angle = np.pi * np.random.uniform(0, 2, size=(m, n))
    dx = length * np.cos(angle)
    dy = length * np.sin(angle)
    return np.stack((dx, dy))


def _sample_mv_gaussian(m, n):
    """
    Parameters
    ----------
    m: float
        Number of rows.
    n: float
        Number of columns.
    """
    mean = [0, 0]
    cov = np.array([[1, 0], [0, 1]])
    return np.random.multivariate_normal(mean, cov, (n, m)).T / 1.96
