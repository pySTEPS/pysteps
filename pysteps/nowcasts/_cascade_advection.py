"""Fixed-flow trajectory caching for Eulerian cascade evolution.

Zero outside the domain represents a zero normalized anomaly. Innovations
are added after advection, including at inflow pixels.
"""

import numpy as np
from scipy.ndimage import map_coordinates

from pysteps.extrapolation.semilagrangian import extrapolate


def prepare_advection_coordinates(velocity, ar_order=2):
    """Prepare backward trajectories for sampling previous cascade states.

    Parameters
    ----------
    velocity: array_like
        Fixed x and y displacements with shape (2, y, x), in pixels per
        observation interval. All values must be finite.
    ar_order: {1, 2}, optional
        Number of previous states to advect (default 2).

    Returns
    -------
    coordinates: dict
        Read-only arrays of shape (2, y, x), keyed by integer lag. Components
        are in (y, x) order for scipy.ndimage.map_coordinates. The two-step
        trajectory uses two unit integration steps with the same velocity.
    """
    velocity = np.asarray(velocity, dtype=float)
    if (
        velocity.ndim != 3
        or velocity.shape[0] != 2
        or min(velocity.shape[1:]) < 2
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("velocity must be finite with shape (2, y, x)")
    if ar_order not in (1, 2):
        raise ValueError("ar_order must be 1 or 2")
    yy, xx = np.indices(velocity.shape[1:])
    xy = np.stack([xx, yy])
    coordinates = {}
    for lag in range(1, ar_order + 1):
        _, displacement = extrapolate(None, velocity, lag, return_displacement=True)
        # Extrapolation returns (x, y) displacements; SciPy samples in (y, x).
        coordinates[lag] = (xy + displacement)[::-1].copy()
        coordinates[lag].setflags(write=False)
    return coordinates


def advect_field(field, coordinates):
    """Advect a normalized cascade field using backward sampling coordinates.

    Parameters
    ----------
    field: ndarray
        Finite two-dimensional normalized cascade field in (y, x) order.
    coordinates: ndarray
        Sampling positions with shape (2, y, x), in (y, x) component order.

    Returns
    -------
    advected: ndarray
        Linearly interpolated field. Samples outside the grid are zero,
        representing the cascade level's mean. Random innovations are added
        separately during forecasting, including at these inflow pixels.
    """
    if not np.isfinite(field).all():
        raise ValueError("field contains non-finite values")
    return map_coordinates(
        field, coordinates, order=1, mode="constant", cval=0.0, prefilter=False
    )
