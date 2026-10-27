"""Fixed-flow trajectory caching for Eulerian cascade evolution.

Zero outside the domain represents a zero normalized anomaly. Innovations
are added after advection, including at inflow pixels.
"""

import numpy as np
from scipy.ndimage import map_coordinates

from pysteps.extrapolation.semilagrangian import extrapolate


def prepare_advection_coordinates(velocity, ar_order=2):
    """Cache each lag using integer unit integration, in scipy (y, x) order."""
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
        coordinates[lag] = (xy + displacement)[::-1].copy()
        coordinates[lag].setflags(write=False)
    return coordinates


def advect_field(field, coordinates):
    """Sample a finite 2D anomaly with the reference constant-zero boundary."""
    if not np.isfinite(field).all():
        raise ValueError("field contains non-finite values")
    return map_coordinates(
        field, coordinates, order=1, mode="constant", cval=0.0, prefilter=False
    )
