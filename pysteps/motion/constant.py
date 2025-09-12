# -*- coding: utf-8 -*-
"""
pysteps.motion.constant
=======================

Implementation of a constant advection field estimation by maximizing the
correlation between two images.

.. autosummary::
    :toctree: ../generated/

    constant
"""

import numpy as np
import scipy.optimize as op
import xarray as xr
from scipy.ndimage import map_coordinates


def constant(dataset: xr.Dataset, **kwargs):
    """
    Compute a constant advection field by finding a translation vector that
    maximizes the correlation between two successive images.

    Parameters
    ----------
    dataset: xarray.Dataset
        Input dataset as described in the documentation of
        :py:mod:`pysteps.io.importers`. It has to contain a precipitation data variable.
        The dataset has to have a time dimension. If the size of this dimension
        is larger than 2, the last 2 entries of this dimension are used.

    Returns
    -------
    out: xarray.Dataset
        The input dataset with the constant advection field added in the ``velocity_x``
        and ``velocity_y`` data variables.
    """
    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    R = dataset[precip_var].values
    m, n = R.shape[1:]
    X, Y = np.meshgrid(np.arange(n), np.arange(m))

    def f(v):
        XYW = [Y + v[1], X + v[0]]
        R_w = map_coordinates(
            R[-2, :, :], XYW, mode="constant", cval=np.nan, order=0, prefilter=False
        )

        mask = np.logical_and(np.isfinite(R[-1, :, :]), np.isfinite(R_w))

        return -np.corrcoef(R[-1, :, :][mask], R_w[mask])[0, 1]

    options = {"initial_simplex": (np.array([(0, 1), (1, 0), (1, 1)]))}
    result = op.minimize(f, (1, 1), method="Nelder-Mead", options=options)

    output = np.stack([-result.x[0] * np.ones((m, n)), -result.x[1] * np.ones((m, n))])
    dataset["velocity_x"] = (["y", "x"], output[0])
    dataset["velocity_y"] = (["y", "x"], output[1])
    return dataset
