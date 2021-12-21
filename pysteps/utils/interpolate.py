# -*- coding: utf-8 -*-
"""
pysteps.utils.interpolate
=========================

Interpolation routines for pysteps.

.. autosummary::
    :toctree: ../generated/

    idwinterp2d
    rbfinterp2d
"""

import numpy as np
import xarray as xr
from pandas import MultiIndex
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from pysteps.decorators import memoize, prepare_interpolator
from pysteps.exceptions import MissingOptionalDependency

# only scipy>=1.7
try:
    from scipy.interpolate import RBFInterpolator

    RBF_IMPORTED = True
except ImportError:
    from scipy import __version__ as scipy_version

    RBF_IMPORTED = False


@prepare_interpolator()
def idwinterp2d(sparse_data, xgrid, ygrid, power=0.5, k=20, dist_offset=0.5, **kwargs):
    """
    Gridding of sparse data points with inverse distance weighting (IDW).
    Samples with missing values or coordinates are dropped.

    Parameters
    ----------
    sparse_data: xr.DataArray
        The sparse dataset with dimension "sample" and coordinates ("x", "y").
    xgrid, ygrid: array_like
        1-D arrays representing the coordinates of the 2-D target grid at which
        to interpolate data.
    power: positive float, optional
        The power parameter used to compute the distance weights as
        ``weight = distance ** (-power)``.
    k: positive int or None, optional
        The number of nearest neighbours used for each target location.
        If set to None, it interpolates using all the data points at once.
    dist_offset: float, optional
        A small, positive constant that is added to distances to avoid zero
        values. It has units of pixels.

    Other Parameters
    ----------------
    {extra_kwargs_doc}

    Returns
    -------
    output_grid: xr.DataArray
        The dataset interpolated on the target grid.
    """
    npoints = sparse_data.sizes["sample"]
    xy_coord = np.column_stack((sparse_data.x.values, sparse_data.y.values))

    # generate the target grid
    xgridv, ygridv = np.meshgrid(xgrid, ygrid)
    gridv = np.column_stack((xgridv.ravel(), ygridv.ravel()))

    if k is not None:
        k = int(np.min((k, npoints)))
        tree = _cKDTree_cached(xy_coord, hkey=kwargs.get("hkey", None))
        dist, inds = tree.query(gridv, k=k)
        if dist.ndim == 1:
            dist = dist[..., None]
            inds = inds[..., None]
    else:
        # use all points
        dist = cdist(xy_coord, gridv, "euclidean").transpose()
        inds = np.arange(npoints)[None, :] * np.ones((gridv.shape[0], npoints)).astype(
            int
        )
    inds = xr.DataArray(inds, dims=("grid", "neighbor"))

    # convert geographical distances to number of pixels
    x_res = np.abs(np.diff(xgrid[:2]))
    y_res = np.abs(np.diff(ygrid[:2]))
    dist /= np.mean([x_res, y_res])

    # compute distance-based weights
    dist += dist_offset  # avoid zero distances
    weights = 1 / np.power(dist, power)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    weights = xr.DataArray(weights, dims=("grid", "neighbor"))

    # interpolate
    output_array = (sparse_data[inds] * weights).sum("neighbor")
    output_array = output_array.astype(sparse_data.dtype)

    # assign multi-index coordinate for the grid
    grid = MultiIndex.from_product([np.array(ygrid), np.array(xgrid)], names=("y", "x"))
    output_array = output_array.assign_coords({"grid": grid})

    return output_array.unstack("grid")  # reshape as grid


@prepare_interpolator()
def rbfinterp2d(sparse_data, xgrid, ygrid, **kwargs):
    """Gridding of sparse data points with radial basis functions (RBF).

    .. _`scipy.interpolate.RBFInterpolator`:\
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html

    This method wraps the `scipy.interpolate.RBFInterpolator`_ class available
    from scipy's version 1.7 onwards.

    Parameters
    ----------
    sparse_data: xr.DataArray
        The sparse dataset with dimension "sample" and coordinates ("x", "y").
    xgrid, ygrid: array_like
        1-D arrays representing the coordinates of the 2-D target grid at which
        to interpolate data.

    Other Parameters
    ----------------
    Any of the parameters from the original `scipy.interpolate.Rbf`_ class.
    {extra_kwargs_doc}

    Returns
    -------
    output_grid: xr.DataArray
        The dataset interpolated on the target grid.
    """
    if not RBF_IMPORTED:
        raise MissingOptionalDependency(
            "Scipy's RBFInterpolate could not be imported. "
            "Check your scipy installation: "
            f"we found version {scipy_version} (should be >=1.7)"
        )

    # generate the target grid
    xgridv, ygridv = np.meshgrid(xgrid, ygrid)
    gridv = np.column_stack((xgridv.ravel(), ygridv.ravel()))

    # interpolate
    rbfi = _Rbf_cached(sparse_data, **kwargs)
    output_array = xr.DataArray(
        rbfi(gridv),
        dims=("grid", "variable"),
        coords=sparse_data.drop_vars(
            ("x", "y", "xi", "yi", "sample"), errors="ignore"
        ).coords,
        attrs=sparse_data.attrs,
    )
    output_array = output_array.astype(sparse_data.dtype)

    # assign multi-index coordinate for the grid
    grid = MultiIndex.from_product([ygrid, xgrid], names=("y", "x"))
    output_array = output_array.assign_coords({"grid": grid})

    return output_array.unstack("grid")  # reshape as grid


@memoize()
def _cKDTree_cached(*args, **kwargs):
    """Add LRU cache to cKDTree class."""
    return cKDTree(*args, **kwargs)


@memoize()
def _Rbf_cached(sparse_data, **kwargs):
    """Add LRU cache to Rbf class."""
    data_coords = np.column_stack((sparse_data.x, sparse_data.y))
    data_values = sparse_data.values
    return RBFInterpolator(data_coords, data_values, **kwargs)
