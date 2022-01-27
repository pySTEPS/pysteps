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

import warnings

import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


from pysteps.decorators import memoize, prepare_interpolator


@prepare_interpolator()
def idwinterp2d(
    xy_coord, values, xgrid, ygrid, power=0.5, k=20, dist_offset=0.5, **kwargs
):
    """
    Inverse distance weighting interpolation of a sparse (multivariate) array.

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    Parameters
    ----------
    xy_coord: ndarray_
        Array of shape (n, 2) containing the coordinates of the data points in
        a 2-dimensional space.
    values: ndarray_
        Array of shape (n) or (n, m) containing the values of the data points,
        where *n* is the number of data points and *m* the number of co-located
        variables. All elements in ``values`` are required to be finite.
    xgrid, ygrid: ndarray_
        1-D arrays representing the coordinates of the 2-D output grid.
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
    output_array: ndarray_
        The interpolated field(s) having shape (``ygrid.size``, ``xgrid.size``)
        or (*m*, ``ygrid.size``, ``xgrid.size``).
    """
    if values.ndim == 1:
        nvar = 1
        values = values[:, None]

    elif values.ndim == 2:
        nvar = values.shape[1]

    npoints = values.shape[0]

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

    # convert geographical distances to number of pixels
    x_res = np.gradient(xgrid)
    y_res = np.gradient(ygrid)
    mean_res = np.mean(np.abs([x_res.mean(), y_res.mean()]))
    dist /= mean_res

    # compute distance-based weights
    dist += dist_offset  # avoid zero distances
    weights = 1 / np.power(dist, power)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    # interpolate
    output_array = np.sum(
        values[inds, :] * weights[..., None],
        axis=1,
    )

    # reshape to final grid size
    output_array = output_array.reshape(ygrid.size, xgrid.size, nvar)

    return np.moveaxis(output_array, -1, 0).squeeze()


@prepare_interpolator()
def rbfinterp2d(xy_coord, values, xgrid, ygrid, **kwargs):
    """
    Radial basis function interpolation of a sparse (multivariate) array.

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
    .. _`scipy.interpolate.Rbf`:\
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html

    This method wraps the `scipy.interpolate.Rbf`_ class.

    Parameters
    ----------
    xy_coord: ndarray_
        Array of shape (n, 2) containing the coordinates of the data points in
        a 2-dimensional space.
    values: ndarray_
        Array of shape (n) or (n, m) containing the values of the data points,
        where *n* is the number of data points and *m* the number of co-located
        variables. All values in ``values`` are required to be finite.
    xgrid, ygrid: ndarray_
        1-D arrays representing the coordinates of the 2-D output grid.

    Other Parameters
    ----------------
    Any of the parameters from the original `scipy.interpolate.Rbf`_ class.
    {extra_kwargs_doc}

    Returns
    -------
    output_array: ndarray_
        The interpolated field(s) having shape (``ygrid.size``, ``xgrid.size``)
        or (*m*, ``ygrid.size``, ``xgrid.size``).
    """
    deprecated_args = ["rbfunction", "k"]
    deprecated_args = [arg for arg in deprecated_args if arg in list(kwargs.keys())]
    if deprecated_args:
        warnings.warn(
            "rbfinterp2d: The following keyword arguments are deprecated:\n"
            + str(deprecated_args),
            DeprecationWarning,
        )

    if values.ndim == 1:
        kwargs["mode"] = "1-D"
    else:
        kwargs["mode"] = "N-D"

    xgridv, ygridv = np.meshgrid(xgrid, ygrid)
    rbfi = _Rbf_cached(*np.split(xy_coord, xy_coord.shape[1], 1), values, **kwargs)
    output_array = rbfi(xgridv, ygridv)

    return np.moveaxis(output_array, -1, 0).squeeze()


@memoize()
def _cKDTree_cached(*args, **kwargs):
    """Add LRU cache to cKDTree class."""
    return cKDTree(*args)


@memoize()
def _Rbf_cached(*args, **kwargs):
    """Add LRU cache to Rbf class."""
    return Rbf(*args, **kwargs)
