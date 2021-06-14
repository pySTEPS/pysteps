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
import scipy.spatial
from scipy.interpolate import Rbf

from pysteps.decorators import preamble_interpolation


@preamble_interpolation()
def idwinterp2d(coord, input_array, xgrid, ygrid, power=1, k=20, nchunks=5, **kwargs):
    """Inverse distance weighting interpolation of a sparse (multivariate) array.

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    Parameters
    ----------
    coord: array_like
        Array of shape (n, 2) containing the coordinates of the data points
        into a 2-dimensional space.
    input_array: array_like
        Array of shape (n) or (n, m) containing the values of the data points,
        where *n* is the number of data points and *m* the number of co-located
        variables. All values in ``input_array`` are required to have finite values.
    xgrid, ygrid: array_like
        1D arrays representing the coordinates of the 2-D output grid.
    power: positive float, optional
        The power parameter used to comptute the distance weights as w = d^(-power).
    k: int or None, optional
        The number of nearest neighbours used for each target location.
        This can also be useful to to speed-up the interpolation.
        If set to None, it interpolates using all the data points at once.

    Keyword Arguments
    -----------------
    {extra_kwargs_doc}

    Returns
    -------
    output_array: ndarray_
        The interpolated field(s) having shape (*m*, ``ygrid.size``, ``xgrid.size``).
    """
    if input_array.ndim == 1:
        nvar = 1
        input_array = input_array[:, None]

    elif input_array.ndim == 2:
        nvar = input_array.shape[1]

    npoints = input_array.shape[0]

    # generate the target grid
    xgridv, ygridv = np.meshgrid(xgrid, ygrid)
    gridv = np.column_stack((xgridv.ravel(), ygridv.ravel()))

    # k-nearest interpolation
    if k is not None and k > 0:
        k = int(np.min((k, npoints)))

        # create cKDTree object to represent source grid
        tree = scipy.spatial.cKDTree(coord)
    else:
        k = 0

    if k == 0:
        # use all points
        dist = scipy.spatial.distance.cdist(coord, gridv, "euclidean").transpose()
        inds = np.arange(npoints)[None, :] * np.ones((gridv.shape[0], npoints)).astype(
            int
        )
    else:
        # use k-nearest neighbours
        dist, inds = tree.query(gridv, k=k)

    if k == 1:
        # nearest neighbour
        output_array = input_array[inds, :]
    else:
        # compute distance-based weights
        weights = 1 / np.power(dist + 1e-6, power)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        # interpolate
        output_array = np.sum(
            input_array[inds, :] * weights[..., None],
            axis=1,
        )

    # reshape to final grid size
    output_array = output_array.reshape(ygrid.size, xgrid.size, nvar)

    return np.moveaxis(output_array, -1, 0).squeeze()


@preamble_interpolation(nchunks=0)
def rbfinterp2d(coord, input_array, xgrid, ygrid, **kwargs):
    """Radial basis function interpolation of a sparse (multivariate) array.

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
    .. _`scipy.interpolate.Rbf`:\
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html

    This method wraps the `scipy.interpolate.Rbf`_ class.

    Parameters
    ----------
    coord: array_like
        Array of shape (n, 2) containing the coordinates of the data points
        into a 2-dimensional space.
    input_array: array_like
        Array of shape (n) or (n, m) containing the values of the data points,
        where *n* is the number of data points and *m* the number of co-located
        variables. All values in ``input_array`` are required to have finite values.
    xgrid, ygrid: array_like
        1D arrays representing the coordinates of the 2-D output grid.

    Keyword Arguments
    -----------------
    Any of the parameters from the original `scipy.interpolate.Rbf`_ class.
    {extra_kwargs_doc}

    Returns
    -------
    output_array: ndarray_
        The interpolated field(s) having shape (*m*, ``ygrid.size``, ``xgrid.size``).
    """
    deprecated_args = ["rbfunction", "k"]
    deprecated_args = [arg for arg in deprecated_args if arg in list(kwargs.keys())]
    if deprecated_args:
        warnings.warn(
            "rbfinterp2d: The following keyword arguments are deprecated:\n"
            + str(deprecated_args),
            DeprecationWarning,
        )

    if input_array.ndim == 1:
        kwargs["mode"] = "1-D"
    else:
        kwargs["mode"] = "N-D"

    xgridv, ygridv = np.meshgrid(xgrid, ygrid)
    rbfi = Rbf(*np.split(coord, coord.shape[1], 1), input_array, **kwargs)
    output_array = rbfi(xgridv, ygridv)

    return np.moveaxis(output_array, -1, 0).squeeze()
