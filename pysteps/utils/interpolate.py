# -*- coding: utf-8 -*-
"""
pysteps.utils.interpolate
=========================

Interpolation routines for pysteps.

.. autosummary::
    :toctree: ../generated/

    rbfinterp2d
"""

import warnings

import numpy as np
import scipy.spatial
from scipy.interpolate import Rbf


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

    Returns
    -------
    output_array: ndarray_
        The interpolated field(s) having shape (*m*, ``ygrid.size``, ``xgrid.size``).

    """
    deprecated_args = ["rbfunction", "epsilon", "k", "nchunks"]
    deprecated_args = [arg for arg in deprecated_args if arg in list(kwargs.keys())]
    if deprecated_args:
        warnings.warn(
            "rbfinterp2d: The following keyword arguments are deprecated:\n"
            + str(deprecated_args)
            + "\nIn version 1.6, passing unsupported arguments will raise an error.",
            DeprecationWarning,
        )

    if input_array.ndim == 1:
        mode = "1-D"
    else:
        mode = "N-D"

    xgrid, ygrid = np.meshgrid(xgrid, ygrid)
    # TODO: catch np.linalg.LinAlgError
    rbfi = Rbf(*np.split(coord, coord.shape[1], 1), input_array, mode=mode)
    output_array = rbfi(xgrid, ygrid)

    return np.moveaxis(output_array, -1, 0).squeeze()


def idwinterp2d(
    coord,
    input_array,
    xgrid,
    ygrid,
    weighting_function="gaussian",
    epsilon=10,
    k=50,
    nchunks=5,
):
    """Fast 2-D grid inverse distance weighting interpolation of a sparse
    (multivariate) array.

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
    weighting_function: {"gaussian", "multiquadric", "inverse quadratic", "inverse multiquadric", "bump"}, optional
        The name of one of the available weighting functions based on a
        normalized Euclidian norm as defined in the **Notes** section below.

    epsilon: float, optional
        The shape parameter used to scale the input distance in the weighting
        function.

        A smaller value for ``epsilon`` produces a smoother interpolation. More
        details provided in the wikipedia reference page linked below.
    k: int or None, optional
        The number of nearest neighbours used for each target location.
        This can also be useful to to speed-up the interpolation.
        If set to None, it interpolates using all the data points at once.
    nchunks: int, optional
        The number of chunks in which the grid points are split to limit the
        memory usage during the interpolation.

    Returns
    -------
    output_array: ndarray_
        The interpolated field(s) having shape (*m*, ``ygrid.size``, ``xgrid.size``).

    Notes
    -----
    The coordinates are normalized before computing the Euclidean norms:

        x = (x - min(x)) / max[max(x) - min(x), max(y) - min(y)],\n
        y = (y - min(y)) / max[max(x) - min(x), max(y) - min(y)],

    where the min and max values are taken as the 2nd and 98th percentiles.

    """

    _wfunction = [
        "nearest",
        "gaussian",
        "inverse quadratic",
        "inverse multiquadric",
        "bump",
    ]

    input_array = np.copy(input_array)

    if np.any(~np.isfinite(input_array)):
        raise ValueError("input_array contains non-finite values")

    if input_array.ndim == 1:
        nvar = 1
        input_array = input_array[:, None]

    elif input_array.ndim == 2:
        nvar = input_array.shape[1]

    else:
        raise ValueError(
            "input_array must have 1 (n) or 2 dimensions (n, m), but it has %i"
            % input_array.ndim
        )

    npoints = input_array.shape[0]

    if npoints == 0:
        raise ValueError(
            "input_array (n, m) must contain at least one sample, but it has %i"
            % npoints
        )

    # only one sample, return uniform fields
    elif npoints == 1:
        output_array = np.ones((nvar, ygrid.size, xgrid.size))
        for i in range(nvar):
            output_array[i, :, :] *= input_array[:, i]
        return output_array

    coord = np.copy(coord)

    if coord.ndim != 2:
        raise ValueError(
            "coord must have 2 dimensions (n, 2), but it has %i" % coord.ndim
        )

    if npoints != coord.shape[0]:
        raise ValueError(
            "the number of samples in the input_array does not match the "
            + "number of coordinates %i!=%i" % (npoints, coord.shape[0])
        )

    # normalize coordinates
    qcoord = np.percentile(coord, [2, 98], axis=0)
    dextent = np.max(np.diff(qcoord, axis=0))
    coord = (coord - qcoord[0, :]) / dextent

    weighting_function = weighting_function.lower()
    if weighting_function not in _wfunction:
        raise ValueError(
            "Unknown weighting_function '{}'\n".format(weighting_function)
            + "The available rbfunctions are: "
            + str(_wfunction)
        ) from None

    # generate the target grid
    X, Y = np.meshgrid(xgrid, ygrid)
    grid = np.column_stack((X.ravel(), Y.ravel()))
    # normalize the grid coordinates
    grid = (grid - qcoord[0, :]) / dextent

    # k-nearest interpolation
    if k is not None and k > 0:
        k = int(np.min((k, npoints)))

        # create cKDTree object to represent source grid
        tree = scipy.spatial.cKDTree(coord)

    else:
        k = 0

    # split grid points in n chunks
    if nchunks > 1:
        subgrids = np.array_split(grid, nchunks, 0)
        subgrids = [x for x in subgrids if x.size > 0]

    else:
        subgrids = [grid]

    # loop subgrids
    i0 = 0
    output_array = np.zeros((grid.shape[0], nvar))
    for i, subgrid in enumerate(subgrids):
        idelta = subgrid.shape[0]

        if k == 0:
            # use all points
            d = scipy.spatial.distance.cdist(coord, subgrid, "euclidean").transpose()
            inds = np.arange(npoints)[None, :] * np.ones(
                (subgrid.shape[0], npoints)
            ).astype(int)

        else:
            # use k-nearest neighbours
            d, inds = tree.query(subgrid, k=k)

        if k == 1:
            # nearest neighbour
            output_array[i0 : (i0 + idelta), :] = input_array[inds, :]

        else:

            # the interpolation weights
            if weighting_function == "gaussian":
                w = np.exp(-((d * epsilon) ** 2))

            elif weighting_function == "inverse quadratic":
                w = 1.0 / (1 + (epsilon * d) ** 2)

            elif weighting_function == "inverse multiquadric":
                w = 1.0 / np.sqrt(1 + (epsilon * d) ** 2)

            elif weighting_function == "bump":
                w = np.exp(-1.0 / (1 - (epsilon * d) ** 2))
                w[d >= 1 / epsilon] = 0.0

            if not np.all(np.sum(w, axis=1)):
                w[np.sum(w, axis=1) == 0, :] = 1.0

            # interpolate
            for j in range(nvar):
                output_array[i0 : (i0 + idelta), j] = np.sum(
                    w * input_array[inds, j], axis=1
                ) / np.sum(w, axis=1)

        i0 += idelta

    # reshape to final grid size
    output_array = output_array.reshape(ygrid.size, xgrid.size, nvar)

    return np.moveaxis(output_array, -1, 0).squeeze()
