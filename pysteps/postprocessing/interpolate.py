# -*- coding: utf-8 -*-
"""
pysteps.postprocessing.interpolate
==================================

Interpolation routines for pysteps.

.. autosummary::
    :toctree: ../generated/

    decluster_sparse_data
    rbfinterp2d

"""

import numpy as np
import scipy.spatial

def decluster_sparse_data(coord, input_array, scale, min_samples, verbose=False):
    """Decluster a set of sparse data points by aggregating (i.e., taking the
    median value) all points within a certain distance (i.e., a cluster).

    Parameters
    ----------

    coord : array_like
        Array of shape (n, 2) containing the coordinates of the input data into
        a 2-dimensional space.

    input_array : array_like
        Array of shape (n) or (n, m), where n is the number of samples and m
        the number of variables.
        All values in input_array are required to have finite values.

    scale : float or array_like
        The scale parameter in the same units of coord. Data points within this
        declustering scale are averaged together.

    min_samples : int
        The minimum number of samples for computing the median within a given
        cluster.

    verbose : bool, optional
        Print out information.

    Returns
    -------

    out : tuple of ndarrays
        A two-element tuple (dinput, dcoord) containing the declustered input_array
        (d, m) and coordinates (d, 2), where d is the new number of samples
        (d < n).

    """

    coord = np.copy(coord)
    input_array = np.copy(input_array)
    scale = np.float(scale)

    # check inputs
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

    if coord.ndim != 2:
        raise ValueError(
            "coord must have 2 dimensions (n, 2), but it has %i" % coord.ndim
        )

    if coord.shape[0] != input_array.shape[0]:
        raise ValueError(
            "the number of samples in the input_array does not match the "
            + "number of coordinates %i!=%i"
            % (input_array.shape[0], coord.shape[0])
        )

    # reduce original coordinates
    coord_ = np.floor(coord / scale)

    # keep only unique pairs of the reduced coordinates
    coordb_ = np.ascontiguousarray(coord_).view(
        np.dtype((np.void, coord_.dtype.itemsize * coord_.shape[1]))
    )
    __, idx = np.unique(coordb_, return_index=True)
    ucoord_ = coord_[idx]

    # loop through these unique values and average vectors which belong to
    # the same declustering grid cell
    dinput = np.empty(shape=(0, nvar))
    dcoord = np.empty(shape=(0, 2))
    for i in range(ucoord_.shape[0]):
        idx = np.logical_and(
            coord_[:, 0] == ucoord_[i, 0], coord_[:, 1] == ucoord_[i, 1]
        )
        npoints = np.sum(idx)
        if npoints >= min_samples:
            dinput = np.append(
                dinput, np.median(input_array[idx, :], axis=0)[None, :], axis=0
            )
            dcoord = np.append(
                dcoord, np.median(coord[idx, :], axis=0)[None, :], axis=0
            )

    if verbose:
        print("--- %i samples left after declustering ---" % dinput.shape[0])

    return dcoord.squeeze(), dinput

def rbfinterp2d(
    coord,
    input_array,
    xgrid,
    ygrid,
    rbfunction="gaussian",
    epsilon=1,
    k=50,
    nchunks=5,
):
    """Fast kernel interpolation of a (multivariate) array over a 2D grid using
    a radial basis function.

    Parameters
    ----------

    coord : array_like
        Array of shape (n, 2) containing the coordinates of the data points into
        a 2-dimensional space.

    input_array : array_like
        Array of shape (n) or (n, m), where n is the number of data points and
        m the number of co-located variables.
        All values in input_array are required to have finite values.

    xgrid, ygrid : array_like
        1D arrays representing the coordinates of the target grid.

    rbfunction : {"gaussian", "multiquadric", "inverse quadratic", "inverse
        multiquadric", "bump"}, optional
        The name of one of the available radial basis function based on the Euclidian
        norm. See also the Notes section below.

    epsilon : float, optional
        The shape parameter > 0 used to scale the input to the radial kernel.

    k : int or None, optional
        The number of nearest neighbours used to speed-up the interpolation.
        If set to None, it interpolates based on all the data points.

    nchunks : int, optional
        The number of chunks in which the grid points are split to limit the
        memory usage during the interpolation.

    Returns
    -------

    output_array : array_like
        The interpolated field(s) having shape (m, ygrid.size, xgrid.size).

    Notes
    -----

    The input coordinates are normalized before computing the euclidean norms:

        x = (x - median(x)) / MAD / 1.4826

    where MAD = median(|x - median(x)|).

    The definitions of the radial basis functions are taken from the following
    wikipedia page: https://en.wikipedia.org/wiki/Radial_basis_function
    """

    _rbfunctions = [
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
    mcoord = np.median(coord, axis=0)
    madcoord = 1.4826 * np.median(np.abs(coord - mcoord), axis=0)
    coord = (coord - mcoord) / madcoord

    rbfunction = rbfunction.lower()
    if rbfunction not in _rbfunctions:
        raise ValueError(
            "Unknown rbfunction '{}'\n".format(rbfunction)
            + "The available rbfunctions are: "
            + str(_rbfunctions)
        ) from None

    # generate the target grid
    X, Y = np.meshgrid(xgrid, ygrid)
    grid = np.column_stack((X.ravel(), Y.ravel()))
    grid = (grid - mcoord) / madcoord

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
            d = scipy.spatial.distance.cdist(
                coord, subgrid, "euclidean"
            ).transpose()
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
            if rbfunction == "gaussian":
                w = np.exp(-(d * epsilon) ** 2)

            elif rbfunction == "inverse quadratic":
                w = 1.0 / (1 + (epsilon * d) ** 2)

            elif rbfunction == "inverse multiquadric":
                w = 1.0 / np.sqrt(1 + (epsilon * d) ** 2)

            elif rbfunction == "bump":
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