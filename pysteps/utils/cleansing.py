# -*- coding: utf-8 -*-
"""
pysteps.utils.cleansing
=======================

Data cleansing routines for pysteps.

.. autosummary::
    :toctree: ../generated/

    decluster
    detect_outliers
"""

import numpy as np
import scipy.spatial


def decluster(coord, input_array, scale, min_samples=1, verbose=False):
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

    min_samples : int, optional
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


def detect_outliers(input_array, thr, coord=None, k=None, verbose=False):
    """Detect outliers in a (multivariate and georeferenced) dataset.

    Assume a (multivariate) Gaussian distribution and detect outliers based on
    the number of standard deviations from the mean.

    If spatial information is provided through coordinates, the outlier
    detection can be localized by considering only the k-nearest neighbours
    when computing the local mean and standard deviation.

    Parameters
    ----------

    input_array : array_like
        Array of shape (n) or (n, m), where n is the number of samples and m
        the number of variables. If m > 1, the Mahalanobis distance is used.
        All values in input_array are required to have finite values.

    thr : float
        The number of standard deviations from the mean that defines an outlier.

    coord : array_like, optional
        Array of shape (n, d) containing the coordinates of the input data into
        a space of d dimensions. Setting coord requires that k is not None.

    k : int or None, optional
        The number of nearest neighbours used to localize the outlier detection.
        If set to None (the default), it employs all the data points (global
        detection). Setting k requires that coord is not None.

    verbose : bool, optional
        Print out information.

    Returns
    -------

    out : array_like
        A boolean array of the same shape as input_array, with True values
        indicating the outliers detected in input_array.
    """

    input_array = np.copy(input_array)

    if np.any(~np.isfinite(input_array)):
        raise ValueError("input_array contains non-finite values")

    if input_array.ndim == 1:
        nvar = 1
    elif input_array.ndim == 2:
        nvar = input_array.shape[1]
    else:
        raise ValueError(
            "input_array must have 1 (n) or 2 dimensions (n, m), but it has %i"
            % coord.ndim
        )

    if coord is not None:

        coord = np.copy(coord)
        if coord.ndim == 1:
            coord = coord[:, None]

        elif coord.ndim > 2:
            raise ValueError(
                "coord must have 2 dimensions (n, d), but it has %i"
                % coord.ndim
            )

        if coord.shape[0] != input_array.shape[0]:
            raise ValueError(
                "the number of samples in input_array does not match the "
                + "number of coordinates %i!=%i"
                % (input_array.shape[0], coord.shape[0])
            )

        if k is None:
            raise ValueError("coord is set but k is None")

        k = np.min((coord.shape[0], k + 1))

    else:
        if k is not None:
            raise ValueError("k is set but coord=None")

    # global

    if k is None:

        if nvar == 1:

            # univariate

            zdata = (input_array - np.mean(input_array)) / np.std(input_array)
            outliers = zdata > thr

        else:

            # multivariate (mahalanobis distance)

            zdata = input_array - np.mean(input_array, axis=0)
            V = np.cov(zdata.T)
            VI = np.linalg.inv(V)
            try:
                VI = np.linalg.inv(V)
                MD = np.sqrt(np.dot(np.dot(zdata, VI), zdata.T).diagonal())
            except np.linalg.LinAlgError:
                MD = np.zeros(input_array.shape)
            outliers = MD > thr

    # local

    else:

        tree = scipy.spatial.cKDTree(coord)
        __, inds = tree.query(coord, k=k)
        outliers = np.empty(shape=0, dtype=bool)
        for i in range(inds.shape[0]):

            if nvar == 1:

                # in terms of velocity

                thisdata = input_array[i]
                neighbours = input_array[inds[i, 1:]]
                thiszdata = (thisdata - np.mean(neighbours)) / np.std(
                    neighbours
                )
                outliers = np.append(outliers, thiszdata > thr)

            else:

                # mahalanobis distance

                thisdata = input_array[i, :]
                neighbours = input_array[inds[i, 1:], :].copy()
                thiszdata = thisdata - np.mean(neighbours, axis=0)
                neighbours = neighbours - np.mean(neighbours, axis=0)
                V = np.cov(neighbours.T)
                try:
                    VI = np.linalg.inv(V)
                    MD = np.sqrt(np.dot(np.dot(thiszdata, VI), thiszdata.T))
                except np.linalg.LinAlgError:
                    MD = 0
                outliers = np.append(outliers, MD > thr)

    if verbose:
        print("--- %i outliers detected ---" % np.sum(outliers))

    return outliers
