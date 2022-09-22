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
import warnings

import numpy as np
import scipy.spatial


def decluster(coord, input_array, scale, min_samples=1, verbose=False):
    """
    Decluster a set of sparse data points by aggregating, that is, taking
    the median value of all values lying within a certain distance (i.e., a
    cluster).

    Parameters
    ----------
    coord: array_like
        Array of shape (n, d) containing the coordinates of the input data into
        a space of *d* dimensions.
    input_array: array_like
        Array of shape (n) or (n, m), where *n* is the number of samples and
        *m* the number of variables.
        All values in ``input_array`` are required to have finite values.
    scale: float or array_like
        The ``scale`` parameter in the same units of ``coord``.
        It can be a scalar or an array_like of shape (d).
        Data points within the declustering ``scale`` are aggregated.
    min_samples: int, optional
        The minimum number of samples for computing the median within a given
        cluster.
    verbose: bool, optional
        Print out information.

    Returns
    -------
    out: tuple of ndarrays
        A two-element tuple (``out_coord``, ``output_array``) containing the
        declustered coordinates (l, d) and input array (l, m), where *l* is
        the new number of samples with *l* <= *n*.
    """

    coord = np.copy(coord)
    input_array = np.copy(input_array)

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
            "coord must have 2 dimensions (n, d), but it has %i" % coord.ndim
        )
    if coord.shape[0] != input_array.shape[0]:
        raise ValueError(
            "the number of samples in the input_array does not match the "
            + "number of coordinates %i!=%i" % (input_array.shape[0], coord.shape[0])
        )

    if np.isscalar(scale):
        scale = float(scale)
    else:
        scale = np.copy(scale)
        if scale.ndim != 1:
            raise ValueError(
                "scale must have 1 dimension (d), but it has %i" % scale.ndim
            )
        if scale.shape[0] != coord.shape[1]:
            raise ValueError(
                "scale must have %i elements, but it has %i"
                % (coord.shape[1], scale.shape[0])
            )
        scale = scale[None, :]

    # reduce original coordinates
    coord_ = np.floor(coord / scale)

    # keep only unique pairs of the reduced coordinates
    ucoord_ = np.unique(coord_, axis=0)

    # loop through these unique values and average data points which belong to
    # the same cluster
    dinput = np.empty(shape=(0, nvar))
    dcoord = np.empty(shape=(0, coord.shape[1]))
    for i in range(ucoord_.shape[0]):
        idx = np.all(coord_ == ucoord_[i, :], axis=1)
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

    return dcoord, dinput


def detect_outliers(input_array, thr, coord=None, k=None, verbose=False):
    """
    Detect outliers in a (multivariate and georeferenced) dataset.

    Assume a (multivariate) Gaussian distribution and detect outliers based on
    the number of standard deviations from the mean.

    If spatial information is provided through coordinates, the outlier
    detection can be localized by considering only the k-nearest neighbours
    when computing the local mean and standard deviation.

    Parameters
    ----------
    input_array: array_like
        Array of shape (n) or (n, m), where *n* is the number of samples and
        *m* the number of variables. If *m* > 1, the Mahalanobis distance
        is used.
        All values in ``input_array`` are required to have finite values.
    thr: float
        The number of standard deviations from the mean used to define an outlier.
    coord: array_like or None, optional
        Array of shape (n, d) containing the coordinates of the input data into
        a space of *d* dimensions.
        Passing ``coord`` requires that ``k`` is not None.
    k: int or None, optional
        The number of nearest neighbours used to localize the outlier
        detection. If set to None (the default), it employs all the data points (global
        detection). Setting ``k`` requires that ``coord`` is not None.
    verbose: bool, optional
        Print out information.

    Returns
    -------
    out: array_like
        A 1-D boolean array of shape (n) with True values indicating the outliers
        detected in ``input_array``.
    """

    input_array = np.copy(input_array)

    if np.any(~np.isfinite(input_array)):
        raise ValueError("input_array contains non-finite values")

    if input_array.ndim == 1:
        nsamples = input_array.size
        nvar = 1
    elif input_array.ndim == 2:
        nsamples = input_array.shape[0]
        nvar = input_array.shape[1]
    else:
        raise ValueError(
            f"input_array must have 1 (n) or 2 dimensions (n, m), "
            f"but it has {input_array.ndim}"
        )

    if nsamples < 2:
        return np.zeros(nsamples, dtype=bool)

    if coord is not None and k is not None:

        coord = np.copy(coord)
        if coord.ndim == 1:
            coord = coord[:, None]

        elif coord.ndim > 2:
            raise ValueError(
                "coord must have 2 dimensions (n, d)," f"but it has {coord.ndim}"
            )

        if coord.shape[0] != nsamples:
            raise ValueError(
                "the number of samples in input_array does not match the "
                f"number of coordinates {nsamples}!={coord.shape[0]}"
            )

        k = np.min((nsamples, k + 1))

    # global

    if k is None or coord is None:

        if nvar == 1:
            # univariate
            zdata = np.abs(input_array - np.mean(input_array)) / np.std(input_array)
            outliers = zdata > thr
        else:
            # multivariate (mahalanobis distance)
            zdata = input_array - np.mean(input_array, axis=0)
            V = np.cov(zdata.T)
            try:
                VI = np.linalg.inv(V)
                MD = np.sqrt(np.dot(np.dot(zdata, VI), zdata.T).diagonal())
            except np.linalg.LinAlgError as err:
                warnings.warn(f"{err} during outlier detection")
                MD = np.zeros(nsamples)
            outliers = MD > thr

    # local
    else:

        tree = scipy.spatial.cKDTree(coord)
        __, inds = tree.query(coord, k=k)
        outliers = np.empty(shape=0, dtype=bool)
        for i in range(inds.shape[0]):

            if nvar == 1:
                # univariate
                thisdata = input_array[i]
                neighbours = input_array[inds[i, 1:]]
                thiszdata = np.abs(thisdata - np.mean(neighbours)) / np.std(neighbours)
                outliers = np.append(outliers, thiszdata > thr)
            else:
                # multivariate (mahalanobis distance)
                thisdata = input_array[i, :]
                neighbours = input_array[inds[i, 1:], :].copy()
                thiszdata = thisdata - np.mean(neighbours, axis=0)
                neighbours = neighbours - np.mean(neighbours, axis=0)
                V = np.cov(neighbours.T)
                try:
                    VI = np.linalg.inv(V)
                    MD = np.sqrt(np.dot(np.dot(thiszdata, VI), thiszdata.T))
                except np.linalg.LinAlgError as err:
                    warnings.warn(f"{err} during outlier detection")
                    MD = 0
                outliers = np.append(outliers, MD > thr)

    if verbose:
        print(f"--- {np.sum(outliers)} outliers detected ---")

    return outliers
