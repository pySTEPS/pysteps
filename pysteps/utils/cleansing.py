# -*- coding: utf-8 -*-
"""
pysteps.utils.cleansing
=======================

Data cleansing routines for pysteps.

.. autosummary::
    :toctree: ../generated/

    decluster
    remove_outliers
"""
import warnings

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


def decluster(coord, input_array, scale, min_samples=1, verbose=False):
    """Decluster a set of sparse data points by aggregating, that is, taking
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


def _compute_standard_score(samples, neighbours=None):
    """
    Compute standard score in one or more dimensionsby using the
    Mahalanobis distance to generalize to multi-dimensions.
    """
    if neighbours is None:
        neighbours = samples

    neighbours_mean = neighbours.mean("sample")
    samples = samples - neighbours_mean
    neighbours = neighbours - neighbours_mean
    cov_matrix = np.cov(neighbours.transpose("variable", ...))
    cov_matrix = np.atleast_2d(cov_matrix)
    try:
        # Mahalanobis distance
        cov_matrix_inv = np.linalg.inv(cov_matrix)
        maha_dist = np.dot(
            np.dot(samples.transpose(..., "variable"), cov_matrix_inv),
            samples.transpose("variable", ...)
        ).diagonal()
        maha_dist = np.sqrt(maha_dist)

    except np.linalg.LinAlgError as err:
        warnings.warn(f"{err} during outlier detection")
        maha_dist = np.zeros(samples.sizes["sample"])

    return maha_dist


def remove_outliers(sparse_data, thr, k=None, verbose=False):
    """Detect outliers in a (multivariate and georeferenced) dataset.

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
    sparse_data = sparse_data.copy()

    if np.any(~np.isfinite(sparse_data)):
        raise ValueError("sparse_data contains non-finite values")

    nsample = sparse_data.sizes["sample"]

    if nsample < 2:
        zvalues = xr.zeros_like(sparse_data)

    else:
        # global
        if k is None:
            zvalues = _compute_standard_score(sparse_data)

        # local neighborhood
        else:
            k = np.min((nsample, k + 1))
            coords = np.column_stack((sparse_data.x, sparse_data.y))
            tree = cKDTree(coords)
            __, inds = tree.query(coords, k=k)
            zvalues = np.zeros(shape=nsample)
            for i in range(inds.shape[0]):
                this_sample = sparse_data.isel(sample=[i])
                neighbours = sparse_data.isel(sample=inds[i, 1:])
                zvalues[i] = _compute_standard_score(this_sample, neighbours)

    outliers = zvalues >= thr
    if verbose:
        print(f"... removed {outliers.sum()} outliers")

    return sparse_data.isel(sample=~outliers)
