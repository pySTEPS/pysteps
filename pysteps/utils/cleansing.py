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
from pandas import MultiIndex
from scipy.spatial import cKDTree


def decluster(sparse_data, scale, verbose=False):
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
    scale: float
        The ``scale`` parameter in the same units of ``coord``.
        Data points within the declustering ``scale`` are aggregated.
    verbose: bool, optional
        Print out information.

    Returns
    -------
    out: tuple of ndarrays
        A two-element tuple (``out_coord``, ``output_array``) containing the
        declustered coordinates (l, d) and input array (l, m), where *l* is
        the new number of samples with *l* <= *n*.
    """
    sparse_data = sparse_data.copy()
    if scale is None:
        return sparse_data

    # this is a bit of a hack, necessary to use groupby on a arbitrary set of
    # multi-index coordinates
    x = sparse_data.x.values
    y = sparse_data.y.values
    reduced_coords = MultiIndex.from_arrays(
        (x // scale, y // scale), names=("xr", "yr")
    )
    sparse_data = sparse_data.assign_coords({"sample": reduced_coords})
    ds = sparse_data.to_dataset(name="name")
    ds = ds.reset_coords(("x", "y", "xi", "yi"))
    ds = ds.groupby("sample").median()
    ds = ds.drop_vars("sample")
    ds = ds.set_coords(("x", "y", "xi", "yi"))
    cluster_data = ds["name"]
    cluster_data.name = sparse_data.name

    # after clustering, reassign original dtype to coordinates
    for coord in cluster_data.coords:
        cluster_data[coord] = cluster_data[coord].astype(sparse_data[coord].dtype)

    if verbose:
        print(f"... {cluster_data.sizes['sample']} samples left after declustering")

    return cluster_data


def _compute_standard_score(samples, neighbours=None):
    """
    Compute standard score in one or more dimensions by using the
    Mahalanobis distance.
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
            samples.transpose("variable", ...),
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
    if np.any(~np.isfinite(sparse_data)):
        raise ValueError("sparse_data contains non-finite values")

    sparse_data = sparse_data.copy()
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
