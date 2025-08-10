# -*- coding: utf-8 -*-
"""
pysteps.utils.pca

Principal component analysis for pysteps.

.. autosummary::
    :toctree: ../generated/

    pca_transform
    pca_backtransform
"""

import numpy as np

from sklearn import decomposition


def pca_transform(
    X: np.ndarray,
    mask: np.ndarray | None = None,
    pca_params: dict | None = None,
    get_params: bool = False,
    **kwargs: dict,
):
    """
    Transform a two-dimensional array into PC space.

    Parameters
    ----------
    X: np.ndarray
        Two-dimensional array of shape (n_ens, n_gridpoints) containing the
        precipitation ensemble forecast that should be decomposed in PC space.

    Other Parameters
    ----------------
    mask: np.ndarray
        Optional mask to transform only grid points at which at least 10 ensemble
        members have forecast precipitaton to fulfill the Lien criterion (Lien et al.,
        2013) that is mentioned in Nerini et al., 2019.
    pca_params: dict
        Optional output dictionary containing the preconstructed Principal Component
        Analysis, since this construction is performed on the full precipitation
        forecast dataset.
    get_params: bool
        Optional flag whether pca_params should output or not.
    n_components: int
        Number of principal components
    svd_solver: {'auto', 'full', 'covariance_eigh', 'arpack', 'randomized'}
        Solver for the singular vector decomposition. For a detailed description see
        the documentation of sklearn.decomposition.PCA

    Returns
    -------
    X_pc: np.ndarray
        Two-dimensional array of shape (n_components, n_ens) containing the input data
        transformed into PC space. If not a mask is given as input, the full dataset is
        transformed, otherwise only the mask-filtered values are transformed.
    pca_params: dict (optional)
        If the respective flag (get_params) is set to True, a dictionary is returned
        containing:
        -principal_components: np.ndarray
            Two-dimensional array of shape (n_components, n_features) containing the
            principal components in feature space.
        -mean: np.ndarray
            One-dimensional array of shape (n_features) containing the per-feature
            empirical mean estimated from the input data.
    """

    # Input data have to be two-dimensional
    if X.ndim != 2:
        raise ValueError("Input array should be two-dimensional!")

    if pca_params is None:

        # Check whether n_components and svd_solver are given as keyword arguments
        n_components = kwargs.get("n_components", X.shape[0])
        svd_solver = kwargs.get("svd_solver", "full")

        # Initialize PCA and fit it to the input data
        pca = decomposition.PCA(n_components=n_components, svd_solver=svd_solver)
        pca.fit(X)

        # Create output dictionary and save principal components and mean
        pca_params = {}
        pca_params["principal_components"] = pca.components_
        pca_params["mean"] = pca.mean_

    else:

        # If output dict is given, check whether principal components and mean are included
        if not "principal_components" in pca_params.keys():
            raise KeyError("Output is not None but has no key 'principal_components'!")
        if not "mean" in pca_params.keys():
            raise KeyError("Output is not None but has no key 'mean'!")

        # Check whether PC and mean have the correct shape
        if X.shape[1] != len(pca_params["mean"]):
            raise ValueError("pca mean has not the same length as the input array!")
        if X.shape[1] != pca_params["principal_components"].shape[1]:
            raise ValueError(
                "principal components have not the same length as the input array"
            )

    # If no mask is given, transform the full input data into PC space.
    if mask is None:
        X_pc = np.dot((X - pca_params["mean"]), pca_params["principal_components"].T)
    else:
        X_pc = np.dot(
            (X[:, mask] - pca_params["mean"][mask]),
            pca_params["principal_components"][:, mask].T,
        )

    if get_params == True:
        return X_pc, pca_params
    else:
        return X_pc


def pca_backtransform(X_pc: np.ndarray, pca_params: dict):
    """
    Transform a given PC transformation back into physical space.

    Parameters
    ----------
    X_pc: np.ndarray
        Two-dimensional array of shape (n_components, n_ens) containing the full input
        data transformed into PC space.
    pca_params: dict
        If the respective flag (get_params) is set to True, a dictionary is returned
        containing:
        -principal_components: np.ndarray
            Two-dimensional array of shape (n_components, n_features) containing the
            principal components in feature space.
        -mean: np.ndarray
            One-dimensional array of shape (n_features) containing the per-feature
            empirical mean estimated from the input data.

    Returns
    -------
    X: np.ndarray
        Two-dimensional of shape (n_ens, n_gridpoints) containing the backtransformed
        precipitation forecast.
    """

    # If output dict is given, check whether principal components and mean are included
    if not "principal_components" in pca_params.keys():
        raise KeyError("Output is not None but has no key 'principal_components'!")
    if not "mean" in pca_params.keys():
        raise KeyError("Output is not None but has no key 'mean'!")

    # Check whether PC and X_pc have the correct shape
    if X_pc.shape[0] != pca_params["principal_components"].shape[0]:
        raise ValueError("pca mean has not the same length as the input array!")

    # Transform X_pc back into physical space.
    X = np.dot(X_pc.T, pca_params["principal_components"]) + pca_params["mean"]

    return X
