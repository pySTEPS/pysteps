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
from pysteps.exceptions import MissingOptionalDependency

try:
    from sklearn import decomposition

    SKLEARN_IMPORTED = True
except ImportError:
    SKLEARN_IMPORTED = False


def pca_transform(
    forecast_ens: np.ndarray,
    mask: np.ndarray | None = None,
    pca_params: dict | None = None,
    get_params: bool = False,
    **kwargs: dict,
):
    """
    Transform ensemble forecasts from physical space into principal component (PC) space.

    Parameters
    ----------
    forecast_ens : np.ndarray
        Array of shape (n_ens, n_features) containing the ensemble forecasts
        in physical space.
    mask : np.ndarray, optional
        Mask to transform only grid points at which at least 10 ensemble
        members have forecast precipitation, to fulfill the Lien criterion
        (Lien et al., 2013) mentioned in Nerini et al., 2019.
        The default is None.
    pca_params : dict, optional
        Preconstructed Principal Component Analysis (PCA) object. If given,
        this is used instead of fitting a new PCA. The default is None.
    get_params : bool, optional
        If True, return the PCA parameters in addition to the transformed data.
        The default is False.
    n_components : int
        Number of principal components to retain.
    svd_solver : {'auto', 'full', 'covariance_eigh', 'arpack', 'randomized'}
        Solver to use for the singular value decomposition. For details, see
        the documentation of ``sklearn.decomposition.PCA``.

    Returns
    -------
    forecast_ens_pc : np.ndarray
        Array of shape (n_components, n_ens) containing the ensemble forecasts
        transformed into PC space. If no mask is given, the full dataset is
        transformed; otherwise only the mask-filtered values are transformed.
    pca_params : dict, optional
        Dictionary containing the PCA parameters, returned if
        ``get_params=True``. The dictionary has the following keys:

        principal_components : np.ndarray
            Array of shape (n_components, n_features) containing the
            principal component vectors in feature space.
        mean : np.ndarray
            Array of shape (n_features,) containing the per-feature
            empirical mean estimated from the input data.
        explained_variance : np.ndarray
            Array of shape (n_features,) containing the per-feature
            explained variance ratio.
    """

    # Test import of sklean
    if not SKLEARN_IMPORTED:
        raise MissingOptionalDependency(
            "scikit-learn package is required for principal component analysis "
            "but it is not installed"
        )

    # Input data have to be two-dimensional
    if forecast_ens.ndim != 2:
        raise ValueError("Input array should be two-dimensional!")

    if pca_params is None:
        # Check whether n_components and svd_solver are given as keyword arguments
        n_components = kwargs.get("n_components", forecast_ens.shape[0])
        svd_solver = kwargs.get("svd_solver", "full")

        # Initialize PCA and fit it to the input data
        pca = decomposition.PCA(n_components=n_components, svd_solver=svd_solver)
        pca.fit(forecast_ens)

        # Create output dictionary and save principal components and mean
        pca_params = {}
        pca_params["principal_components"] = pca.components_
        pca_params["mean"] = pca.mean_
        pca_params["explained_variance"] = pca.explained_variance_ratio_

    else:
        # If output dict is given, check whether principal components and mean are included
        if not "principal_components" in pca_params.keys():
            raise KeyError("Output is not None but has no key 'principal_components'!")
        if not "mean" in pca_params.keys():
            raise KeyError("Output is not None but has no key 'mean'!")

        # Check whether PC and mean have the correct shape
        if forecast_ens.shape[1] != len(pca_params["mean"]):
            raise ValueError("pca mean has not the same length as the input array!")
        if forecast_ens.shape[1] != pca_params["principal_components"].shape[1]:
            raise ValueError(
                "principal components have not the same length as the input array"
            )

    # If no mask is given, transform the full input data into PC space.
    if mask is None:
        forecast_ens_pc = np.dot(
            (forecast_ens - pca_params["mean"]), pca_params["principal_components"].T
        )
    else:
        forecast_ens_pc = np.dot(
            (forecast_ens[:, mask] - pca_params["mean"][mask]),
            pca_params["principal_components"][:, mask].T,
        )

    if get_params:
        return forecast_ens_pc, pca_params
    else:
        return forecast_ens_pc


def pca_backtransform(forecast_ens_pc: np.ndarray, pca_params: dict):
    """
    Reconstruct ensemble forecasts from principal component (PC) space back into physical space.

    Parameters
    ----------
    forecast_ens_pc : np.ndarray
        Array of shape (n_components, n_ens) containing the ensemble forecasts
        represented in PC space.
    pca_params : dict
        Parameters of the PCA transformation. The dictionary contains the following keys:

        principal_components : np.ndarray
            Array of shape (n_components, n_features) containing the principal
            component vectors in feature space.
        mean : np.ndarray
            Array of shape (n_features,) containing the per-feature empirical mean
            estimated from the training data.

    Returns
    -------
    forecast_ens : np.ndarray
        Array of shape (n_ens, n_features) containing the ensemble forecasts
        reconstructed in physical space.
    """

    # If output dict is given, check whether principal components and mean are included
    if not "principal_components" in pca_params.keys():
        raise KeyError("Output is not None but has no key 'principal_components'!")
    if not "mean" in pca_params.keys():
        raise KeyError("Output is not None but has no key 'mean'!")

    # Check whether PC and forecast_ens_pc have the correct shape
    if forecast_ens_pc.shape[1] != pca_params["principal_components"].shape[0]:
        raise ValueError("pca mean has not the same length as the input array!")

    # Transform forecast_ens_pc back into physical space.
    forecast_ens = (
        np.dot(forecast_ens_pc, pca_params["principal_components"]) + pca_params["mean"]
    )

    return forecast_ens
