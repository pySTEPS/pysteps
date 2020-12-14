# -*- coding: utf-8 -*-
"""
pysteps.feature.blob
====================

Blob detection methods.

.. autosummary::
    :toctree: ../generated/

    detection
"""

import numpy as np

from pysteps.exceptions import MissingOptionalDependency

try:
    from skimage import feature

    SKIMAGE_IMPORTED = True
except ImportError:
    SKIMAGE_IMPORTED = False


def detection(
    input_image,
    method="log",
    threshold=0.5,
    min_sigma=3,
    max_sigma=20,
    overlap=0.5,
    return_sigmas=False,
    **kwargs,
):
    """
    .. _`feature.blob_*`:\
    https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html

    Interface to the `feature.blob_*`_ methods implemented in scikit-image. A
    blob is defined as a scale-space maximum of a Gaussian-filtered image.

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    Parameters
    ----------
    input_image: array_like
        Array of shape (m, n) containing the input image. Nan values are ignored.
    method: {'log', 'dog', 'doh'}, optional
        The method to use: 'log' = Laplacian of Gaussian, 'dog' = Difference of
        Gaussian, 'doh' = Determinant of Hessian.
    threshold: float, optional
        Detection threshold.
    min_sigma: float, optional
        The minimum standard deviation for the Gaussian kernel.
    max_sigma: float, optional
        The maximum standard deviation for the Gaussian kernel.
    overlap: float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a fraction
        greater than the value for overlap, the smaller blob is eliminated.
    return_sigmas: bool, optional
        If True, return the standard deviations of the Gaussian kernels
        corresponding to the detected blobs.

    Returns
    -------
    points: ndarray_
        Array of shape (p, 2) or (p, 3) indicating the pixel coordinates of *p*
        detected blobs. If return_sigmas is True, the third column contains
        the standard deviations of the Gaussian kernels corresponding to the
        blobs.
    """
    if method not in ["log", "dog", "doh"]:
        raise ValueError("unknown method %s, must be 'log', 'dog' or 'doh'" % method)

    if not SKIMAGE_IMPORTED:
        raise MissingOptionalDependency(
            "skimage is required for the blob_detection routine but it is not installed"
        )

    if method == "log":
        detector = feature.blob_log
    elif method == "dog":
        detector = feature.blob_dog
    else:
        detector = feature.blob_doh

    blobs = detector(
        input_image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold,
        overlap=overlap,
        **kwargs,
    )

    if not return_sigmas:
        return np.column_stack([blobs[:, 1], blobs[:, 0]])
    else:
        return np.column_stack([blobs[:, 1], blobs[:, 0], blobs[:, 2]])
