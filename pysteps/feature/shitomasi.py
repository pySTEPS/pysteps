"""
pysteps.feature.shitomasi
=========================

Shi-Tomasi features detection method to detect corners in an image.

.. autosummary::
    :toctree: ../generated/

    detection
"""

import numpy as np
from numpy.ma.core import MaskedArray

from pysteps.exceptions import MissingOptionalDependency

try:
    import cv2

    CV2_IMPORTED = True
except ImportError:
    CV2_IMPORTED = False


def detection(
    input_image,
    max_corners=1000,
    max_num_features=None,
    quality_level=0.01,
    min_distance=10,
    block_size=5,
    buffer_mask=5,
    use_harris=False,
    k=0.04,
    verbose=False,
    **kwargs,
):
    """
    .. _`Shi-Tomasi`:\
        https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541

    Interface to the OpenCV `Shi-Tomasi`_ features detection method to detect
    corners in an image.

    Corners are used for local tracking methods.

    .. _MaskedArray:\
        https://docs.scipy.org/doc/numpy/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    .. _`Harris detector`:\
        https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345

    .. _cornerMinEigenVal:\
        https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#ga3dbce297c1feb859ee36707e1003e0a8

    Parameters
    ----------
    input_image: ndarray_ or MaskedArray_
        Array of shape (m, n) containing the input image.

        In case of ndarray_, invalid values (Nans or infs) are masked,
        otherwise the mask of the MaskedArray_ is used. Such mask defines a
        region where features are not detected.

        The fill value for the masked pixels is taken as the minimum of all
        valid pixels.
    max_corners: int, optional
        The ``maxCorners`` parameter in the `Shi-Tomasi`_ corner detection
        method.
        It represents the maximum number of points to be tracked (corners).
        If set to zero, all detected corners are used.
    max_num_features: int, optional
        If specified, this argument is substituted for max_corners. Set to None
        for no restriction. Added for compatibility with the feature detector
        interface.
    quality_level: float, optional
        The ``qualityLevel`` parameter in the `Shi-Tomasi`_ corner detection
        method.
        It represents the minimal accepted quality for the image corners.
    min_distance: int, optional
        The ``minDistance`` parameter in the `Shi-Tomasi`_ corner detection
        method.
        It represents minimum possible Euclidean distance in pixels between
        corners.
    block_size: int, optional
        The ``blockSize`` parameter in the `Shi-Tomasi`_ corner detection
        method.
        It represents the window size in pixels used for computing a derivative
        covariation matrix over each pixel neighbourhood.
    use_harris: bool, optional
        Whether to use a `Harris detector`_  or cornerMinEigenVal_.
    k: float, optional
        Free parameter of the Harris detector.
    buffer_mask: int, optional
        A mask buffer width in pixels. This extends the input mask (if any)
        to limit edge effects.
    verbose: bool, optional
        Print the number of features detected.

    Returns
    -------
    points: ndarray_
        Array of shape (p, 2) indicating the pixel coordinates of *p* detected
        corners.

    References
    ----------
    Jianbo Shi and Carlo Tomasi. Good features to track. In Computer Vision and
    Pattern Recognition, 1994. Proceedings CVPR'94., 1994 IEEE Computer Society
    Conference on, pages 593–600. IEEE, 1994.
    """
    if not CV2_IMPORTED:
        raise MissingOptionalDependency(
            "opencv package is required for the goodFeaturesToTrack() "
            "routine but it is not installed"
        )

    input_image = input_image.copy()

    if input_image.ndim != 2:
        raise ValueError("input_image must be a two-dimensional array")

    # Check if a MaskedArray is used. If not, mask the ndarray
    if not isinstance(input_image, MaskedArray):
        input_image = np.ma.masked_invalid(input_image)

    np.ma.set_fill_value(input_image, input_image.min())

    # buffer the quality mask to ensure that no vectors are computed nearby
    # the edges of the radar mask
    mask = np.ma.getmaskarray(input_image).astype("uint8")
    if buffer_mask > 0:
        mask = cv2.dilate(
            mask, np.ones((int(buffer_mask), int(buffer_mask)), np.uint8), 1
        )
        input_image[mask] = np.ma.masked

    # scale image between 0 and 255
    im_min = input_image.min()
    im_max = input_image.max()
    if im_max - im_min > 1e-8:
        input_image = (input_image.filled() - im_min) / (im_max - im_min) * 255
    else:
        input_image = input_image.filled() - im_min

    # convert to 8-bit
    input_image = np.ndarray.astype(input_image, "uint8")
    mask = (-1 * mask + 1).astype("uint8")

    params = dict(
        maxCorners=max_num_features if max_num_features is not None else max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=use_harris,
        k=k,
    )
    points = cv2.goodFeaturesToTrack(input_image, mask=mask, **params)
    if points is None:
        points = np.empty(shape=(0, 2))
    else:
        points = points[:, 0, :]

    if verbose:
        print(f"--- {points.shape[0]} good features to track detected ---")

    return points
