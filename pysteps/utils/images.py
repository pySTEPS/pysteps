# -*- coding: utf-8 -*-
"""
pysteps.utils.images
====================

Image processing routines for pysteps.

.. _`Shi-Tomasi`:\
    https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541


.. autosummary::
    :toctree: ../generated/

    ShiTomasi_detection
    morph_opening
"""

import numpy as np
from numpy.ma.core import MaskedArray

from pysteps.exceptions import MissingOptionalDependency

try:
    import cv2

    CV2_IMPORTED = True
except ImportError:
    CV2_IMPORTED = False


def ShiTomasi_detection(input_image, max_corners=500, quality_level=0.1,
                        min_distance=3, block_size=15, buffer_mask=0,
                        use_harris=False, k=0.04,
                        verbose=False,
                        **kwargs):
    """
    Interface to the OpenCV `Shi-Tomasi`_ features detection method to detect
    corners in an image.

    Corners are used for local tracking methods.

    .. _`Shi-Tomasi`:\
        https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541

    .. _MaskedArray:\
        https://docs.scipy.org/doc/numpy/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray

    .. _`Harris detector`:\
        https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345

    .. _cornerMinEigenVal:\
        https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#ga3dbce297c1feb859ee36707e1003e0a8


    Parameters
    ----------

    input_image : array_like or MaskedArray_
        Array of shape (m, n) containing the input image.

        In case of array_like, invalid values (Nans or infs) are masked,
        otherwise the mask of the MaskedArray_ is used. Such mask defines a
        region where features are not detected.

        The fill value for the masked pixels is taken as the minimum of all
        valid pixels.

    max_corners : int, optional
        The **maxCorners** parameter in the `Shi-Tomasi`_ corner detection
        method.
        It represents the maximum number of points to be tracked (corners).
        If set to zero, all detected corners are used.

    quality_level : float, optional
        The **qualityLevel** parameter in the `Shi-Tomasi`_ corner detection
        method.
        It represents the minimal accepted quality for the image corners.

    min_distance : int, optional
        The **minDistance** parameter in the `Shi-Tomasi`_ corner detection
        method.
        It represents minimum possible Euclidean distance in pixels between
        corners.

    block_size : int, optional
        The **blockSize** parameter in the `Shi-Tomasi`_ corner detection
        method.
        It represents the window size in pixels used for computing a derivative
        covariation matrix over each pixel neighborhood.

    use_harris : bool, optional
        Whether to use a `Harris detector`_  or cornerMinEigenVal_.

    k : float, optional
        Free parameter of the Harris detector.

    buffer_mask : int, optional
        A mask buffer width in pixels. This extends the input mask (if any)
        to limit edge effects.

    verbose : bool, optional
        Print the number of features detected.

    Returns
    -------

    points : array_like
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

    input_image = np.copy(input_image)

    if input_image.ndim != 2:
        raise ValueError("input_image must be a two-dimensional array")

    # masked array
    if ~isinstance(input_image, MaskedArray):
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
    input_image = (
        (input_image.filled() - input_image.min())
        / (input_image.max() - input_image.min())
        * 255
    )

    # convert to 8-bit
    input_image = np.ndarray.astype(input_image, "uint8")
    mask = (-1 * mask + 1).astype("uint8")

    params = dict(
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        useHarrisDetector=use_harris,
        k=k,
    )
    points = cv2.goodFeaturesToTrack(input_image, mask=mask, **params)
    if points is None:
        points = np.empty(shape=(0, 2))
    else:
        points = points.squeeze()

    if verbose:
        print("--- %i good features to track detected ---" % points.shape[0])

    return points


def morph_opening(input_image, thr, n):
    """Filter out small scale noise on the image by applying a binary
    morphological opening, that is, erosion followed by dilation.

    Parameters
    ----------

    input_image : array_like
        Array of shape (m, n) containing the input image.

    thr : float
        The threshold used to convert the image into a binary image.

    n : int
        The structuring element size [pixels].

    Returns
    -------

    input_image : array_like
        Array of shape (m,n) containing the filtered image.
    """
    if not CV2_IMPORTED:
        raise MissingOptionalDependency(
            "opencv package is required for the morphologyEx "
            "routine but it is not installed"
        )

    # Convert to binary image
    field_bin = np.ndarray.astype(input_image > thr, "uint8")

    # Build a structuring element of size n
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

    # Apply morphological opening (i.e. erosion then dilation)
    field_bin_out = cv2.morphologyEx(field_bin, cv2.MORPH_OPEN, kernel)

    # Build mask to be applied on the original image
    mask = (field_bin - field_bin_out) > 0

    # Filter out small isolated pixels based on mask
    input_image[mask] = np.nanmin(input_image)

    return input_image
