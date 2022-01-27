# -*- coding: utf-8 -*-
"""
pysteps.utils.images
====================

Image processing routines for pysteps.

.. autosummary::
    :toctree: ../generated/

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


def morph_opening(input_image, thr, n):
    """
    Filter out small scale noise on the image by applying a binary
    morphological opening, that is, erosion followed by dilation.

    .. _MaskedArray:\
        https://docs.scipy.org/doc/numpy/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    Parameters
    ----------
    input_image: ndarray_ or MaskedArray_
        Array of shape (m, n) containing the input image.
    thr: float
        The threshold used to convert the image into a binary image.
    n: int
        The structuring element size [pixels].

    Returns
    -------
    input_image: ndarray_ or MaskedArray_
        Array of shape (m,n) containing the filtered image.
    """
    if not CV2_IMPORTED:
        raise MissingOptionalDependency(
            "opencv package is required for the morphologyEx "
            "routine but it is not installed"
        )

    input_image = input_image.copy()

    # Check if a MaskedArray is used. If not, mask the ndarray
    to_ndarray = False
    if not isinstance(input_image, MaskedArray):
        to_ndarray = True
        input_image = np.ma.masked_invalid(input_image)

    np.ma.set_fill_value(input_image, input_image.min())

    # Convert to binary image
    field_bin = np.ndarray.astype(input_image.filled() > thr, "uint8")

    # Build a structuring element of size n
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

    # Apply morphological opening (i.e. erosion then dilation)
    field_bin_out = cv2.morphologyEx(field_bin, cv2.MORPH_OPEN, kernel)

    # Build mask to be applied on the original image
    mask = (field_bin - field_bin_out) > 0

    # Filter out small isolated pixels based on mask
    input_image[mask] = np.nanmin(input_image)

    if to_ndarray:
        input_image = np.array(input_image)

    return input_image
