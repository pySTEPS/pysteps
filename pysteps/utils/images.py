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
import xarray as xr

from pysteps.exceptions import MissingOptionalDependency

try:
    import cv2

    CV2_IMPORTED = True
except ImportError:
    CV2_IMPORTED = False


def morph_opening(input_array, thr, kernel):
    """Filter out small scale noise on one image or sequence of images
    by applying a binary morphological opening, that is, erosion followed
    by dilation.

    Parameters
    ----------
    input_array: xr.DataArray
        Image or sequence of images as an array with dimensions (y, x) or (t, y, x).
    thr: float
        The threshold used to distinguish features from the background.
    kernel: int or 2D array_like
        The size of a rounded structuring element in number of pixels or
        an arbitrary kernel as a two-dimensional array.

    Returns
    -------
    input_image: xr.DataArray
        Array of shape (m,n) containing the filtered image.
    """
    if not CV2_IMPORTED:
        raise MissingOptionalDependency(
            "opencv package is required for the morphologyEx "
            "routine but it is not installed"
        )

    input_array = input_array.copy()
    fill_value = input_array.attrs.get("zerovalue", input_array.min())

    # Convert to binary image
    field_bin = (input_array > thr).astype(np.uint8)

    # Build a structuring element
    if isinstance(kernel, int):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel, kernel)
        )
    else:
        kernel = np.array(kernel, dtype=np.uint8)

    if "t" not in field_bin.dims:
        field_bin = field_bin.expand_dims("t")

    # Apply morphological opening (i.e. erosion then dilation)
    field_bin_out = xr.apply_ufunc(
        cv2.morphologyEx,
        field_bin.groupby("t"),
        cv2.MORPH_OPEN,
        kernel,
    )

    # Build mask to be applied on the original image
    mask_remove = (field_bin - field_bin_out) > 0

    if mask_remove.t.size == 1:
        mask_remove = mask_remove.squeeze("t")

    # Filter out small isolated pixels based on mask
    input_image = input_array.where(~mask_remove, fill_value)

    return input_image
