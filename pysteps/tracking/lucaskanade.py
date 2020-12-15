# -*- coding: utf-8 -*-
"""
pysteps.tracking.lucaskanade
============================

The Lucas-Kanade (LK) feature tracking module.

This module implements the interface to the local `Lucas-Kanade`_ routine
available in OpenCV_.

.. _OpenCV: https://opencv.org/

.. _`Lucas-Kanade`:\
    https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

.. autosummary::
    :toctree: ../generated/

    track_features
"""

import numpy as np
from numpy.ma.core import MaskedArray

from pysteps.exceptions import MissingOptionalDependency

from pysteps import utils

try:
    import cv2

    CV2_IMPORTED = True
except ImportError:
    CV2_IMPORTED = False


def track_features(
    prvs_image,
    next_image,
    points,
    winsize=(50, 50),
    nr_levels=3,
    criteria=(3, 10, 0),
    flags=0,
    min_eig_thr=1e-4,
    verbose=False,
):
    """
    Interface to the OpenCV `Lucas-Kanade`_ feature tracking algorithm
    (cv.calcOpticalFlowPyrLK).

    .. _`Lucas-Kanade`:\
       https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

    .. _calcOpticalFlowPyrLK:\
       https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323


    .. _MaskedArray:\
        https://docs.scipy.org/doc/numpy/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    Parameters
    ----------
    prvs_image: ndarray_ or MaskedArray_
        Array of shape (m, n) containing the first image.
        Invalid values (Nans or infs) are replaced with the min value.
    next_image: ndarray_ or MaskedArray_
        Array of shape (m, n) containing the successive image.
        Invalid values (Nans or infs) are replaced with the min value.
    points: array_like
        Array of shape (p, 2) indicating the pixel coordinates of the
        tracking points (corners).
    winsize: tuple of int, optional
        The **winSize** parameter in calcOpticalFlowPyrLK_.
        It represents the size of the search window that it is used at each
        pyramid level. The default is (50, 50).
    nr_levels: int, optional
        The **maxLevel** parameter in calcOpticalFlowPyrLK_.
        It represents the 0-based maximal pyramid level number.
        The default is 3.
    criteria: tuple of int, optional
        The **TermCriteria** parameter in calcOpticalFlowPyrLK_ ,
        which specifies the termination criteria of the iterative search
        algorithm. The default is (3, 10, 0).
    flags: int, optional
        Operation flags, see documentation calcOpticalFlowPyrLK_. The
        default is 0.
    min_eig_thr: float, optional
        The **minEigThreshold** parameter in calcOpticalFlowPyrLK_. The
        default is 1e-4.
    verbose: bool, optional
        Print the number of vectors that have been found. The default
        is False.

    Returns
    -------
    xy: ndarray_
        Array of shape (d, 2) with the x- and y-coordinates of *d* <= *p*
        detected sparse motion vectors.
    uv: ndarray_
        Array of shape (d, 2) with the u- and v-components of *d* <= *p*
        detected sparse motion vectors.

    Notes
    -----
    The tracking points can be obtained with the
    :py:func:`pysteps.utils.images.ShiTomasi_detection` routine.

    See also
    --------
    pysteps.motion.lucaskanade.dense_lucaskanade

    References
    ----------
    Bouguet,  J.-Y.:  Pyramidal  implementation  of  the  affine  Lucas Kanade
    feature tracker description of the algorithm, Intel Corp., 5, 4, 2001

    Lucas, B. D. and Kanade, T.: An iterative image registration technique with
    an application to stereo vision, in: Proceedings of the 1981 DARPA Imaging
    Understanding Workshop, pp. 121–130, 1981.
    """

    if not CV2_IMPORTED:
        raise MissingOptionalDependency(
            "opencv package is required for the calcOpticalFlowPyrLK() "
            "routine but it is not installed"
        )

    prvs_img = prvs_image.copy()
    next_img = next_image.copy()
    p0 = np.copy(points)

    # Check if a MaskedArray is used. If not, mask the ndarray
    if not isinstance(prvs_img, MaskedArray):
        prvs_img = np.ma.masked_invalid(prvs_img)
    np.ma.set_fill_value(prvs_img, prvs_img.min())

    if not isinstance(next_img, MaskedArray):
        next_img = np.ma.masked_invalid(next_img)
    np.ma.set_fill_value(next_img, next_img.min())

    # scale between 0 and 255
    im_min = prvs_img.min()
    im_max = prvs_img.max()
    if (im_max - im_min) > 1e-8:
        prvs_img = (prvs_img.filled() - im_min) / (im_max - im_min) * 255
    else:
        prvs_img = prvs_img.filled() - im_min

    im_min = next_img.min()
    im_max = next_img.max()
    if (im_max - im_min) > 1e-8:
        next_img = (next_img.filled() - im_min) / (im_max - im_min) * 255
    else:
        next_img = next_img.filled() - im_min

    # convert to 8-bit
    prvs_img = np.ndarray.astype(prvs_img, "uint8")
    next_img = np.ndarray.astype(next_img, "uint8")

    # Lucas-Kanade
    # TODO: use the error returned by the OpenCV routine
    params = dict(
        winSize=winsize,
        maxLevel=nr_levels,
        criteria=criteria,
        flags=flags,
        minEigThreshold=min_eig_thr,
    )
    p1, st, __ = cv2.calcOpticalFlowPyrLK(prvs_img, next_img, p0, None, **params)

    # keep only features that have been found
    st = np.atleast_1d(st.squeeze()) == 1
    if np.any(st):
        p1 = p1[st, :]
        p0 = p0[st, :]

        # extract vectors
        xy = p0
        uv = p1 - p0

    else:
        xy = uv = np.empty(shape=(0, 2))

    if verbose:
        print(f"--- {xy.shape[0]} sparse vectors found ---")

    return xy, uv
