# -*- coding: utf-8 -*-
"""
pysteps.motion.lucaskanade
==========================

The Lucas-Kanade (LK) Module.

This module implements the interface to the local Lucas-Kanade routine available
in OpenCV, as well as other auxiliary methods such as the interpolation of the
LK vectors over a grid.

.. _`goodFeaturesToTrack()`:\
    https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541


.. _`calcOpticalFlowPyrLK()`:\
   https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

.. autosummary::
    :toctree: ../generated/

    dense_lucaskanade
    track_features

"""

import numpy as np
from numpy.ma.core import MaskedArray

from pysteps.decorators import check_input_frames
from pysteps.exceptions import MissingOptionalDependency

from pysteps import utils
from pysteps.utils.cleansing import decluster, detect_outliers
from pysteps.utils.images import morph_opening

try:
    import cv2

    CV2_IMPORTED = True
except ImportError:
    CV2_IMPORTED = False

import time
import warnings


@check_input_frames(2)
def dense_lucaskanade(
    input_images,
    lk_kwargs=None,
    fd_method="ShiTomasi",
    fd_kwargs=None,
    interp_method="rbfinterp2d",
    interp_kwargs=None,
    dense=True,
    nr_std_outlier=3,
    k_outlier=30,
    size_opening=3,
    decl_scale=10,
    verbose=False
):
    """Run the Lucas-Kanade optical flow and interpolate the motion vectors.

    .. _opencv: https://opencv.org/

    .. _`Lucas-Kanade`: https://docs.opencv.org/3.4/dc/d6b/\
    group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

    .. _MaskedArray: https://docs.scipy.org/doc/numpy/reference/\
        maskedarray.baseclass.html#numpy.ma.MaskedArray

    .. _Shi-Tomasi: https://docs.opencv.org/3.4.1/dd/d1a/group__\
        imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541

    Interface to the OpenCV_ implementation of the local `Lucas-Kanade`_ optical
    flow method applied in combination to the `Shi-Tomasi`_ corner detection
    routine. The sparse motion vectors are finally interpolated to return the whole
    motion field.

    Parameters
    ----------

    input_images : array_like or MaskedArray_
        Array of shape (T, m, n) containing a sequence of T two-dimensional input
        images of shape (m, n). T = 2 is the minimum required number of images.
        With T > 2, the sparse vectors detected by Lucas-Kanade are pooled
        together prior to the final interpolation.

        In case of an array_like, invalid values (Nans or infs) are masked.
        The mask in the MaskedArray_ defines a region where velocity vectors are
        not computed.

    lk_kwargs : dict, optional
        Optional dictionary containing keyword arguments for the Lucas-Kanade
        features tracking algorithm. See the documentation of 
        pysteps.motion.lucaskanade.track_features.

    fd_method : {"ShiTomasi"}, optional
      Name of the feature detection method to use. See the documentation
      of pysteps.utils.interpolate.

    fd_kwargs : dict, optional
        Optional dictionary containing keyword arguments for the features detection
        algorithm. See the documentation of pysteps.utils.iamges.corner_detection.

    interp_method : {"rbfinterp2d"}, optional
      Name of the interpolation method to use. See the documentation
      of pysteps.utils.interpolate.

    interp_kwargs : dict, optional
        Optional dictionary containing keyword arguments for the interpolation
        algorithm. See the documentation of pysteps.utils.interpolate.

    dense : bool, optional
        If True (the default), it returns the three-dimensional array (2,m,n)
        containing the dense x- and y-components of the motion field. If false,
        it returns the sparse motion vectors as 1D arrays x, y, u, v, where
        x, y define the vector locations, u, v define the x and y direction
        components of the vectors.

    nr_std_outlier : int, optional
        Maximum acceptable deviation from the mean in terms of number of
        standard deviations. Any anomaly larger than this value is flagged as
        outlier and excluded from the interpolation.

    k_outlier : int or None, optional
        The number of nearest neighbours used to localize the outlier detection.

        If set to None, it employs all the data points (global detection).

    size_opening : int, optional
        The size of the structuring element kernel in pixels. This is used to
        perform a binary morphological opening on the input fields in order to
        filter isolated echoes due to clutter.

        If set to zero, the fitlering is not perfomed.

    decl_scale : int, optional
        The scale declustering parameter in pixels used to reduce the number of
        redundant sparse vectors before the interpolation.
        Sparse vectors within this declustering scale are averaged together.

        If set to less than 2 pixels, the declustering is not perfomed.

    verbose : bool, optional
        If set to True, it prints information about the program.

    Returns
    -------

    out : array_like or tuple
        If dense=True (the default), it returns the three-dimensional array (2,m,n)
        containing the dense x- and y-components of the motion field in units of
        pixels / timestep as given by the input array input_images.

        If dense=False, it returns a tuple containing the 2-dimensional arrays
        xy and uv, where x, y define the vector locations, u, v define the x
        and y direction components of the vectors.

        Return a zero motion field when no motion is detected.

    References
    ----------

    Bouguet,  J.-Y.:  Pyramidal  implementation  of  the  affine  Lucas Kanade
    feature tracker description of the algorithm, Intel Corp., 5, 4,
    https://doi.org/10.1109/HPDC.2004.1323531, 2001

    Lucas, B. D. and Kanade, T.: An iterative image registration technique with
    an application to stereo vision, in: Proceedings of the 1981 DARPA Imaging
    Understanding Workshop, pp. 121–130, 1981.
    """

    input_images = input_images.copy()

    if fd_kwargs is None:
        fd_kwargs = dict()

    if lk_kwargs is None:
        lk_kwargs = dict()

    if interp_kwargs is None:
        interp_kwargs = dict()

    if verbose:
        print("Computing the motion field with the Lucas-Kanade method.")
        t0 = time.time()

    nr_fields = input_images.shape[0]
    domain_size = (input_images.shape[1], input_images.shape[2])

    feature_detection_method = utils.get_method(fd_method)
    interpolation_method = utils.get_method(interp_method)

    xy = np.empty(shape=(0, 2))
    uv = np.empty(shape=(0, 2))
    for n in range(nr_fields - 1):

        # extract consecutive images
        prvs = input_images[n, :, :].copy()
        next = input_images[n + 1, :, :].copy()

        if ~isinstance(prvs, MaskedArray):
            prvs = np.ma.masked_invalid(prvs)
        np.ma.set_fill_value(prvs, prvs.min())

        if ~isinstance(next, MaskedArray):
            next = np.ma.masked_invalid(next)
        np.ma.set_fill_value(next, next.min())

        # remove small noise with a morphological operator (opening)
        if size_opening > 0:
            prvs = morph_opening(prvs, prvs.min(), size_opening)
            next = morph_opening(next, next.min(), size_opening)

        # features detection
        points = feature_detection_method(prvs, **fd_kwargs)

        # skip loop if no features to track
        if points.shape[0] == 0:
            continue

        # get sparse u, v vectors with Lucas-Kanade tracking
        xy_, uv_ = track_features(prvs, next, points, **lk_kwargs)

        # skip loop if no vectors
        if xy_.shape[0] == 0:
            continue

        # stack vectors
        xy = np.append(xy, xy_, axis=0)
        uv = np.append(uv, uv_, axis=0)

    # return zero motion field is no sparse vectors are found
    if xy.shape[0] == 0:
        if dense:
            return np.zeros((2, domain_size[0], domain_size[1]))
        else:
            return xy, uv

    # detect and remove outliers
    outliers = detect_outliers(uv, nr_std_outlier, xy, k_outlier, verbose)
    xy = xy[~outliers, :]
    uv = uv[~outliers, :]

    if verbose:
        print("--- LK found %i sparse vectors ---" % xy.shape[0])

    # return sparse vectors if required
    if not dense:
        return xy, uv

    # decluster sparse motion vectors
    if decl_scale > 1:
        xy, uv = decluster(xy, uv, decl_scale, 1, verbose)

    # return zero motion field if no sparse vectors are left for interpolation
    if xy.shape[0] == 0:
        return np.zeros((2, domain_size[0], domain_size[1]))

    # interpolation
    xgrid = np.arange(domain_size[1])
    ygrid = np.arange(domain_size[0])
    UV = interpolation_method(xy, uv, xgrid, ygrid, **interp_kwargs)

    if verbose:
        print("--- total time: %.2f seconds ---" % (time.time() - t0))

    return UV


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
    Interface to the OpenCV calcOpticalFlowPyrLK_ features tracking algorithm.

    .. _calcOpticalFlowPyrLK:\
       https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

    .. _MaskedArray:\
        https://docs.scipy.org/doc/numpy/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray

    Parameters
    ----------

    prvs_image : array_like or MaskedArray_
        Array of shape (m, n) containing the first image.
        Invalid values (Nans or infs) are filled using the min value.

    next_image : array_like or MaskedArray_
        Array of shape (m, n) containing the successive image.
        Invalid values (Nans or infs) are filled using the min value.

    points : array_like
        Array of shape (p, 2) indicating the pixel coordinates of the
        tracking points (corners).

    winsize : tuple of int, optional
        The winSize parameter in calcOpticalFlowPyrLK_.
        It represents the size of the search window that it is used at each
        pyramid level.

    nr_levels : int, optional
        The maxLevel parameter in calcOpticalFlowPyrLK_.
        It represents the 0-based maximal pyramid level number.

    criteria : tuple of int, optional
        The TermCriteria parameter in calcOpticalFlowPyrLK_ ,
        which specifies the termination criteria of the iterative search algorithm

    flags : int, optional
        Operation flags, see documentation calcOpticalFlowPyrLK_.

    min_eig_thr : float, optional
        The minEigThreshold parameter in calcOpticalFlowPyrLK_.

    verbose : bool, optional
        Print the number of vectors that have been found.

    Returns
    -------

    xy : array_like
        Array of shape (d, 2) with the x- and y-coordinates of d <= p detected
        sparse motion vectors.

    uv : array_like
        Array of shape (d, 2) with the u- and v-components of d <= p detected
        sparse motion vectors.

    Notes
    -----

    The tracking points can be obtained with the pysteps.utils.images.corner_detection
    routine.
    """
    if not CV2_IMPORTED:
        raise MissingOptionalDependency(
            "opencv package is required for the calcOpticalFlowPyrLK() "
            "routine but it is not installed"
        )

    prvs = np.copy(prvs_image)
    next = np.copy(next_image)
    p0 = np.copy(points)

    if ~isinstance(prvs, MaskedArray):
        prvs = np.ma.masked_invalid(prvs)
    np.ma.set_fill_value(prvs, prvs.min())

    if ~isinstance(next, MaskedArray):
        next = np.ma.masked_invalid(next)
    np.ma.set_fill_value(next, next.min())

    # scale between 0 and 255
    prvs = (prvs.filled() - prvs.min()) / (prvs.max() - prvs.min()) * 255
    next = (next.filled() - next.min()) / (next.max() - next.min()) * 255

    # convert to 8-bit
    prvs = np.ndarray.astype(prvs, "uint8")
    next = np.ndarray.astype(next, "uint8")

    # Lucas-Kanade
    # TODO: use the error returned by the OpenCV routine
    params = dict(
        winSize=winsize,
        maxLevel=nr_levels,
        criteria=criteria,
        flags=flags,
        minEigThreshold=min_eig_thr,
    )
    p1, st, __ = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **params)

    # keep only features that have been found
    st = st.squeeze() == 1
    if np.any(st):
        p1 = p1[st, :]
        p0 = p0[st, :]

        # extract vectors
        xy = p0
        uv = p1 - p0

    else:
        xy = uv = np.empty(shape=(0, 2))

    if verbose:
        print("--- %i sparse vectors found ---" % xy.shape[0])

    return xy, uv
