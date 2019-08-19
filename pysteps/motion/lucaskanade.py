# -*- coding: utf-8 -*-
"""
pysteps.motion.lucaskanade
==========================

The Lucas-Kanade (LK) local feature tracking module.

This module implements the interface to the local `Lucas-Kanade`_ routine
available in OpenCV_.

For its dense method, it additionally interpolates the sparse vectors over a
regular grid to return a motion field.

.. _OpenCV: https://opencv.org/

.. _`Lucas-Kanade`:\
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
def dense_lucaskanade(input_images,
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
                      verbose=False):
    """Run the Lucas-Kanade optical flow routine and interpolate the motion
    vectors.

    .. _OpenCV: https://opencv.org/

    .. _`Lucas-Kanade`:\
        https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

    .. _MaskedArray:\
        https://docs.scipy.org/doc/numpy/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray

    Interface to the OpenCV_ implementation of the local `Lucas-Kanade`_ optical
    flow method applied in combination to a feature detection routine.

    The sparse motion vectors are finally interpolated to return the whole
    motion field.

    Parameters
    ----------

    input_images : array_like or MaskedArray_
        Array of shape (T, m, n) containing a sequence of *T* two-dimensional
        input images of shape (m, n). The indexing order in **input_images** is
        assumed to be (time, latitude, longitude).

        *T* = 2 is the minimum required number of images.
        With *T* > 2, all the resulting sparse vectors are pooled together for
        the final interpolation on a regular grid.

        In case of array_like, invalid values (Nans or infs) are masked,
        otherwise the mask of the MaskedArray_ is used. Such mask defines a
        region where features are not detected for the tracking algorithm.

    lk_kwargs : dict, optional
        Optional dictionary containing keyword arguments for the `Lucas-Kanade`_
        features tracking algorithm. See the documentation of
        :py:func:`pysteps.motion.lucaskanade.track_features`.

    fd_method : {"ShiTomasi"}, optional
      Name of the feature detection routine. See feature detection methods in
      :py:mod:`pysteps.utils.images`.

    fd_kwargs : dict, optional
        Optional dictionary containing keyword arguments for the features
        detection algorithm.
        See the documentation of :py:mod:`pysteps.utils.images`.

    interp_method : {"rbfinterp2d"}, optional
      Name of the interpolation method to use. See interpolation methods in
      :py:mod:`pysteps.utils.interpolate`.

    interp_kwargs : dict, optional
        Optional dictionary containing keyword arguments for the interpolation
        algorithm. See the documentation of :py:mod:`pysteps.utils.interpolate`.

    dense : bool, optional
        If True, return the three-dimensional array (2, m, n) containing
        the dense x- and y-components of the motion field.

        If False, return the sparse motion vectors as 2-D **xy** and **uv**
        arrays, where **xy** defines the vector positions, **uv** defines the
        x and y direction components of the vectors.

    nr_std_outlier : int, optional
        Maximum acceptable deviation from the mean in terms of number of
        standard deviations. Any sparse vector with a deviation larger than
        this threshold is flagged as outlier and excluded from the
        interpolation.
        See the documentation of
        :py:func:`pysteps.utils.cleansing.detect_outliers`.

    k_outlier : int or None, optional
        The number of nearest neighbours used to localize the outlier detection.
        If set to None, it employs all the data points (global detection).
        See the documentation of
        :py:func:`pysteps.utils.cleansing.detect_outliers`.

    size_opening : int, optional
        The size of the structuring element kernel in pixels. This is used to
        perform a binary morphological opening on the input fields in order to
        filter isolated echoes due to clutter. If set to zero, the filtering
        is not perfomed.
        See the documentation of
        :py:func:`pysteps.utils.images.morph_opening`.

    decl_scale : int, optional
        The scale declustering parameter in pixels used to reduce the number of
        redundant sparse vectors before the interpolation.
        Sparse vectors within this declustering scale are averaged together.
        If set to less than 2 pixels, the declustering is not perfomed.
        See the documentation of
        :py:func:`pysteps.utils.cleansing.decluster`.

    verbose : bool, optional
        If set to True, print some information about the program.

    Returns
    -------

    out : array_like or tuple
        If **dense=True** (the default), return the advection field having shape
        (2, m, n), where out[0, :, :] contains the x-components of the motion
        vectors and out[1, :, :] contains the y-components.
        The velocities are in units of pixels / timestep, where timestep is the
        time difference between the two input images.
        Return a zero motion field of shape (2, m, n) when no motion is
        detected.

        If **dense=False**, it returns a tuple containing the 2-dimensional
        arrays **xy** and **uv**, where x, y define the vector locations,
        u, v define the x and y direction components of the vectors.
        Return two empty arrays when no motion is detected.

    See also
    --------

    pysteps.motion.lucaskanade.track_features

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

    if verbose:
        print("Computing the motion field with the Lucas-Kanade method.")
        t0 = time.time()

    nr_fields = input_images.shape[0]
    domain_size = (input_images.shape[1], input_images.shape[2])

    feature_detection_method = utils.get_method(fd_method)
    interpolation_method = utils.get_method(interp_method)

    if fd_kwargs is None:
        fd_kwargs = dict()

    if lk_kwargs is None:
        lk_kwargs = dict()

    if interp_kwargs is None:
        interp_kwargs = dict()

    xy = np.empty(shape=(0, 2))
    uv = np.empty(shape=(0, 2))
    for n in range(nr_fields - 1):

        # extract consecutive images
        prvs_img = input_images[n, :, :].copy()
        next_img = input_images[n + 1, :, :].copy()

        if ~isinstance(prvs_img, MaskedArray):
            prvs_img = np.ma.masked_invalid(prvs_img)
        np.ma.set_fill_value(prvs_img, prvs_img.min())

        if ~isinstance(next_img, MaskedArray):
            next_img = np.ma.masked_invalid(next_img)
        np.ma.set_fill_value(next_img, next_img.min())

        # remove small noise with a morphological operator (opening)
        if size_opening > 0:
            prvs_img = morph_opening(prvs_img, prvs_img.min(), size_opening)
            next_img = morph_opening(next_img, next_img.min(), size_opening)

        # features detection
        points = feature_detection_method(prvs_img, **fd_kwargs)

        # skip loop if no features to track
        if points.shape[0] == 0:
            continue

        # get sparse u, v vectors with Lucas-Kanade tracking
        xy_, uv_ = track_features(prvs_img, next_img, points, **lk_kwargs)

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
    Interface to the OpenCV `Lucas-Kanade`_ features tracking algorithm
    (cv.calcOpticalFlowPyrLK).

    .. _`Lucas-Kanade`:\
       https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

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
        The **winSize** parameter in calcOpticalFlowPyrLK_.
        It represents the size of the search window that it is used at each
        pyramid level.

    nr_levels : int, optional
        The **maxLevel** parameter in calcOpticalFlowPyrLK_.
        It represents the 0-based maximal pyramid level number.

    criteria : tuple of int, optional
        The **TermCriteria** parameter in calcOpticalFlowPyrLK_ ,
        which specifies the termination criteria of the iterative search
        algorithm.

    flags : int, optional
        Operation flags, see documentation calcOpticalFlowPyrLK_.

    min_eig_thr : float, optional
        The **minEigThreshold** parameter in calcOpticalFlowPyrLK_.

    verbose : bool, optional
        Print the number of vectors that have been found.

    Returns
    -------

    xy : array_like
        Array of shape (d, 2) with the x- and y-coordinates of *d* <= *p*
        detected sparse motion vectors.

    uv : array_like
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
    feature tracker description of the algorithm, Intel Corp., 5, 4,
    https://doi.org/10.1109/HPDC.2004.1323531, 2001

    Lucas, B. D. and Kanade, T.: An iterative image registration technique with
    an application to stereo vision, in: Proceedings of the 1981 DARPA Imaging
    Understanding Workshop, pp. 121–130, 1981.
    """

    if not CV2_IMPORTED:
        raise MissingOptionalDependency(
            "opencv package is required for the calcOpticalFlowPyrLK() "
            "routine but it is not installed"
        )

    prvs_img = np.copy(prvs_image)
    next_img = np.copy(next_image)
    p0 = np.copy(points)

    if ~isinstance(prvs_img, MaskedArray):
        prvs_img = np.ma.masked_invalid(prvs_img)
    np.ma.set_fill_value(prvs_img, prvs_img.min())

    if ~isinstance(next_img, MaskedArray):
        next_img = np.ma.masked_invalid(next_img)
    np.ma.set_fill_value(next_img, next_img.min())

    # scale between 0 and 255
    prvs_img = ((prvs_img.filled() - prvs_img.min()) /
                (prvs_img.max() - prvs_img.min()) * 255)

    next_img = ((next_img.filled() - next_img.min()) /
                (next_img.max() - next_img.min()) * 255)

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
    p1, st, __ = cv2.calcOpticalFlowPyrLK(prvs_img, next_img,
                                          p0, None, **params)

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
