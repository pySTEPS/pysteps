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
    features_to_track
    track_features
    morph_opening
    detect_outliers

"""

import numpy as np
from numpy.ma.core import MaskedArray

from pysteps.decorators import check_input_frames
from pysteps.exceptions import MissingOptionalDependency

from pysteps.postprocessing.interpolate import (
    decluster_sparse_data,
    rbfinterp2d,
)

try:
    import cv2

    CV2_IMPORTED = True
except ImportError:
    CV2_IMPORTED = False

import scipy.spatial
import time
import warnings


@check_input_frames(2)
def dense_lucaskanade(input_images, **kwargs):
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

    Other Parameters
    ----------------
    dense : bool, optional
        If True (the default), it returns the three-dimensional array (2,m,n)
        containing the dense x- and y-components of the motion field. If false,
        it returns the sparse motion vectors as 1D arrays x, y, u, v, where
        x, y define the vector locations, u, v define the x and y direction
        components of the vectors.

    buffer_mask : int, optional
        A mask buffer width in pixels. This extends the input mask (if any)
        to help avoiding the erroneous interpretation of velocities near the
        maximum range of the radars (0 by default).

    max_corners_ST : int, optional
        The maxCorners parameter in the `Shi-Tomasi`_ corner detection method.
        It represents the maximum number of points to be tracked (corners),
        by default this is 500. If set to zero, all detected corners are used.

    quality_level_ST : float, optional
        The qualityLevel parameter in the `Shi-Tomasi`_ corner detection method.
        It represents the minimal accepted quality for the points to be tracked
        (corners), by default this is set to 0.1. Higher quality thresholds can
        lead to no detection at all.

    min_distance_ST : int, optional
        The minDistance parameter in the `Shi-Tomasi`_ corner detection method.
        It represents minimum possible Euclidean distance in pixels
        between corners, by default this is set to 3 pixels.

    block_size_ST : int, optional
        The blockSize parameter in the `Shi-Tomasi`_ corner detection method.
        It represents the window size in pixels used for computing a derivative
        covariation matrix over each pixel neighborhood, by default this is set
        to 15 pixels.

    winsize_LK : tuple of int, optional
        The winSize parameter in the `Lucas-Kanade`_ optical flow method.
        It represents the size of the search window that it is used at each
        pyramid level, by default this is set to (50, 50) pixels.

    nr_levels_LK : int, optional
        The maxLevel parameter in the `Lucas-Kanade`_ optical flow method.
        It represents the 0-based maximal pyramid level number, by default this
        is set to 3.

    nr_std_outlier : int, optional
        Maximum acceptable deviation from the mean in terms of number of
        standard deviations. Any anomaly larger than this value is flagged as
        outlier and excluded from the interpolation.
        By default this is set to 3.

    k_outlier : int or None, optional
        The number of nearest neighbours used to localize the outlier detection.
        If set to None, it employs all the data points (global detection).
        The default is 30.

    size_opening : int, optional
        The size of the structuring element kernel in pixels. This is used to
        perform a binary morphological opening on the input fields in order to
        filter isolated echoes due to clutter. By default this is set to 3.
        If set to zero, the fitlering is not perfomed.

    decl_scale : int, optional
        The scale declustering parameter in pixels used to reduce the number of
        redundant sparse vectors before the interpolation.
        Sparse vectors within this declustering scale are averaged together.
        By default this is set to 20 pixels. If set to less than 2 pixels, the
        declustering is not perfomed.

    min_decl_samples : int, optional
        The minimum number of samples necessary for computing the median vector
        within given declustering cell, otherwise all sparse vectors in that
        cell are discarded. By default this is set to 2.

    rbfunction : {"gaussian", "multiquadric", "inverse quadratic", "inverse
        multiquadric", "bump"}, optional
        The name of one of the available radial basis function based on the
        Euclidean norm. "gaussian" by default.

    k : int or None, optional
        The number of nearest neighbours used to speed-up the interpolation.
        If set to None, it interpolates based on all the data points.
        This is 50 by default.

    epsilon : float, optional
        The shape parameter > 0 used to scale the input to the radial kernel.
        It defaults to 1.0.

    nchunks : int, optional
        Split the grid points in n chunks to limit the memory usage during the
        interpolation. By default this is set to 5, if set to 1 the interpolation
        is computed with the whole grid.

    verbose : bool, optional
        If set to True, it prints information about the program (True by default).

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

    # defaults
    dense = kwargs.get("dense", True)
    max_corners_ST = kwargs.get("max_corners_ST", 500)
    quality_level_ST = kwargs.get("quality_level_ST", 0.1)
    min_distance_ST = kwargs.get("min_distance_ST", 3)
    block_size_ST = kwargs.get("block_size_ST", 15)
    winsize_LK = kwargs.get("winsize_LK", (50, 50))
    nr_levels_LK = kwargs.get("nr_levels_LK", 3)
    nr_std_outlier = kwargs.get("nr_std_outlier", 3)
    nr_IQR_outlier = kwargs.get("nr_IQR_outlier", None)
    if nr_IQR_outlier is not None:
        nr_std_outlier = nr_IQR_outlier
        warnings.warn(
            "the 'nr_IQR_outlier' argument will be deprecated in the next release; "
            + "use 'nr_std_outlier' instead.",
            category=FutureWarning,
        )
    k_outlier = kwargs.get("k_outlier", 30)
    size_opening = kwargs.get("size_opening", 3)
    decl_scale = kwargs.get("decl_scale", 20)
    min_decl_samples = kwargs.get("min_decl_samples", 2)
    rbfunction = kwargs.get("rbfunction", "gaussian")
    k = kwargs.get("k", 50)
    epsilon = kwargs.get("epsilon", 1.0)
    nchunks = kwargs.get("nchunks", 5)
    verbose = kwargs.get("verbose", True)
    buffer_mask = kwargs.get("buffer_mask", 10)

    if verbose:
        print("Computing the motion field with the Lucas-Kanade method.")
        t0 = time.time()

    nr_fields = input_images.shape[0]
    domain_size = (input_images.shape[1], input_images.shape[2])

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

        # find good features to track
        gf_params = dict(
            maxCorners=max_corners_ST,
            qualityLevel=quality_level_ST,
            minDistance=min_distance_ST,
            blockSize=block_size_ST,
        )
        points = features_to_track(prvs, gf_params, buffer_mask, False)

        # skip loop if no features to track
        if points.shape[0] == 0:
            continue

        # get sparse u, v vectors with Lucas-Kanade tracking
        lk_params = dict(
            winSize=winsize_LK,
            maxLevel=nr_levels_LK,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0),
        )
        xy_, uv_ = track_features(prvs, next, points, lk_params, False)

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
        xy, uv = decluster_sparse_data(
            xy, uv, decl_scale, min_decl_samples, verbose
        )

    # return zero motion field if no sparse vectors are left for interpolation
    if xy.shape[0] == 0:
        return np.zeros((2, domain_size[0], domain_size[1]))

    # kernel interpolation
    xgrid = np.arange(domain_size[1])
    ygrid = np.arange(domain_size[0])
    UV = rbfinterp2d(
        xy,
        uv,
        xgrid,
        ygrid,
        rbfunction=rbfunction,
        epsilon=epsilon,
        k=k,
        nchunks=nchunks,
    )

    if verbose:
        print("--- total time: %.2f seconds ---" % (time.time() - t0))

    return UV


def features_to_track(input_image, params, buffer_mask=0, verbose=False):
    """
    Interface to the OpenCV `goodFeaturesToTrack()`_ method to detect strong corners
    on an image.

    Corners are used for local tracking methods.

    .. _`goodFeaturesToTrack()`:\
        https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541

    .. _MaskedArray: https://docs.scipy.org/doc/numpy/reference/\
        maskedarray.baseclass.html#numpy.ma.MaskedArray

    Parameters
    ----------

    input_image : array_like or MaskedArray_
        Array of shape (m, n) containing the input image.

        In case of an array_like, invalid values (Nans or infs) define a
        validity mask, which represents the region where velocity vectors are not
        computed. The corresponding fill value is taken as the minimum of all
        valid pixels.

    params : dict
        Any additional parameter to the original routine as described in the
        `goodFeaturesToTrack()`_ documentation.

    buffer_mask : int, optional
        A mask buffer width in pixels. This extends the input mask (if any)
        to help avoiding the erroneous interpretation of velocities near the
        maximum range of the radars (0 by default).

    verbose : bool, optional
        Print the number of features detected.

    Returns
    -------

    points : array_like
        Array of shape (p, 2) indicating the pixel coordinates of p detected
        corners.

    See also
    --------

    pysteps.motion.lucaskanade.track_features
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

    points = cv2.goodFeaturesToTrack(input_image, mask=mask, **params)
    if points is None:
        points = np.empty(shape=(0, 2))
    else:
        points = points.squeeze()

    if verbose:
        print("--- %i good features to track detected ---" % points.shape[0])

    return points


def track_features(prvs_image, next_image, points, params, verbose=False):
    """
    Interface to the OpenCV `calcOpticalFlowPyrLK()`_ features tracking algorithm.

    .. _`calcOpticalFlowPyrLK()`:\
       https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

    .. _MaskedArray: https://docs.scipy.org/doc/numpy/reference/\
        maskedarray.baseclass.html#numpy.ma.MaskedArray

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

    params : dict
        Any additional parameter to the original routine as described in the
        `calcOpticalFlowPyrLK()`_ documentation.

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

    See also
    --------

    pysteps.motion.lucaskanade.features_to_track
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


def morph_opening(input_image, thr, n):
    """Filter out small scale noise on the image by applying a binary morphological
    opening (i.e., erosion then dilation).

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


def detect_outliers(input_array, thr, coord=None, k=None, verbose=False):
    """Detect outliers in a (multivariate and georeferenced) dataset.

    Assume a (multivariate) Gaussian distribution and detect outliers based on
    the number of standard deviations from the mean.

    If spatial information is provided through coordinates, the outlier
    detection can be localized by considering only the k-nearest neighbours
    when computing the local mean and standard deviation.

    Parameters
    ----------

    input_array : array_like
        Array of shape (n) or (n, m), where n is the number of samples and m
        the number of variables. If m > 1, the Mahalanobis distance is used.
        All values in input_array are required to have finite values.

    thr : float
        The number of standard deviations from the mean that defines an outlier.

    coord : array_like, optional
        Array of shape (n, d) containing the coordinates of the input data into
        a space of d dimensions. Setting coord requires that k is not None.

    k : int or None, optional
        The number of nearest neighbours used to localize the outlier detection.
        If set to None (the default), it employs all the data points (global
        detection). Setting k requires that coord is not None.

    verbose : bool, optional
        Print out information.

    Returns
    -------

    out : array_like
        A boolean array of the same shape as input_array, with True values
        indicating the outliers detected in input_array.
    """

    input_array = np.copy(input_array)

    if np.any(~np.isfinite(input_array)):
        raise ValueError("input_array contains non-finite values")

    if input_array.ndim == 1:
        nvar = 1
    elif input_array.ndim == 2:
        nvar = input_array.shape[1]
    else:
        raise ValueError(
            "input_array must have 1 (n) or 2 dimensions (n, m), but it has %i"
            % coord.ndim
        )

    if coord is not None:

        coord = np.copy(coord)
        if coord.ndim == 1:
            coord = coord[:, None]

        elif coord.ndim > 2:
            raise ValueError(
                "coord must have 2 dimensions (n, d), but it has %i"
                % coord.ndim
            )

        if coord.shape[0] != input_array.shape[0]:
            raise ValueError(
                "the number of samples in input_array does not match the "
                + "number of coordinates %i!=%i"
                % (input_array.shape[0], coord.shape[0])
            )

        if k is None:
            raise ValueError("coord is set but k is None")

        k = np.min((coord.shape[0], k + 1))

    else:
        if k is not None:
            raise ValueError("k is set but coord=None")

    # global

    if k is None:

        if nvar == 1:

            # univariate

            zdata = (input_array - np.mean(input_array)) / np.std(input_array)
            outliers = zdata > thr

        else:

            # multivariate (mahalanobis distance)

            zdata = input_array - np.mean(input_array, axis=0)
            V = np.cov(zdata.T)
            VI = np.linalg.inv(V)
            try:
                VI = np.linalg.inv(V)
                MD = np.sqrt(np.dot(np.dot(zdata, VI), zdata.T).diagonal())
            except np.linalg.LinAlgError:
                MD = np.zeros(input_array.shape)
            outliers = MD > thr

    # local

    else:

        tree = scipy.spatial.cKDTree(coord)
        __, inds = tree.query(coord, k=k)
        outliers = np.empty(shape=0, dtype=bool)
        for i in range(inds.shape[0]):

            if nvar == 1:

                # in terms of velocity

                thisdata = input_array[i]
                neighbours = input_array[inds[i, 1:]]
                thiszdata = (thisdata - np.mean(neighbours)) / np.std(
                    neighbours
                )
                outliers = np.append(outliers, thiszdata > thr)

            else:

                # mahalanobis distance

                thisdata = input_array[i, :]
                neighbours = input_array[inds[i, 1:], :].copy()
                thiszdata = thisdata - np.mean(neighbours, axis=0)
                neighbours = neighbours - np.mean(neighbours, axis=0)
                V = np.cov(neighbours.T)
                try:
                    VI = np.linalg.inv(V)
                    MD = np.sqrt(np.dot(np.dot(thiszdata, VI), thiszdata.T))
                except np.linalg.LinAlgError:
                    MD = 0
                outliers = np.append(outliers, MD > thr)

    if verbose:
        print("--- %i outliers detected ---" % np.sum(outliers))

    return outliers
