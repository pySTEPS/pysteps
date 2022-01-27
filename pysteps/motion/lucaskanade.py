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
"""

import numpy as np
from numpy.ma.core import MaskedArray

from pysteps.decorators import check_input_frames

from pysteps import utils, feature
from pysteps.tracking.lucaskanade import track_features
from pysteps.utils.cleansing import decluster, detect_outliers
from pysteps.utils.images import morph_opening

import time


@check_input_frames(2)
def dense_lucaskanade(
    input_images,
    lk_kwargs=None,
    fd_method="shitomasi",
    fd_kwargs=None,
    interp_method="idwinterp2d",
    interp_kwargs=None,
    dense=True,
    nr_std_outlier=3,
    k_outlier=30,
    size_opening=3,
    decl_scale=20,
    verbose=False,
):
    """
    Run the Lucas-Kanade optical flow routine and interpolate the motion
    vectors.

    .. _OpenCV: https://opencv.org/

    .. _`Lucas-Kanade`:\
        https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

    .. _MaskedArray:\
        https://docs.scipy.org/doc/numpy/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    Interface to the OpenCV_ implementation of the local `Lucas-Kanade`_ optical
    flow method applied in combination to a feature detection routine.

    The sparse motion vectors are finally interpolated to return the whole
    motion field.

    Parameters
    ----------
    input_images: ndarray_ or MaskedArray_
        Array of shape (T, m, n) containing a sequence of *T* two-dimensional
        input images of shape (m, n). The indexing order in **input_images** is
        assumed to be (time, latitude, longitude).

        *T* = 2 is the minimum required number of images.
        With *T* > 2, all the resulting sparse vectors are pooled together for
        the final interpolation on a regular grid.

        In case of ndarray_, invalid values (Nans or infs) are masked,
        otherwise the mask of the MaskedArray_ is used. Such mask defines a
        region where features are not detected for the tracking algorithm.

    lk_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the `Lucas-Kanade`_
        features tracking algorithm. See the documentation of
        :py:func:`pysteps.tracking.lucaskanade.track_features`.

    fd_method: {"shitomasi", "blob", "tstorm"}, optional
      Name of the feature detection routine. See feature detection methods in
      :py:mod:`pysteps.feature`.

    fd_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the features
        detection algorithm.
        See the documentation of :py:mod:`pysteps.feature`.

    interp_method: {"idwinterp2d", "rbfinterp2d"}, optional
      Name of the interpolation method to use. See interpolation methods in
      :py:mod:`pysteps.utils.interpolate`.

    interp_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the interpolation
        algorithm. See the documentation of :py:mod:`pysteps.utils.interpolate`.

    dense: bool, optional
        If True, return the three-dimensional array (2, m, n) containing
        the dense x- and y-components of the motion field.

        If False, return the sparse motion vectors as 2-D **xy** and **uv**
        arrays, where **xy** defines the vector positions, **uv** defines the
        x and y direction components of the vectors.

    nr_std_outlier: int, optional
        Maximum acceptable deviation from the mean in terms of number of
        standard deviations. Any sparse vector with a deviation larger than
        this threshold is flagged as outlier and excluded from the
        interpolation.
        See the documentation of
        :py:func:`pysteps.utils.cleansing.detect_outliers`.

    k_outlier: int or None, optional
        The number of nearest neighbors used to localize the outlier detection.
        If set to None, it employs all the data points (global detection).
        See the documentation of
        :py:func:`pysteps.utils.cleansing.detect_outliers`.

    size_opening: int, optional
        The size of the structuring element kernel in pixels. This is used to
        perform a binary morphological opening on the input fields in order to
        filter isolated echoes due to clutter. If set to zero, the filtering
        is not performed.
        See the documentation of
        :py:func:`pysteps.utils.images.morph_opening`.

    decl_scale: int, optional
        The scale declustering parameter in pixels used to reduce the number of
        redundant sparse vectors before the interpolation.
        Sparse vectors within this declustering scale are averaged together.
        If set to less than 2 pixels, the declustering is not performed.
        See the documentation of
        :py:func:`pysteps.utils.cleansing.decluster`.

    verbose: bool, optional
        If set to True, print some information about the program.

    Returns
    -------
    out: ndarray_ or tuple
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
    feature tracker description of the algorithm, Intel Corp., 5, 4, 2001

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

    feature_detection_method = feature.get_method(fd_method)
    interpolation_method = utils.get_method(interp_method)

    if fd_kwargs is None:
        fd_kwargs = dict()
    if fd_method == "tstorm":
        fd_kwargs.update({"output_feat": True})

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

        # Check if a MaskedArray is used. If not, mask the ndarray
        if not isinstance(prvs_img, MaskedArray):
            prvs_img = np.ma.masked_invalid(prvs_img)
        np.ma.set_fill_value(prvs_img, prvs_img.min())

        if not isinstance(next_img, MaskedArray):
            next_img = np.ma.masked_invalid(next_img)
        np.ma.set_fill_value(next_img, next_img.min())

        # remove small noise with a morphological operator (opening)
        if size_opening > 0:
            prvs_img = morph_opening(prvs_img, prvs_img.min(), size_opening)
            next_img = morph_opening(next_img, next_img.min(), size_opening)

        # features detection
        points = feature_detection_method(prvs_img, **fd_kwargs).astype(np.float32)

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
    uvgrid = interpolation_method(xy, uv, xgrid, ygrid, **interp_kwargs)

    if verbose:
        print("--- total time: %.2f seconds ---" % (time.time() - t0))

    return uvgrid
