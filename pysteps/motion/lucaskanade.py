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
import xarray as xr

from pysteps.decorators import check_input_frames

from pysteps import utils, feature
from pysteps.tracking.lucaskanade import track_features
from pysteps.utils.cleansing import decluster, remove_outliers
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
    """Run the Lucas-Kanade optical flow routine and interpolate the motion
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
        If set to None, the declustering is not performed.
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

    nr_fields = input_images.sizes["t"]
    detect_features = feature.get_method(fd_method)
    interpolate_vectors = utils.get_method(interp_method)

    if fd_kwargs is None:
        fd_kwargs = dict()
    if fd_method == "tstorm":
        fd_kwargs.update({"output_feat": True})

    if lk_kwargs is None:
        lk_kwargs = dict()

    if interp_kwargs is None:
        interp_kwargs = dict()

    xgrid_res = abs(float(input_images.x.isel(x=slice(2)).diff("x")))
    ygrid_res = abs(float(input_images.y.isel(y=slice(2)).diff("y")))
    grid_res = np.mean((xgrid_res, ygrid_res))

    # remove small-scale noise with a morphological operator (opening)
    input_images = morph_opening(input_images, input_images.min(), size_opening)

    sparse_vectors = []
    for n in range(nr_fields - 1):
        prvs_img = input_images.isel(t=n)
        next_img = input_images.isel(t=n + 1)
        features = detect_features(prvs_img, **fd_kwargs)
        if features.size == 0:
            continue
        sparse_vectors += [track_features(prvs_img, next_img, features, **lk_kwargs)]

    if len(sparse_vectors) > 0:
        sparse_vectors = xr.concat((sparse_vectors), "sample")
    else:
        if dense:
            return _make_zero_velocity(input_images.x, input_images.y)
        else:
            return sparse_vectors

    if verbose:
        print(f"... found {sparse_vectors.sizes['sample']} sparse vectors")

    sparse_vectors = remove_outliers(sparse_vectors, nr_std_outlier, k_outlier, verbose)
    if not dense:
        return sparse_vectors

    if not sparse_vectors.sizes["sample"]:
        return _make_zero_velocity(input_images.x, input_images.y)

    if decl_scale is not None:
        decl_scale = decl_scale * grid_res
    sparse_vectors = decluster(sparse_vectors, decl_scale, verbose)

    dense_vectors = interpolate_vectors(
        sparse_vectors, input_images.x, input_images.y, **interp_kwargs
    )

    if verbose:
        print("... total time: %.2f seconds" % (time.time() - t0))

    return dense_vectors


def _make_zero_velocity(xcoord, ycoord):
    """Return a zero motion field"""
    return xr.DataArray(
        np.zeros((2, len(ycoord), len(xcoord))),
        dims=("variable", "y", "x"),
        coords={
            "variable": ("variable", ["u", "v"]),
            "y": ycoord,
            "x": xcoord,
        },
        attrs={"units": "pixels / timestep"},
    )
