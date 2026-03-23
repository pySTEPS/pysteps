# -*- coding: utf-8 -*-
"""
pysteps.motion.farneback
========================

The Farneback dense optical flow module.

This module implements the interface to the local `Farneback`_ routine
available in OpenCV_.

.. _OpenCV: https://opencv.org/

.. _`Farneback`:\
    https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af

.. autosummary::
    :toctree: ../generated/

    farneback
"""

import numpy as np
from numpy.ma.core import MaskedArray
import scipy.ndimage as sndi
import time

from pysteps.decorators import check_input_frames
from pysteps.exceptions import MissingOptionalDependency
from pysteps.utils.images import morph_opening

try:
    import cv2

    cv2_imported = True
except ImportError:
    cv2_imported = False


@check_input_frames(2)
def farneback(
    input_images,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.1,
    flags=0,
    size_opening=3,
    sigma=60.0,
    verbose=False,
):
    """Run the Farneback optical flow routine.

    .. _OpenCV: https://opencv.org/

    .. _`Farneback`:\
        https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af

    .. _MaskedArray:\
        https://docs.scipy.org/doc/numpy/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray

    .. _ndarray:\
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

    Parameters
    ----------
    input_images: ndarray_ or MaskedArray_
        Array of shape (T, m, n) containing a sequence of *T* two-dimensional
        input images of shape (m, n). The indexing order in **input_images** is
        assumed to be (time, latitude, longitude).

        *T* = 2 is the minimum required number of images.
        With *T* > 2, all the resulting motion vectors are averaged together.

        In case of ndarray_, invalid values (Nans or infs) are masked,
        otherwise the mask of the MaskedArray_ is used. Such mask defines a
        region where features are not detected for the tracking algorithm.

    pyr_scale : float, optional
        Parameter specifying the image scale (<1) to build pyramids for each
        image; pyr_scale=0.5 means a classical pyramid, where each next layer
        is twice smaller than the previous one.  This and parameter documented 
        below are taken directly from the original documentation. 
        (See https://docs.opencv.org).
        
    levels : int, optional
        number of pyramid layers including the initial image; levels=1 means
        that no extra layers are created and only the original images are used.
        
    winsize : int, optional
        Averaging window size; larger values increase the algorithm robustness
        to image noise and give more 
        Small windows (e.g. 10) lead to unrealistic motion.
        
    iterations : int, optional
        Number of iterations the algorithm does at each pyramid level.
        
    poly_n : int
        Size of the pixel neighborhood used to find polynomial expansion in
        each pixel; larger values mean that the image will be approximated with
        smoother surfaces, yielding more robust algorithm and more blurred
        motion field, typically poly_n = 5 or 7.
        
    poly_sigma : float
        Standard deviation of the Gaussian that is used to smooth derivatives
        used as a basis for the polynomial expansion; for poly_n=5, you can set
        poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
        
    flags : int, optional
        Operation flags that can be a combination of the following:

        OPTFLOW_USE_INITIAL_FLOW uses the input 'flow' as an initial flow
        approximation.

        OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize x winsize filter
        instead of a box filter of the same size for optical flow estimation;
        usually, this option gives z more accurate flow that with a box filter,
        at the cost of lower speed; normally, winsize for a Gaussian window
        should be set to larger value to achieve the same level of robustness.
        
    size_opening : int, optional
        Non-OpenCV parameter:
        The structuring element size for the filtering of isolated pixels [px].

    sigma : float, optional
        Non-OpenCV parameter:
        The smoothing bandwidth of the motion field. The motion field amplitude
        is adjusted by multiplying by the ratio of average magnitude before and
        after smoothing to avoid damping of the motion field.

    verbose: bool, optional
        If set to True, print some information about the program.

    Returns
    -------
    out : ndarray_, shape (2,m,n)
        Return the advection field having shape
        (2, m, n), where out[0, :, :] contains the x-components of the motion
        vectors and out[1, :, :] contains the y-components.
        The velocities are in units of pixels / timestep, where timestep is the
        time difference between the two input images.
        Return a zero motion field of shape (2, m, n) when no motion is
        detected.
        
    References
    ----------
    Farnebäck, G.: Two-frame motion estimation based on polynomial expansion, 
    In Image Analysis, pages 363–370. Springer, 2003.

    """

    if len(input_images.shape) != 3:
        raise ValueError(
            "input_images has %i dimensions, but a "
            "three-dimensional array is expected" % len(input_images.shape)
        )

    input_images = input_images.copy()

    if verbose:
        print("Computing the motion field with the Farneback method.")
        t0 = time.time()

    if not cv2_imported:
        raise MissingOptionalDependency(
            "opencv package is required for the Farneback method "
            "optical flow method but it is not installed"
        )

    nr_pairs = input_images.shape[0] - 1
    domain_size = (input_images.shape[1], input_images.shape[2])
    u_sum = np.zeros(domain_size)
    v_sum = np.zeros(domain_size)
    for n in range(nr_pairs):
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

        # remove small noise with a morphological operator (opening)
        if size_opening > 0:
            prvs_img = morph_opening(prvs_img, prvs_img.min(), size_opening)
            next_img = morph_opening(next_img, next_img.min(), size_opening)

        flow = cv2.calcOpticalFlowFarneback(
            prvs_img,
            next_img,
            None,
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
            flags,
        )

        fa, fb = np.dsplit(flow, 2)
        u_sum += fa.reshape(domain_size)
        v_sum += fb.reshape(domain_size)

    # Compute the average motion field
    u = u_sum / nr_pairs
    v = v_sum / nr_pairs

    # Smoothing
    if sigma > 0:
        uv2 = u * u + v * v  # squared magnitude of motion field
        us = sndi.gaussian_filter(u, sigma, mode="nearest")
        vs = sndi.gaussian_filter(v, sigma, mode="nearest")
        uvs2 = us * us + vs * vs  # squared magnitude of smoothed motion field

        mean_uv2 = np.nanmean(uv2)
        mean_uvs2 = np.nanmean(uvs2)
        if mean_uvs2 > 0:
            mult = np.sqrt(mean_uv2 / mean_uvs2)
        else:
            mult = 1.0
    else:
        mult = 1.0
    if verbose:
        print("mult factor of smoothed motion field=", mult)

    UV = np.stack([us * mult, vs * mult])

    if verbose:
        print("--- %s seconds ---" % (time.time() - t0))

    return UV
