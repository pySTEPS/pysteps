# -*- coding: utf-8 -*-
"""
pysteps.motion.proesmans
========================

Implementation of the anisotropic diffusion method of Proesmans et al. (1994).

.. autosummary::
    :toctree: ../generated/

    proesmans
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from pysteps.decorators import check_input_frames
from pysteps.motion._proesmans import _compute_advection_field


@check_input_frames(2, 2)
def proesmans(
    input_images,
    lam=50.0,
    num_iter=100,
    num_levels=6,
    filter_std=0.0,
    verbose=True,
    full_output=False,
):
    """Implementation of the anisotropic diffusion method of Proesmans et al.
    (1994).

    Parameters
    ----------
    input_images: array_like
        Array of shape (2, m, n) containing the first and second input image.
    lam: float
        Multiplier of the smoothness term. Smaller values give a smoother motion
        field.
    num_iter: float
        The number of iterations to use.
    num_levels: int
        The number of image pyramid levels to use.
    filter_std: float
        Standard deviation of an optional Gaussian filter that is applied before
        computing the optical flow.
    verbose: bool, optional
        Verbosity enabled if True (default).
    full_output: bool, optional
        If True, the output is a two-element tuple containing the
        forward-backward advection and consistency fields. The first element
        is shape (2, 2, m, n), where the index along the first dimension refers
        to the forward and backward advection fields. The second element is an
        array of shape (2, m, n), where the index along the first dimension
        refers to the forward and backward consistency fields.
        Default: False.

    Returns
    -------
    out: ndarray
        If full_output=False, the advection field having shape (2, m, n), where
        out[0, :, :] contains the x-components of the motion vectors and
        out[1, :, :] contains the y-components. The velocities are in units of
        pixels / timestep, where timestep is the time difference between the
        two input images.

    References
    ----------
    :cite:`PGPO1994`

    """
    del verbose  # Not used

    im1 = input_images[-2, :, :].copy()
    im2 = input_images[-1, :, :].copy()

    im = np.stack([im1, im2])
    im_min = np.min(im)
    im_max = np.max(im)
    if im_max - im_min > 1e-8:
        im = (im - im_min) / (im_max - im_min) * 255.0

    if filter_std > 0.0:
        im[0, :, :] = gaussian_filter(im[0, :, :], filter_std)
        im[1, :, :] = gaussian_filter(im[1, :, :], filter_std)

    advfield, quality = _compute_advection_field(im, lam, num_iter, num_levels)

    if not full_output:
        return advfield[0]
    else:
        return advfield, quality
