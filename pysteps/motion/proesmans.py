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
import xarray as xr
from scipy.ndimage import gaussian_filter

from pysteps.decorators import check_input_frames
from pysteps.motion._proesmans import _compute_advection_field


@check_input_frames(2, 2)
def proesmans(
    dataset: xr.Dataset,
    lam=50.0,
    num_iter=100,
    num_levels=6,
    filter_std=0.0,
    verbose=True,
    full_output=False,
):
    """
    Implementation of the anisotropic diffusion method of Proesmans et al.
    (1994).

    Parameters
    ----------
    dataset: xarray.Dataset
        Input dataset as described in the documentation of
        :py:mod:`pysteps.io.importers`. It has to contain a precipitation data variable.
        The dataset has to have a time dimension. The size of this dimension
        has to be 2.
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
        If True, both the forward and backwards advection fields are returned
        and the consistency fields are returned as well in the ``velocity_quality``
        data variable.

    Returns
    -------
    out: ndarray
        The input dataset with the advection field added in the ``velocity_x``
        and ``velocity_y`` data variables.

        If full_output=True, a ``velocity_direction`` dimension
        is added to the dataset, so that the velocity data can be returned containing
        the forward and backwards advection fields. Also the ``velocity_quality`` data
        coordinate is present containing the forward and backward consistency fields.

    References
    ----------
    :cite:`PGPO1994`

    """
    del verbose  # Not used

    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    input_images = dataset[precip_var].values
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
        dataset["velocity_x"] = (["y", "x"], advfield[0, 0])
        dataset["velocity_y"] = (["y", "x"], advfield[0, 1])
    else:
        dataset["velocity_x"] = (["direction", "y", "x"], advfield[:, 0])
        dataset["velocity_y"] = (["direction", "y", "x"], advfield[:, 1])
        dataset["velocity_quality"] = (["direction", "y", "x"], quality)

    return dataset
