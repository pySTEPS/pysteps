# -*- coding: utf-8 -*-
"""
pysteps.downscaling.rainfarm
============================

Implementation of the RainFARM stochastic downscaling method as described in
:cite:`Rebora2006`.

.. autosummary::
    :toctree: ../generated/

    downscale
"""

import warnings

import numpy as np
from scipy.ndimage import convolve


def _log_slope(log_k, log_power_spectrum):
    lk_min = log_k.min()
    lk_max = log_k.max()
    lk_range = lk_max - lk_min
    lk_min += (1 / 6) * lk_range
    lk_max -= (1 / 6) * lk_range

    selected = (lk_min <= log_k) & (log_k <= lk_max)
    lk_sel = log_k[selected]
    ps_sel = log_power_spectrum[selected]
    alpha = np.polyfit(lk_sel, ps_sel, 1)[0]
    alpha = -alpha

    return alpha


def _balanced_spatial_average(x, k):
    ones = np.ones_like(x)
    return convolve(x, k) / convolve(ones, k)


def downscale(precip, alpha=None, ds_factor=16, threshold=None, return_alpha=False):
    """
    Downscale a rainfall field by a given factor.

    Parameters
    ----------

    precip: array_like
        Array of shape (m,n) containing the input field.
        The input is expected to contain rain rate values.

    alpha: float, optional
        Spectral slope. If none, the slope is estimated from
        the input array.

    ds_factor: int, optional
        Downscaling factor.

    threshold: float, optional
        Set all values lower than the threshold to zero.

    return_alpha: bool, optional
        Whether to return the estimated spectral slope `alpha`.


    Returns
    -------
    r: array_like
        Array of shape (m*ds_factor,n*ds_factor) containing
        the downscaled field.
    alpha: float
        Returned only when `return_alpha=True`.

    Notes
    -----
    Currently, the pysteps implementation of RainFARM only covers spatial downscaling.
    That is, it can improve the spatial resolution of a rainfall field. However, unlike
    the original algorithm from Rebora et al. (2006), it cannot downscale the temporal
    dimension.

    References
    ----------
    :cite:`Rebora2006`

    """

    ki = np.fft.fftfreq(precip.shape[0])
    kj = np.fft.fftfreq(precip.shape[1])
    k_sqr = ki[:, None] ** 2 + kj[None, :] ** 2
    k = np.sqrt(k_sqr)

    ki_ds = np.fft.fftfreq(precip.shape[0] * ds_factor, d=1 / ds_factor)
    kj_ds = np.fft.fftfreq(precip.shape[1] * ds_factor, d=1 / ds_factor)
    k_ds_sqr = ki_ds[:, None] ** 2 + kj_ds[None, :] ** 2
    k_ds = np.sqrt(k_ds_sqr)

    if alpha is None:
        fp = np.fft.fft2(precip)
        fp_abs = abs(fp)
        log_power_spectrum = np.log(fp_abs ** 2)
        valid = (k != 0) & np.isfinite(log_power_spectrum)
        alpha = _log_slope(np.log(k[valid]), log_power_spectrum[valid])

    fg = np.exp(complex(0, 1) * 2 * np.pi * np.random.rand(*k_ds.shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fg *= np.sqrt(k_ds_sqr ** (-alpha / 2))
    fg[0, 0] = 0
    g = np.fft.ifft2(fg).real
    g /= g.std()
    r = np.exp(g)

    P_u = np.repeat(np.repeat(precip, ds_factor, axis=0), ds_factor, axis=1)
    rad = int(round(ds_factor / np.sqrt(np.pi)))
    (mx, my) = np.mgrid[-rad : rad + 0.01, -rad : rad + 0.01]
    tophat = ((mx ** 2 + my ** 2) <= rad ** 2).astype(float)
    tophat /= tophat.sum()

    P_agg = _balanced_spatial_average(P_u, tophat)
    r_agg = _balanced_spatial_average(r, tophat)
    r *= P_agg / r_agg

    if threshold is not None:
        r[r < threshold] = 0

    if return_alpha:
        return r, alpha

    return r
