# -*- coding: utf-8 -*-
"""
pysteps.downscaling.rainfarm
============================

Implementation of the RainFARM stochastic downscaling method as described in
:cite:`Rebora2006`.

RainFARM is a downscaling algorithm for rainfall fields developed by Rebora et
al. (2006). The method can represent the realistic small-scale variability of the
downscaled precipitation field by means of Gaussian random fields.


.. autosummary::
    :toctree: ../generated/

    downscale
"""

import warnings

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import zoom
from pysteps.utils import spectral


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


def _balanced_spatial_average(pr_high_res, tophat=None, orig_res=None, kernel="tophat"):

    """
    Parameters
    ----------
    pr_high_res : matrix
        matrix with the input field to smoothen, with dimensions ns*ns.
    tophat : array_like
        Array of weights
    orig_res : int
        original size of the precipitation field.
    kernel : string
        Choose the smoothing method

    Returns
    -------
    The smoothened field.

    References
    ----------
    :cite:`Terzago2018`

    """

    indmask = ~np.isfinite(pr_high_res)
    pr_high_res[indmask] = 0.0

    if kernel == "tophat":
        ones = np.ones_like(pr_high_res)
        return convolve(pr_high_res, tophat) / convolve(ones, tophat)

    elif kernel == "Gaussian":
        ns = np.shape(pr_high_res)[1]
        sdim = (ns / orig_res) / 2
        mask = np.zeros([ns, ns])
        for i in range(ns):
            for j in range(ns):

                kx = i
                ky = j
                if i > (ns / 2):
                    kx = i - ns
                if j > (ns / 2):
                    ky = j - ns
                r2 = kx * kx + ky * ky
                mask[i, j] = np.exp(-(r2 / (sdim * sdim)) / 2)

        fm = np.fft.fft2(mask)
        pf = np.real(np.fft.ifft2(fm * np.fft.fft2(pr_high_res))) / np.sum(mask)
        if np.sum(indmask) > 0:
            pr_high_res[~indmask] = 1
            pf = pf / (
                np.real(np.fft.ifft2(fm * np.fft.fft2(pr_high_res)))
                / np.sum(pr_high_res)
                / len(fm)
            )

        pf[indmask] = np.nan
        return pf

    else:
        raise RuntimeError(
            'the smooth value does not exist, choose from this list {None,"Gaussian","tophat"}'
        )


def _check_smooth_value(smooth):

    if smooth is None:
        return 0
    elif smooth == "Gaussian":
        return 1
    elif smooth == "tophat":
        return 2
    else:
        raise RuntimeError(
            'the smooth value does not exist, choose from this list {None,"Gaussian","tophat"}'
        )


def gaussianize(precip):
    """
    Gaussianize field using rank ordering

    Parameters
    ----------
    precip : matrix
        matrix containing the input field to be Gaussianized.

    Returns
    -------
        The Gaussianized field with the same dimensions as the input field.

    References
    ----------
    :cite:`DOnofrio2014`

    """
    m = np.shape(precip)[0]
    n = np.shape(precip)[1]
    nn = m * n
    ii = np.argsort(precip.reshape(nn))
    precip_gaussianize = np.zeros(nn)
    precip_gaussianize[ii] = sorted(np.random.normal(0, 1, nn))
    precip_gaussianize = precip_gaussianize.reshape(m, n)
    sd = np.std(precip_gaussianize)
    if sd == 0:
        sd = 1

    precip_gaussianize /= sd

    return precip_gaussianize


def downscale(
    precip,
    ds_factor,
    alpha=None,
    threshold=None,
    return_alpha=False,
    smooth="Gaussian",
    spectral_fusion=False,
):
    """
    Downscale a rainfall field by increasing its spatial resolution by
    a positive integer factor.

    Parameters
    ----------
    precip: array-like
        Array of shape (m, n) containing the input field.
        The input is expected to contain rain rate values.
        All values are required to be finite.
    alpha: float, optional
        Spectral slope. If None, the slope is estimated from
        the input array.
    ds_factor: positive int
        Downscaling factor, it specifies by how many times
        to increase the initial grid resolution.
    threshold: float, optional
        Set all values lower than the threshold to zero.
    return_alpha: bool, optional
        Whether to return the estimated spectral slope ``alpha``.
    smooth: list, optional
        Add the smoothing operatore from this list {None,"Gaussian","tophat"}
    Returns
    -------
    r: array-like
        Array of shape (m * ds_factor, n * ds_factor) containing
        the downscaled field.
    alpha: float
        Returned only when ``return_alpha=True``.

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
    type_smooth = _check_smooth_value(smooth)
    orig_res = np.shape(precip)[1]

    if spectral_fusion:
        precip_bis = gaussianize(precip)
    else:
        precip_bis = precip

    ki = np.fft.fftfreq(precip_bis.shape[0])
    kj = np.fft.fftfreq(precip_bis.shape[1])
    k_sqr = ki[:, None] ** 2 + kj[None, :] ** 2
    k = np.sqrt(k_sqr)

    ki_ds = np.fft.fftfreq(precip_bis.shape[0] * ds_factor, d=1 / ds_factor)
    kj_ds = np.fft.fftfreq(precip_bis.shape[1] * ds_factor, d=1 / ds_factor)
    k_ds_sqr = ki_ds[:, None] ** 2 + kj_ds[None, :] ** 2
    k_ds = np.sqrt(k_ds_sqr)

    if alpha is None:
        fp = np.fft.fft2(precip_bis)
        fp_abs = abs(fp)
        log_power_spectrum = np.log(fp_abs**2)
        valid = (k != 0) & np.isfinite(log_power_spectrum)
        alpha = _log_slope(np.log(k[valid]), log_power_spectrum[valid])

    fg = np.exp(complex(0, 1) * 2 * np.pi * np.random.rand(*k_ds.shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fg *= np.sqrt(k_ds_sqr ** (-alpha / 2))
    fg[0, 0] = 0

    if spectral_fusion:
        fp = np.fft.fft2(precip_bis)
        kmax = int(precip_bis.shape[0] / 2)
        nx = fg.shape[0]
        Pk0 = spectral.rapsd(precip_bis)[kmax - 1]
        gk0 = spectral.rapsd(np.fft.ifft2(fg).real / np.fft.ifft2(fg).real.std())[
            kmax - 1
        ]
        fg *= Pk0 / gk0
        fg[0:kmax, 0:kmax] = fp[0:kmax, 0:kmax]
        fg[nx - kmax :, 0:kmax] = fp[-kmax:, 0:kmax]
        fg[0:kmax, nx - kmax :] = fp[0:kmax, -kmax:]
        fg[nx - kmax :, nx - kmax :] = fp[-kmax:, -kmax:]

    g = np.fft.ifft2(fg).real
    g /= g.std()
    r = np.exp(g)

    P_u = np.repeat(np.repeat(precip, ds_factor, axis=0), ds_factor, axis=1)
    rad = int(round(ds_factor / np.sqrt(np.pi)))
    (mx, my) = np.mgrid[-rad : rad + 0.01, -rad : rad + 0.01]
    tophat = ((mx**2 + my**2) <= rad**2).astype(float)
    tophat /= tophat.sum()

    if type_smooth == 0:

        P_agg = precip
        r_agg = zoom(r, 1 / ds_factor, order=1)
        factor = np.repeat(
            np.repeat(P_agg / r_agg, ds_factor, axis=0), ds_factor, axis=1
        )
        r *= factor

    elif type_smooth == 1:

        P_agg = _balanced_spatial_average(P_u, orig_res=orig_res, kernel="Gaussian")
        r_agg = _balanced_spatial_average(r, orig_res=orig_res, kernel="Gaussian")
        r *= P_agg / r_agg

    elif type_smooth == 2:

        P_agg = _balanced_spatial_average(P_u, tophat=tophat, kernel="tophat")
        r_agg = _balanced_spatial_average(r, tophat=tophat, kernel="tophat")
        r *= P_agg / r_agg

    if threshold is not None:
        r[r < threshold] = 0

    if return_alpha:
        return r, alpha

    return r
