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
from scipy.ndimage import zoom
from scipy.signal import convolve


def _gaussianize(precip):
    """
    Gaussianize field using rank ordering as in :cite:`DOnofrio2014`.
    """
    m, n = np.shape(precip)
    nn = m * n
    ii = np.argsort(precip.reshape(nn))
    precip_gaussianize = np.zeros(nn)
    precip_gaussianize[ii] = sorted(np.random.normal(0, 1, nn))
    precip_gaussianize = precip_gaussianize.reshape(m, n)
    sd = np.std(precip_gaussianize)
    if sd == 0:
        sd = 1
    return precip_gaussianize / sd


def _compute_freq_array(array, ds_factor=1):
    freq_i = np.fft.fftfreq(array.shape[0] * ds_factor, d=1 / ds_factor)
    freq_j = np.fft.fftfreq(array.shape[1] * ds_factor, d=1 / ds_factor)
    freq_sqr = freq_i[:, None] ** 2 + freq_j[None, :] ** 2
    return np.sqrt(freq_sqr)


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


def _estimate_alpha(array, k):
    fp = np.fft.fft2(array)
    fp_abs = abs(fp)
    log_power_spectrum = np.log(fp_abs**2)
    valid = (k != 0) & np.isfinite(log_power_spectrum)
    alpha = _log_slope(np.log(k[valid]), log_power_spectrum[valid])
    return alpha


def _apply_spectral_fusion(array_low, array_high):

    kmax = array_low.shape[0] // 2

    (nax, nay) = np.shape(array_low)
    (nx, ny) = np.shape(array_high)

    DFTr = np.fft.fft2(array_low)
    DFTf = np.fft.fft2(array_high)

    DFTr = np.fft.fftshift(DFTr)
    DFTf = np.fft.fftshift(DFTf)

    DFTfm = np.zeros([nx, nx])
    DFTr2 = np.zeros([nax + 1, nax + 1])

    DFTr2[0:nax, 0:nax] = DFTr[:, :]
    DFTr2[-1, 0:nax] = np.conj(DFTr[0, :])
    DFTr2[0:nax, -1] = np.conj(DFTr[:, 0])

    kmax2 = kmax**2

    ddx = 2 * np.pi / nax / 2 - 2 * np.pi / nx / 2

    for i in range(nx):
        for j in range(nx):
            kx = -(nx // 2) + i - 1
            ky = -(nx // 2) + j - 1
            k2 = kx**2 + ky**2
            ir = (nax // 2) + 1 + kx
            jr = (nax // 2) + 1 + ky
            if k2 <= kmax2:
                DFTfm[i, j] = DFTr2[ir - 1, jr - 1] * np.exp(
                    complex(0, ddx * kx - ddx * ky)
                )
            else:
                DFTfm[i, j] = DFTf[i, j]

    DFTfm = np.fft.ifftshift(DFTfm)
    fm = np.fft.ifft2(DFTfm).real
    fm /= fm.std()
    fm = np.exp(fm)
    return fm


def _compute_kernel_radius(ds_factor):
    return int(round(ds_factor / np.sqrt(np.pi)))


def _make_tophat_kernel(ds_factor):
    """Compute 2d uniform (tophat) kernel"""
    radius = _compute_kernel_radius(ds_factor)
    (mx, my) = np.mgrid[-radius : radius + 0.01, -radius : radius + 0.01]
    tophat = ((mx**2 + my**2) <= radius**2).astype(float)
    return tophat / tophat.sum()


def _make_gaussian_kernel(ds_factor):
    """
    Compute 2d gaussian kernel
    ref: https://github.com/scipy/scipy/blob/de80faf9d3480b9dbb9b888568b64499e0e70c19/scipy/ndimage/_filters.py#L179
    the smoothing sigma has width half a large pixel
    """
    radius = _compute_kernel_radius(ds_factor)
    sigma = ds_factor / 2
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    kern1d = np.exp(-0.5 / sigma2 * x**2)
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def _balanced_spatial_average(array, kernel):
    array = array.copy()
    mask_valid = np.isfinite(array)
    array[~mask_valid] = 0.0
    array_conv = convolve(array, kernel, mode="same")
    array_conv /= convolve(mask_valid, kernel, mode="same")
    array_conv[~mask_valid] = np.nan
    return array_conv


_make_kernel = dict()
_make_kernel["gaussian"] = _make_gaussian_kernel
_make_kernel["tophat"] = _make_tophat_kernel
_make_kernel["uniform"] = _make_tophat_kernel


def downscale(
    precip,
    ds_factor,
    alpha=None,
    threshold=None,
    return_alpha=False,
    kernel_type="gaussian",
    spectral_fusion=False,
):
    """
    Downscale a rainfall field by increasing its spatial resolution by a positive
    integer factor.

    Parameters
    ----------
    precip: array_like
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
    kernel_type: {"gaussian", "uniform", "tophat"}
        The name of the smoothing operator.

    Returns
    -------
    precip_highres: ndarray
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

    if spectral_fusion:
        precip_transformed = _gaussianize(precip)
    else:
        precip_transformed = precip

    freq_array = _compute_freq_array(precip_transformed)
    freq_array_highres = _compute_freq_array(precip_transformed, ds_factor)

    if alpha is None:
        alpha = _estimate_alpha(precip_transformed, freq_array)

    white_noise_field = np.random.rand(*freq_array_highres.shape)
    white_noise_field_complex = np.exp(complex(0, 1) * 2 * np.pi * white_noise_field)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise_field_complex = white_noise_field_complex * np.sqrt(
            freq_array_highres**-alpha
        )
    noise_field_complex[0, 0] = 0
    noise_field = np.fft.ifft2(noise_field_complex).real
    noise_field /= noise_field.std()
    noise_field = np.exp(noise_field)

    if spectral_fusion:
        noise_field = _apply_spectral_fusion(precip_transformed, noise_field)

    try:
        kernel = _make_kernel[kernel_type](ds_factor)
    except KeyError:
        raise ValueError(
            f"kernel type '{kernel_type}' is invalid, "
            f"available kernels: {list(_make_kernel)}"
        )

    precip_zoomed = zoom(precip, ds_factor, order=1)
    precip_smoothed = _balanced_spatial_average(precip_zoomed, kernel)
    noise_smoothed = _balanced_spatial_average(noise_field, kernel)
    precip_highres = noise_field * precip_smoothed / noise_smoothed

    if threshold is not None:
        precip_highres[precip_highres < threshold] = 0

    if return_alpha:
        return precip_highres, alpha

    return precip_highres
