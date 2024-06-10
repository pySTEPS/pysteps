# -*- coding: utf-8 -*-
"""
pysteps.downscaling.rainfarm
============================

Implementation of the RainFARM stochastic downscaling method as described in
:cite:`Rebora2006` and :cite:`DOnofrio2014`.

RainFARM is a downscaling algorithm for rainfall fields developed by Rebora et
al. (2006). The method can represent the realistic small-scale variability of the
downscaled precipitation field by means of Gaussian random fields.


.. autosummary::
    :toctree: ../generated/

    downscale
"""

import warnings

import numpy as np
from scipy.signal import convolve
from pysteps.utils.spectral import rapsd
from pysteps.utils.dimension import aggregate_fields


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
    """
    Compute the frequency array following a given downscaling factor.
    """
    freq_i = np.fft.fftfreq(array.shape[0] * ds_factor, d=1 / ds_factor)
    freq_j = np.fft.fftfreq(array.shape[1] * ds_factor, d=1 / ds_factor)
    freq_sqr = freq_i[:, None] ** 2 + freq_j[None, :] ** 2
    return np.sqrt(freq_sqr)


def _log_slope(log_k, log_power_spectrum):
    """
    Calculate the log-slope of the power spectrum given an array of logarithmic wavenumbers
    and an array of logarithmic power spectrum values.
    """
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
    """
    Estimate the alpha parameter using the power spectrum of the input array.
    """
    fp = np.fft.fft2(array)
    fp_abs = abs(fp)
    log_power_spectrum = np.log(fp_abs**2)
    valid = (k != 0) & np.isfinite(log_power_spectrum)
    alpha = _log_slope(np.log(k[valid]), log_power_spectrum[valid])
    return alpha


def _compute_noise_field(freq_array_highres, alpha):
    """
    Compute a field of correlated noise field using the given frequency array and alpha
    value.
    """
    white_noise_field = np.random.rand(*freq_array_highres.shape)
    white_noise_field_complex = np.exp(complex(0, 1) * 2 * np.pi * white_noise_field)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise_field_complex = white_noise_field_complex * np.sqrt(
            freq_array_highres**-alpha
        )
    noise_field_complex[0, 0] = 0
    return np.fft.ifft2(noise_field_complex).real


def _apply_spectral_fusion(
    array_low, array_high, freq_array_low, freq_array_high, ds_factor
):
    """
    Apply spectral fusion to merge two arrays in the frequency domain.
    """

    # Validate inputs
    if array_low.shape != freq_array_low.shape:
        raise ValueError("Shape of array_low must match shape of freq_array_low.")
    if array_high.shape != freq_array_high.shape:
        raise ValueError("Shape of array_high must match shape of freq_array_high.")

    nax, _ = np.shape(array_low)
    nx, _ = np.shape(array_high)
    k0 = nax // 2

    # Calculate power spectral density at specific frequency
    def compute_psd(array, fft_size):
        return rapsd(array, fft_method=np.fft)[k0 - 1] * fft_size**2

    psd_low = compute_psd(array_low, nax)
    psd_high = compute_psd(array_high, nx)

    # Normalize high-resolution array
    normalization_factor = np.sqrt(psd_low / psd_high)
    array_high *= normalization_factor

    # Perform FFT on both arrays
    fft_low = np.fft.fft2(array_low)
    fft_high = np.fft.fft2(array_high)

    # Initialize the merged FFT array with low-resolution data
    fft_merged = np.zeros_like(fft_high, dtype=np.complex128)
    fft_merged[0:k0, 0:k0] = fft_low[0:k0, 0:k0]
    fft_merged[nx - k0 : nx, 0:k0] = fft_low[k0 : 2 * k0, 0:k0]
    fft_merged[0:k0, nx - k0 : nx] = fft_low[0:k0, k0 : 2 * k0]
    fft_merged[nx - k0 : nx, nx - k0 : nx] = fft_low[k0 : 2 * k0, k0 : 2 * k0]

    fft_merged[k0, 0] = np.conj(fft_merged[nx - k0, 0])
    fft_merged[0, k0] = np.conj(fft_merged[0, nx - k0])

    # Compute frequency arrays
    freq_i = np.fft.fftfreq(nx, d=1 / ds_factor)
    freq_i = np.tile(freq_i, (nx, 1))
    freq_j = freq_i.T

    # Compute frequency domain adjustment
    ddx = np.pi * (1 / nax - 1 / nx) / np.abs(freq_i[0, 1] - freq_i[0, 0])
    freq_squared_high = freq_array_high**2
    freq_squared_low_center = freq_array_low[k0, k0] ** 2

    # Fuse in the frequency domain
    mask_high = freq_squared_high > freq_squared_low_center
    mask_low = ~mask_high
    fft_merged = fft_high * mask_high + fft_merged * mask_low * np.exp(
        -1j * ddx * freq_i - 1j * ddx * freq_j
    )

    # Inverse FFT to obtain the merged array in the spatial domain
    merged = np.real(np.fft.ifftn(fft_merged)) / fft_merged.size

    return merged


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
    """
    Compute the balanced spatial average of an array using a given kernel while handling
    missing or invalid values.
    """
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
    kernel_type=None,
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
    kernel_type: {None, "gaussian", "uniform", "tophat"}
        The name of the smoothing operator. If None no smoothing is applied.
    spectral_fusion: bool, optional
        Whether to apply spectral merging as in :cite:`DOnofrio2014`.

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
    dimension. It implements spectral merging from D'Onofrio et al. (2014).

    References
    ----------
    :cite:`Rebora2006`
    :cite:`DOnofrio2014`

    """

    # Validate inputs
    if not np.isfinite(precip).all():
        raise ValueError("All values in 'precip' must be finite.")
    if not isinstance(ds_factor, int) or ds_factor <= 0:
        raise ValueError("'ds_factor' must be a positive integer.")

    # Preprocess the input field if spectral fusion is enabled
    precip_transformed = _gaussianize(precip) if spectral_fusion else precip

    # Compute frequency arrays for the original and high-resolution fields
    freq_array = _compute_freq_array(precip_transformed)
    freq_array_highres = _compute_freq_array(precip_transformed, ds_factor)

    # Estimate spectral slope alpha if not provided
    if alpha is None:
        alpha = _estimate_alpha(precip_transformed, freq_array)

    # Generate noise field
    noise_field = _compute_noise_field(freq_array_highres, alpha)

    # Apply spectral fusion if enabled
    if spectral_fusion:
        noise_field /= noise_field.shape[0] ** 2
        noise_field = np.exp(noise_field)
        noise_field = _apply_spectral_fusion(
            precip_transformed, noise_field, freq_array, freq_array_highres, ds_factor
        )

    # Normalize and exponentiate the noise field
    noise_field /= noise_field.std()
    noise_field = np.exp(noise_field)

    # Aggregate the noise field to low resolution
    noise_lowres = aggregate_fields(noise_field, ds_factor, axis=(0, 1))

    # Expand input and noise fields to high resolution
    precip_expanded = np.kron(precip, np.ones((ds_factor, ds_factor)))
    noise_lowres_expanded = np.kron(noise_lowres, np.ones((ds_factor, ds_factor)))

    # Apply smoothing if a kernel type is provided
    if kernel_type:
        if kernel_type not in _make_kernel:
            raise ValueError(
                f"kernel type '{kernel_type}' is invalid, available kernels: {list(_make_kernel)}"
            )
        kernel = _make_kernel[kernel_type](ds_factor)
        precip_expanded = _balanced_spatial_average(precip_expanded, kernel)
        noise_lowres_expanded = _balanced_spatial_average(noise_lowres_expanded, kernel)

    # Normalize the high-res precipitation field by the low-res noise field
    norm_k0 = precip_expanded / noise_lowres_expanded
    precip_highres = noise_field * norm_k0

    # Apply thresholding if specified
    if threshold is not None:
        precip_highres[precip_highres < threshold] = 0

    # Return the downscaled field and optionally the spectral slope alpha
    if return_alpha:
        return precip_highres, alpha

    return precip_highres
