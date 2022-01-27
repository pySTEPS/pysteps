"""
pysteps.utils.spectral
======================

Utility methods for processing and analyzing precipitation fields in the
Fourier domain.

.. autosummary::
    :toctree: ../generated/

    corrcoef
    mean
    rapsd
    remove_rain_norain_discontinuity
    std
"""

import numpy as np
from . import arrays


def corrcoef(X, Y, shape, use_full_fft=False):
    """
    Compute the correlation coefficient between two-dimensional arrays in
    the spectral domain.

    Parameters
    ----------
    X: array_like
        A complex array representing the Fourier transform of a two-dimensional
        array.
    Y: array_like
        A complex array representing the Fourier transform of a two-dimensional
        array.
    shape: tuple
        A two-element tuple specifying the shape of the original input arrays
        in the spatial domain.
    use_full_fft: bool
        If True, X and Y represent the full FFTs of the original arrays.
        Otherwise, they are assumed to contain only the symmetric part, i.e.
        in the format returned by numpy.fft.rfft2.

    Returns
    -------
    out: float
        The correlation coefficient. Gives the same result as
        numpy.corrcoef(X.flatten(), Y.flatten())[0, 1].
    """
    if len(X.shape) != 2:
        raise ValueError("X is not a two-dimensional array")

    if len(Y.shape) != 2:
        raise ValueError("Y is not a two-dimensional array")

    if X.shape != Y.shape:
        raise ValueError(
            "dimension mismatch between X and Y: "
            + "X.shape=%d,%d , " % (X.shape[0], X.shape[1])
            + "Y.shape=%d,%d" % (Y.shape[0], Y.shape[1])
        )

    n = np.real(np.sum(X * np.conj(Y))) - np.real(X[0, 0] * Y[0, 0])
    d1 = np.sum(np.abs(X) ** 2) - np.real(X[0, 0]) ** 2
    d2 = np.sum(np.abs(Y) ** 2) - np.real(Y[0, 0]) ** 2

    if not use_full_fft:
        if shape[1] % 2 == 1:
            n += np.real(np.sum(X[:, 1:] * np.conj(Y[:, 1:])))
            d1 += np.sum(np.abs(X[:, 1:]) ** 2)
            d2 += np.sum(np.abs(Y[:, 1:]) ** 2)
        else:
            n += np.real(np.sum(X[:, 1:-1] * np.conj(Y[:, 1:-1])))
            d1 += np.sum(np.abs(X[:, 1:-1]) ** 2)
            d2 += np.sum(np.abs(Y[:, 1:-1]) ** 2)

    return n / np.sqrt(d1 * d2)


def mean(X, shape):
    """
    Compute the mean value of a two-dimensional array in the spectral domain.

    Parameters
    ----------
    X: array_like
        A complex array representing the Fourier transform of a two-dimensional
        array.
    shape: tuple
        A two-element tuple specifying the shape of the original input array
        in the spatial domain.

    Returns
    -------
    out: float
        The mean value.
    """
    return np.real(X[0, 0]) / (shape[0] * shape[1])


def rapsd(
    field, fft_method=None, return_freq=False, d=1.0, normalize=False, **fft_kwargs
):
    """
    Compute radially averaged power spectral density (RAPSD) from the given
    2D input field.

    Parameters
    ----------
    field: array_like
        A 2d array of shape (m, n) containing the input field.
    fft_method: object
        A module or object implementing the same methods as numpy.fft and
        scipy.fftpack. If set to None, field is assumed to represent the
        shifted discrete Fourier transform of the input field, where the
        origin is at the center of the array
        (see numpy.fft.fftshift or scipy.fftpack.fftshift).
    return_freq: bool
        Whether to also return the Fourier frequencies.
    d: scalar
        Sample spacing (inverse of the sampling rate). Defaults to 1.
        Applicable if return_freq is 'True'.
    normalize: bool
        If True, normalize the power spectrum so that it sums to one.

    Returns
    -------
    out: ndarray
      One-dimensional array containing the RAPSD. The length of the array is
      int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
    freq: ndarray
      One-dimensional array containing the Fourier frequencies.

    References
    ----------
    :cite:`RC2011`
    """

    if len(field.shape) != 2:
        raise ValueError(
            f"{len(field.shape)} dimensions are found, but the number "
            "of dimensions should be 2"
        )

    if np.sum(np.isnan(field)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = field.shape

    yc, xc = arrays.compute_centred_coord_array(m, n)
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    l = max(field.shape[0], field.shape[1])

    if l % 2 == 1:
        r_range = np.arange(0, int(l / 2) + 1)
    else:
        r_range = np.arange(0, int(l / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(field, **fft_kwargs))
        psd = np.abs(psd) ** 2 / psd.size
    else:
        psd = field

    result = []
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        freq = np.fft.fftfreq(l, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result


def remove_rain_norain_discontinuity(R):
    """Function to remove the rain/no-rain discontinuity.
    It can be used before computing Fourier filters to reduce
    the artificial increase of power at high frequencies
    caused by the discontinuity.

    Parameters
    ----------
    R: array-like
        Array of any shape to be transformed.

    Returns
    -------
    R: array-like
        Array of any shape containing the transformed data.
    """
    R = R.copy()
    zerovalue = np.nanmin(R)
    threshold = np.nanmin(R[R > zerovalue])
    R[R > zerovalue] -= threshold - zerovalue
    R -= np.nanmin(R)

    return R


def std(X, shape, use_full_fft=False):
    """
    Compute the standard deviation of a two-dimensional array in the
    spectral domain.

    Parameters
    ----------
    X: array_like
        A complex array representing the Fourier transform of a two-dimensional
        array.
    shape: tuple
        A two-element tuple specifying the shape of the original input array
        in the spatial domain.
    use_full_fft: bool
        If True, X represents the full FFT of the original array. Otherwise, it
        is assumed to contain only the symmetric part, i.e. in the format
        returned by numpy.fft.rfft2.

    Returns
    -------
    out: float
        The standard deviation.
    """
    res = np.sum(np.abs(X) ** 2) - np.real(X[0, 0]) ** 2
    if not use_full_fft:
        if shape[1] % 2 == 1:
            res += np.sum(np.abs(X[:, 1:]) ** 2)
        else:
            res += np.sum(np.abs(X[:, 1:-1]) ** 2)

    return np.sqrt(res / (shape[0] * shape[1]) ** 2)
