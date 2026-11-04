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
    rapsd_logarithmic
    fit_power_law_spectrum
    remove_rain_norain_discontinuity
    std
"""

import numpy as np
from . import arrays


def rapsd_logarithmic(field, min_stdev=0.1):
    """Compute a standardized field's radial spectrum in logarithmic bins.

    Parameters
    ----------
    field: array_like
        Finite, real two-dimensional field in (y, x) order. Dimensions must
        each contain at least four pixels. The input is not modified.
    min_stdev: float, optional
        Minimum population standard deviation required for estimation.

    Returns
    -------
    frequency, power: ndarray
        Bin-centre frequencies in cycles per pixel and mean power per Fourier
        mode, both in linear units. Empty arrays indicate insufficient variance
        or bin support.

    Notes
    -----
    Preserves the ``pysteps_param.power_spectrum_1D`` binning convention:
    min(shape)//2 logarithmic edges from 1/(min(shape)//2) to the historical
    diagonal frequency limit. This differs from :func:`rapsd` and deliberately
    omits the lowest frequencies. Empty bins are omitted. Power is the squared
    unscaled FFT of a zero-mean, unit-variance field, not annulus-integrated
    energy. Real-FFT multiplicities reproduce the full Fourier-plane mean.
    """
    if np.iscomplexobj(field):
        raise ValueError("field must contain real values")
    field = np.asarray(field, dtype=float)
    if field.ndim != 2 or min(field.shape) < 4:
        raise ValueError(
            "field must be two-dimensional with at least 4 pixels per axis"
        )
    if not np.isfinite(field).all():
        raise ValueError("field must contain only finite values")
    if not np.isfinite(min_stdev) or min_stdev < 0:
        raise ValueError("min_stdev must be finite and nonnegative")
    stdev = np.std(field)
    if stdev == 0 or stdev < min_stdev:
        return np.empty(0), np.empty(0)
    normalized = (field - np.mean(field)) / stdev
    power = np.abs(np.fft.rfft2(normalized)) ** 2
    rows, cols = field.shape
    fx, fy = np.fft.rfftfreq(cols), np.fft.fftfreq(rows)
    radius = np.hypot(fy[:, None], fx[None, :])
    weights = np.full(power.shape, 2.0)
    weights[:, 0] = 1.0
    if cols % 2 == 0:
        weights[:, -1] = 1.0
    n_bins = min(rows, cols) // 2
    maximum = np.hypot(
        np.max(np.abs(np.fft.fftfreq(rows)[: rows // 2])),
        np.max(np.abs(np.fft.fftfreq(cols)[: cols // 2])),
    )
    edges = np.logspace(np.log10(1 / n_bins), np.log10(maximum), n_bins)
    frequency, radial_power = [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (radius >= lower) & (radius < upper)
        if np.any(selected):
            frequency.append(0.5 * (lower + upper))
            radial_power.append(np.average(power[selected], weights=weights[selected]))
    return np.asarray(frequency), np.asarray(radial_power)


def fit_power_law_spectrum(
    frequency, power, scale_break=None, skip_bins=2, max_frequency=10 ** (-0.4)
):
    """Fit signed spectral slopes, optionally at a prescribed wavelength break.

    Parameters
    ----------
    frequency, power: array_like
        Equally sized one-dimensional arrays in linear units, with frequencies
        positive and strictly increasing. Power must be finite and nonnegative.
    scale_break: float, optional
        Fixed break wavelength in pixels when frequencies are cycles per pixel.
        For physical frequencies, use the corresponding length unit. None fits
        one slope. The fit is continuous at frequency 1/scale_break.
    skip_bins: int, optional
        Number of initial bins excluded from regression (default 2).
    max_frequency: float, optional
        Exclusive upper fit limit. Defaults preserve the source estimator's
        interval of 10*log10(frequency) < -4 for pixel frequencies.

    Returns
    -------
    result: dict
        ``beta_1``, ``beta_2``, ``c1``, ``c2``, and ``scale_break`` describe
        10*log10(power) = beta*10*log10(frequency) + c. Slopes are signed.
        ``status``, ``n_samples``, and ``rmse_db`` describe the fit. Unavailable
        coefficients are NaN; beta_2 and c2 are None for a single-slope model.

    Notes
    -----
    Zero-power bins cannot enter log-space regression. Insufficient samples,
    an unidentified break, or failed optimization are reported without inventing
    parameter estimates. The model is linear in its coefficients and is solved
    by unweighted least squares. No climatological clipping or calibration is
    applied.
    """
    frequency, power = np.asarray(frequency, float), np.asarray(power, float)
    if frequency.ndim != 1 or power.shape != frequency.shape:
        raise ValueError("frequency and power must be equally sized 1D arrays")
    if (
        not np.isfinite(frequency).all()
        or np.any(frequency <= 0)
        or np.any(np.diff(frequency) <= 0)
        or not np.isfinite(power).all()
        or np.any(power < 0)
    ):
        raise ValueError(
            "frequencies must increase and be positive; power must be nonnegative"
        )
    if not isinstance(skip_bins, (int, np.integer)) or skip_bins < 0:
        raise ValueError("skip_bins must be a nonnegative integer")
    if not np.isfinite(max_frequency) or max_frequency <= 0:
        raise ValueError("max_frequency must be positive and finite")
    if scale_break is not None and (not np.isfinite(scale_break) or scale_break <= 0):
        raise ValueError("scale_break must be positive and finite")
    selected = (np.arange(frequency.size) >= skip_bins) & (frequency < max_frequency)
    selected &= power > 0
    x, y = 10 * np.log10(frequency[selected]), 10 * np.log10(power[selected])
    result = dict(
        beta_1=np.nan,
        beta_2=np.nan if scale_break is not None else None,
        c1=np.nan,
        c2=np.nan if scale_break is not None else None,
        scale_break=scale_break,
        status="insufficient_samples",
        n_samples=int(x.size),
        rmse_db=np.nan,
    )
    if scale_break is None:

        def model(x, beta, intercept):
            return beta * x + intercept

        design = np.column_stack([x, np.ones(x.size)])
    else:
        breakpoint = 10 * np.log10(1 / scale_break)

        def model(x, beta_1, beta_2, c1):
            c2 = (beta_1 - beta_2) * breakpoint + c1
            return np.where(x <= breakpoint, beta_1 * x + c1, beta_2 * x + c2)

        design = np.column_stack(
            [np.minimum(x, breakpoint), np.maximum(x - breakpoint, 0), np.ones(x.size)]
        )
    if x.size < design.shape[1]:
        return result
    if np.linalg.matrix_rank(design) < design.shape[1]:
        result["status"] = "unidentified_model"
        return result
    try:
        fitted, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError as error:
        result.update(status="linear_solver_failed", message=str(error))
        return result
    if not np.isfinite(fitted).all():
        result["status"] = "linear_solver_failed"
        return result
    result.update(beta_1=float(fitted[0]), c1=float(fitted[-1]), status="ok")
    if scale_break is not None:
        result.update(
            beta_2=float(fitted[1]),
            c2=float((fitted[0] - fitted[1]) * breakpoint + fitted[2]),
        )
    result["rmse_db"] = float(np.sqrt(np.mean((model(x, *fitted) - y) ** 2)))
    return result


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
