# -*- coding: utf-8 -*-
"""
pysteps.timeseries.correlation
==============================

Methods for computing spatial and temporal correlation of time series of
two-dimensional fields.

.. autosummary::
    :toctree: ../generated/

    temporal_autocorrelation
    temporal_autocorrelation_multivariate
"""

import numpy as np
from scipy import ndimage
from pysteps.utils import spectral


def temporal_autocorrelation(
    x,
    d=0,
    domain="spatial",
    x_shape=None,
    mask=None,
    use_full_fft=False,
    window="gaussian",
    window_radius=np.inf,
):
    """
    Compute lag-l temporal autocorrelation coefficients
    :math:`\gamma_l=\mbox{corr}(x(t),x(t-l))`, :math:`l=1,2,\dots,n-1`,
    from a time series :math:`x_1,x_2,\dots,x_n`. If a multivariate time series
    is given, each element of :math:`x_i` is treated as one sample from the
    process generating the time series. Use
    :py:func:`temporal_autocorrelation_multivariate` if cross-correlations
    between different elements of the time series are desired.

    Parameters
    ----------
    x: array_like
        Array of shape (n, ...), where each row contains one sample from the
        time series :math:`x_i`. The inputs are assumed to be in increasing
        order with respect to time, and the time step is assumed to be regular.
        All inputs are required to have finite values. The remaining dimensions
        after the first one are flattened before computing the correlation
        coefficients.
    d: {0,1}
        The order of differencing. If d=1, the input time series is differenced
        before computing the correlation coefficients. In this case, a time
        series of length n+1 is needed for computing the n-1 coefficients.
    domain: {"spatial", "spectral"}
        The domain of the time series x. If domain is "spectral", the elements
        of x are assumed to represent the FFTs of the original elements.
    x_shape: tuple
        The shape of the original arrays in the spatial domain before applying
        the FFT. Required if domain is "spectral".
    mask: array_like
        Optional mask to use for computing the correlation coefficients. Input
        elements with mask==False are excluded from the computations. The shape
        of the mask is expected to be x.shape[1:]. Applicable if domain is
        "spatial".
    use_full_fft: bool
        If True, x represents the full FFTs of the original arrays. Otherwise,
        the elements of x are assumed to contain only the symmetric part, i.e.
        in the format returned by numpy.fft.rfft2. Applicable if domain is
        'spectral'. Defaults to False.
    window: {"gaussian", "uniform"}
        The weight function to use for the moving window. Applicable if
        window_radius < np.inf. Defaults to 'gaussian'.
    window_radius: float
        If window_radius < np.inf, the correlation coefficients are computed in
        a moving window. Defaults to np.inf (i.e. the coefficients are computed
        over the whole domain). If window is 'gaussian', window_radius is the
        standard deviation of the Gaussian filter. If window is 'uniform', the
        size of the window is 2*window_radius+1.

    Returns
    -------
    out: list
        List of length n-1 containing the temporal autocorrelation coefficients
        :math:`\gamma_i` for time lags :math:`l=1,2,...,n-1`. If
        window_radius<np.inf, the elements of the list are arrays of shape
        x.shape[1:]. In this case, nan values are assigned, when the sample size
        for computing the correlation coefficients is too small.

    Notes
    -----
    Computation of correlation coefficients in the spectral domain is currently
    implemented only for two-dimensional fields.

    """
    if len(x.shape) < 2:
        raise ValueError("the dimension of x must be >= 2")
    if len(x.shape) != 3 and domain == "spectral":
        raise NotImplementedError(
            "len(x.shape[1:]) = %d, but with domain == 'spectral', this function has only been implemented for two-dimensional fields"
            % len(x.shape[1:])
        )
    if mask is not None and mask.shape != x.shape[1:]:
        raise ValueError(
            "dimension mismatch between x and mask: x.shape[1:]=%s, mask.shape=%s"
            % (str(x.shape[1:]), str(mask.shape))
        )
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    if d == 1:
        x = np.diff(x, axis=0)

    if domain == "spatial" and mask is None:
        mask = np.ones(x.shape[1:], dtype=bool)

    gamma = []
    for k in range(x.shape[0] - 1):
        if domain == "spatial":
            if window_radius == np.inf:
                cc = np.corrcoef(x[-1, :][mask], x[-(k + 2), :][mask])[0, 1]
            else:
                cc = _moving_window_corrcoef(
                    x[-1, :], x[-(k + 2), :], window_radius, mask=mask
                )
        else:
            cc = spectral.corrcoef(
                x[-1, :, :], x[-(k + 2), :], x_shape, use_full_fft=use_full_fft
            )
        gamma.append(cc)

    return gamma


def temporal_autocorrelation_multivariate(
    x, d=0, mask=None, window="gaussian", window_radius=np.inf
):
    """
    For a :math:`q`-variate time series
    :math:`\mathbf{x}_1,\mathbf{x}_2,\dots,\mathbf{x}_n`, compute the lag-l
    correlation matrices :math:`\mathbf{\Gamma}_l`, where
    :math:`\Gamma_{l,i,j}=\gamma_{l,i,j}` and
    :math:`\gamma_{l,i,j}=\mbox{corr}(x_i(t),x_j(t-l))` for
    :math:`i,j=1,2,\dots,q` and :math:`l=0,1,\dots,n-1`.

    Parameters
    ----------
    x: array_like
        Array of shape (n, q, ...) containing the time series :math:`\mathbf{x}_i`.
        The inputs are assumed to be in increasing order with respect to time,
        and the time step is assumed to be regular. All inputs are required to
        have finite values. The remaining dimensions after the second one are
        flattened before computing the correlation coefficients.
    d: {0,1}
        The order of differencing. If d=1, the input time series is differenced
        before computing the correlation coefficients. In this case, a time
        series of length n+1 is needed for computing the n-1 coefficients.
    mask: array_like
        Optional mask to use for computing the correlation coefficients. Input
        elements with mask==False are excluded from the computations. The shape
        of the mask is expected to be x.shape[2:].
    window: {"gaussian", "uniform"}
        The weight function to use for the moving window. Applicable if
        window_radius < np.inf. Defaults to 'gaussian'.
    window_radius: float
        If window_radius < np.inf, the correlation coefficients are computed in
        a moving window. Defaults to np.inf (i.e. the correlations are computed
        over the whole domain). If window is 'gaussian', window_radius is the
        standard deviation of the Gaussian filter. If window is 'uniform', the
        size of the window is 2*window_radius+1.

    Returns
    -------
    out: list
        List of correlation matrices :math:`\Gamma_0,\Gamma_1,\dots,\Gamma_{n-1}`
        of shape (q,q). If window_radius<np.inf, the elements of the list are
        arrays of shape (x.shape[2:],q,q). In this case, nan values are assigned,
        when the sample size for computing the correlation coefficients is too
        small.

    References
    ----------
    :cite:`CP2002`

    """
    if len(x.shape) < 3:
        raise ValueError("the dimension of x must be >= 3")
    if mask is not None and mask.shape != x.shape[2:]:
        raise ValueError(
            "dimension mismatch between x and mask: x.shape[2:]=%s, mask.shape=%s"
            % (str(x.shape[2:]), str(mask.shape))
        )
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values")

    if d == 1:
        x = np.diff(x, axis=0)

    p = x.shape[0] - 1
    q = x.shape[1]

    gamma = []
    for k in range(p + 1):
        if window_radius == np.inf:
            gamma_k = np.empty((q, q))
        else:
            gamma_k = np.empty(np.hstack([x.shape[2:], [q, q]]))
        for i in range(q):
            x_i = x[-1, i, :]
            for j in range(q):
                x_j = x[-(k + 1), j, :]
                if window_radius == np.inf:
                    gamma_k[i, j] = np.corrcoef(x_i.flatten(), x_j.flatten())[0, 1]
                else:
                    gamma_k[..., i, j] = _moving_window_corrcoef(
                        x_i, x_j, window_radius, window=window, mask=mask
                    )

        gamma.append(gamma_k)

    return gamma


def _moving_window_corrcoef(x, y, window_radius, window="gaussian", mask=None):
    if window not in ["gaussian", "uniform"]:
        raise ValueError(
            "unknown window type %s, the available options are 'gaussian' and 'uniform'"
            % window
        )

    if mask is None:
        mask = np.ones(x.shape)
    else:
        x = x.copy()
        x[~mask] = 0.0
        y = y.copy()
        y[~mask] = 0.0
        mask = mask.astype(float)

    if window == "gaussian":
        convol_filter = ndimage.gaussian_filter
    else:
        convol_filter = ndimage.uniform_filter

    if window == "uniform":
        window_size = 2 * window_radius + 1
    else:
        window_size = window_radius

    n = convol_filter(mask, window_size, mode="constant") * window_size**2

    sx = convol_filter(x, window_size, mode="constant") * window_size**2
    sy = convol_filter(y, window_size, mode="constant") * window_size**2

    ssx = convol_filter(x**2, window_size, mode="constant") * window_size**2
    ssy = convol_filter(y**2, window_size, mode="constant") * window_size**2
    sxy = convol_filter(x * y, window_size, mode="constant") * window_size**2

    mux = sx / n
    muy = sy / n

    stdx = np.sqrt(ssx - 2 * mux * sx + n * mux**2)
    stdy = np.sqrt(ssy - 2 * muy * sy + n * muy**2)
    cov = sxy - muy * sx - mux * sy + n * mux * muy

    mask = np.logical_and(stdx > 1e-8, stdy > 1e-8)
    mask = np.logical_and(mask, stdx * stdy > 1e-8)
    mask = np.logical_and(mask, n >= 3)
    corr = np.empty(x.shape)
    corr[mask] = cov[mask] / (stdx[mask] * stdy[mask])
    corr[~mask] = np.nan

    return corr
