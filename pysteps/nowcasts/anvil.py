"""
pysteps.nowcasts.anvil
======================

Implementation of the autoregressive nowcasting using VIL (ANVIL) nowcasting
method developed in :cite:`PCHL2020`.

.. autosummary::
    :toctree: ../generated/
    
    forecast
"""

import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from pysteps import cascade, extrapolation
from pysteps.timeseries import autoregression


def forecast(vil, rainrate, velocity, n_timesteps, n_cascade_levels=8,
             extrap_method="semilagrangian", ar_order=2, ar_window_radius=50,
             r_vil_window_radius=5, fft_method="numpy", extrap_kwargs=None,
             filter_kwargs=None):
    """Generate a nowcast by using the autoregressive nowcasting using VIL
    (ANVIL) method. VIL is acronym for vertically integrated liquid.

    Parameters
    ----------
    vil : array_like
        Array of shape (ar_order+2,m,n) containing the input vil fields ordered
        by timestamp from oldest to newest. The time steps between the inputs
        are assumed to be regular.
    rainrate : array_like
        Array of shape (m,n) containing the most recently observed rain rate
        field. If set to None, the vil array is assumed to contain rain rates
        and no R(VIL) conversion is done.
    velocity : array_like
        Array of shape (2,m,n) containing the x- and y-components of the
        advection field. The velocities are assumed to represent one time step
        between the inputs. All values are required to be finite.
    n_timesteps : int
       Number of time steps to forecast. 
    n_cascade_levels : int, optional
        The number of cascade levels to use.
    extrap_method : str, optional
        Name of the extrapolation method to use. See the documentation of
        pysteps.extrapolation.interface.
    ar_order : int, optional
        The order of the autoregressive model to use.
    ar_window_radius : int, optional
        The radius of the window to use for determining the parameters of the
        autoregressive model.
    r_vil_window_radius : int, optional
        The radius of the window to use for determining the R(VIL) relation.
        Applicable if rainrate is not None.
    fft_method : str, optional
        A string defining the FFT method to use (see utils.fft.get_method).
        Defaults to 'numpy' for compatibility reasons. If pyFFTW is installed,
        the recommended method is 'pyfftw'.
    extrap_kwargs : dict, optional
        Optional dictionary containing keyword arguments for the extrapolation
        method. See the documentation of pysteps.extrapolation.
    filter_kwargs : dict, optional
        Optional dictionary containing keyword arguments for the filter method.
        See the documentation of pysteps.cascade.bandpass_filters.py.

    Returns
    -------
    out : ndarray
        A three-dimensional array of shape (n_timesteps,m,n) containing a time
        series of forecast precipitation fields. The time series starts from
        t0+timestep, where timestep is taken from the input VIL/rain rate fields.

    References
    ----------
    :cite:`PCHL2020`

    Notes
    -----
    The original ANVIL method developed in :cite:`PCHL2020` uses VIL as the
    input quantity. The forecast model is, however, more general and can take
    any two-dimensional input field.
    """
    if len(vil.shape) != 3:
        raise ValueError("vil.shape = %s, but a three-dimensional array expected" % str(vil.shape))

    if rainrate is not None:
        if len(rainrate.shape) != 2:
            raise ValueError("rainrate.shape = %s, but a two-dimensional array expected" % str(rainrate.shape))

    if vil.shape[0] != ar_order + 2:
        raise ValueError("vil.shape[0] = %d, but vil.shape[0] = ar_order = %d required" % (vil.shape[0], ar_order))

    if len(velocity.shape) != 3:
        raise ValueError("velocity.shape = %s, but a three-dimensional array expected" % str(velocity.shape))

    if extrap_kwargs is None:
        extrap_kwargs = dict()

    if filter_kwargs is None:
        filter_kwargs = dict()

    m, n = vil.shape[1:]
    vil = vil.copy()

    if rainrate is not None:
        r_vil_a, r_vil_b = _R_VIL_regression(vil[-1, :], rainrate, r_vil_window_radius)

    extrapolator = extrapolation.get_method(extrap_method)

    for i in range(vil.shape[0] - 1):
        vil[i, :] = extrapolator(vil[i, :], velocity, vil.shape[0]-1-i,
                                 allow_nonfinite_values=True)[-1]

    mask = np.isfinite(vil[0, :])
    for i in range(1, vil.shape[0]):
        mask = np.logical_and(mask, np.isfinite(vil[i, :]))

    bp_filter_method = cascade.get_method("gaussian")
    bp_filter = bp_filter_method((m, n), n_cascade_levels, **filter_kwargs)

    decomp_method, recomp_method = cascade.get_method("fft")

    vil_dec = np.empty((n_cascade_levels, vil.shape[0], m, n))
    for i in range(vil.shape[0]):
        vil_ = vil[i, :].copy()
        vil_[~np.isfinite(vil_)] = 0.0
        vil_dec_i = decomp_method(vil_, bp_filter)
        for j in range(n_cascade_levels):
            vil_dec[j, i, :] = vil_dec_i["cascade_levels"][j, :]

    gamma = np.empty((n_cascade_levels, ar_order, m, n))
    for i in range(n_cascade_levels):
        vil_diff = np.diff(vil_dec[i, :], axis=0)
        vil_diff[~np.isfinite(vil_diff)] = 0.0
        for j in range(ar_order):
            gamma[i, j, :] = _moving_window_corrcoef(vil_diff[-1, :],
                                                     vil_diff[-(j+2), :],
                                                     ar_window_radius)

    if ar_order == 2:
        for i in range(n_cascade_levels):
            gamma[i, 1, :] = autoregression.adjust_lag2_corrcoef2(gamma[i, 0, :],
                                                                  gamma[i, 1, :])

    phi = []
    for i in range(n_cascade_levels):
        phi.append(autoregression.estimate_ar_params_yw_localized(gamma[i, :], d=1))

    vil_dec = vil_dec[:, -3:, :]

    r_f = []
    dp = None
    for t in range(n_timesteps):
        print("Computing nowcast for time step %d... " % (t + 1), end="")
        sys.stdout.flush()

        for i in range(n_cascade_levels):
            vil_dec[i, :] = autoregression.iterate_ar_model(vil_dec[i, :], phi[i])

        vil_dec_dict = {}
        vil_dec_dict["cascade_levels"] = vil_dec[:, -1, :]
        vil_dec_dict["domain"] = "spatial"
        vil_dec_dict["normalized"] = False
        vil_f = recomp_method(vil_dec_dict)
        vil_f[~mask] = np.nan

        if rainrate is not None:
            r_f_ = r_vil_a * vil_f + r_vil_b
        else:
            r_f_ = vil_f

        extrap_kwargs.update({"D_prev": dp, "return_displacement": True,
                              "allow_nonfinite_values": True})
        r_f_, dp = extrapolator(r_f_, velocity, 1, **extrap_kwargs)

        print("done.")

        r_f.append(r_f_[-1])

    return np.stack(r_f)


def _moving_window_corrcoef(x, y, window_radius):
    mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x.copy()
    x[~mask] = 0.0
    y = y.copy()
    y[~mask] = 0.0
    mask = mask.astype(float)

    n = gaussian_filter(mask, window_radius, mode="constant")

    ssx = gaussian_filter(x**2, window_radius, mode="constant")
    ssy = gaussian_filter(y**2, window_radius, mode="constant")
    sxy = gaussian_filter(x*y, window_radius, mode="constant")

    stdx = np.sqrt(ssx / n)
    stdy = np.sqrt(ssy / n)
    cov = sxy / n

    mask = np.logical_and(stdx > 1e-8, stdy > 1e-8)
    mask = np.logical_and(mask, stdx * stdy > 1e-8)
    mask = np.logical_and(mask, n > 1e-3)
    corr = np.empty(x.shape)
    corr[mask] = cov[mask] / (stdx[mask] * stdy[mask])
    corr[~mask] = 0.0

    return corr


def _R_VIL_regression(vil, r, window_radius):
    vil = vil.copy()
    vil[~np.isfinite(vil)] = 0.0

    r = r.copy()
    r[~np.isfinite(r)] = 0.0

    mask = np.logical_and(vil > 1e-3, r > 1e-3)
    vil[~mask] = 0.0
    r[~mask] = 0.0

    n = gaussian_filter(mask.astype(float), window_radius, mode="constant")

    sx = gaussian_filter(vil, window_radius, mode="constant")
    sx2 = gaussian_filter(vil*vil, window_radius, mode="constant")
    sxy = gaussian_filter(vil*r, window_radius, mode="constant")
    sy = gaussian_filter(r, window_radius, mode="constant")

    rhs1 = sxy
    rhs2 = sy

    m1 = sx2
    m2 = sx
    m3 = sx
    m4 = n

    c = 1.0 / (m1*m4 - m2*m3)

    m_inv_11 = c * m4
    m_inv_12 = -c * m2
    m_inv_21 = -c * m3
    m_inv_22 = c * m1

    mask = np.abs(m1*m4 - m2*m3) > 1e-8
    mask = np.logical_and(mask, n > 1e-3)
    a = np.empty(vil.shape)
    a[mask] = m_inv_11[mask] * rhs1[mask] + m_inv_12[mask] * rhs2[mask]
    a[~mask] = 0.0
    b = np.empty(vil.shape)
    b[mask] = m_inv_21[mask] * rhs1[mask] + m_inv_22[mask] * rhs2[mask]
    b[~mask] = 0.0

    return a, b
