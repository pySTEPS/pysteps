# -*- coding: utf-8 -*-
"""
pysteps.nowcasts.anvil
======================

Implementation of the autoregressive nowcasting using VIL (ANVIL) nowcasting
method developed in :cite:`PCLH2020`. Compared to S-PROG, the main improvements
are using an autoregressive integrated (ARI) model and the option to use
vertically integrated liquid (VIL) as the input variable. Using the ARI model
avoids biasedness and loss of small-scale features in the forecast field, and
no statistical post-processing is needed. In addition, the model allows
localization of parameter estimates. It was shown in :cite:`PCLH2020` that due
to the above improvements, ANVIL produces more reliable deterministic nowcasts
than S-PROG.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import time
import numpy as np
from scipy.ndimage import gaussian_filter
from pysteps import cascade, extrapolation
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.timeseries import autoregression
from pysteps import utils

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


def forecast(
    vil,
    velocity,
    timesteps,
    rainrate=None,
    n_cascade_levels=8,
    extrap_method="semilagrangian",
    ar_order=2,
    ar_window_radius=50,
    r_vil_window_radius=3,
    fft_method="numpy",
    apply_rainrate_mask=True,
    num_workers=1,
    extrap_kwargs=None,
    filter_kwargs=None,
    measure_time=False,
):
    """
    Generate a nowcast by using the autoregressive nowcasting using VIL
    (ANVIL) method. ANVIL is built on top of an extrapolation-based nowcast.
    The key features are:

    1) Growth and decay: implemented by using a cascade decomposition and
       a multiscale autoregressive integrated ARI(p,1) model. Instead of the
       original time series, the ARI model is applied to the differenced one
       corresponding to time derivatives.
    2) Originally designed for using integrated liquid (VIL) as the input data.
       In this case, the rain rate (R) is obtained from VIL via an empirical
       relation. This implementation is more general so that the input can be
       any two-dimensional precipitation field.
    3) The parameters of the ARI model and the R(VIL) relation are allowed to
       be spatially variable. The estimation is done using a moving window.

    Parameters
    ----------
    vil: array_like
        Array of shape (ar_order+2,m,n) containing the input fields ordered by
        timestamp from oldest to newest. The inputs are expected to contain VIL
        or rain rate. The time steps between the inputs are assumed to be regular.
    velocity: array_like
        Array of shape (2,m,n) containing the x- and y-components of the
        advection field. The velocities are assumed to represent one time step
        between the inputs. All values are required to be finite.
    timesteps: int or list of floats
        Number of time steps to forecast or a list of time steps for which the
        forecasts are computed (relative to the input time step). The elements
        of the list are required to be in ascending order.
    rainrate: array_like
        Array of shape (m,n) containing the most recently observed rain rate
        field. If set to None, no R(VIL) conversion is done and the outputs
        are in the same units as the inputs.
    n_cascade_levels: int, optional
        The number of cascade levels to use.
    extrap_method: str, optional
        Name of the extrapolation method to use. See the documentation of
        pysteps.extrapolation.interface.
    ar_order: int, optional
        The order of the autoregressive model to use. The recommended values
        are 1 or 2. Using a higher-order model is strongly discouraged because
        the stationarity of the AR process cannot be guaranteed.
    ar_window_radius: int, optional
        The radius of the window to use for determining the parameters of the
        autoregressive model. Set to None to disable localization.
    r_vil_window_radius: int, optional
        The radius of the window to use for determining the R(VIL) relation.
        Applicable if rainrate is not None.
    fft_method: str, optional
        A string defining the FFT method to use (see utils.fft.get_method).
        Defaults to 'numpy' for compatibility reasons. If pyFFTW is installed,
        the recommended method is 'pyfftw'.
    apply_rainrate_mask: bool
        Apply mask to prevent producing precipitation to areas where it was not
        originally observed. Defaults to True. Disabling this may improve some
        verification metrics but increases the number of false alarms. Applicable
        if rainrate is None.
    num_workers: int, optional
        The number of workers to use for parallel computation. Applicable if
        dask is installed or pyFFTW is used for computing the FFT.
        When num_workers>1, it is advisable to disable OpenMP by setting
        the environment variable OMP_NUM_THREADS to 1.
        This avoids slowdown caused by too many simultaneous threads.
    extrap_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the extrapolation
        method. See the documentation of pysteps.extrapolation.
    filter_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the filter method.
        See the documentation of pysteps.cascade.bandpass_filters.py.
    measure_time: bool, optional
        If True, measure, print and return the computation time.

    Returns
    -------
    out: ndarray
        A three-dimensional array of shape (num_timesteps,m,n) containing a time
        series of forecast precipitation fields. The time series starts from
        t0+timestep, where timestep is taken from the input VIL/rain rate
        fields. If measure_time is True, the return value is a three-element
        tuple containing the nowcast array, the initialization time of the
        nowcast generator and the time used in the main loop (seconds).

    References
    ----------
    :cite:`PCLH2020`
    """
    _check_inputs(vil, rainrate, velocity, timesteps, ar_order)

    if extrap_kwargs is None:
        extrap_kwargs = dict()
    else:
        extrap_kwargs = extrap_kwargs.copy()

    if filter_kwargs is None:
        filter_kwargs = dict()

    print("Computing ANVIL nowcast:")
    print("------------------------")
    print("")

    print("Inputs:")
    print("-------")
    print("input dimensions: %dx%d" % (vil.shape[1], vil.shape[2]))
    print("")

    print("Methods:")
    print("--------")
    print("extrapolation:   %s" % extrap_method)
    print("FFT:             %s" % fft_method)
    print("")

    print("Parameters:")
    print("-----------")
    if isinstance(timesteps, int):
        print("number of time steps:        %d" % timesteps)
    else:
        print("time steps:                  %s" % timesteps)
    print("parallel threads:            %d" % num_workers)
    print("number of cascade levels:    %d" % n_cascade_levels)
    print("order of the ARI(p,1) model: %d" % ar_order)
    if type(ar_window_radius) == int:
        print("ARI(p,1) window radius:      %d" % ar_window_radius)
    else:
        print("ARI(p,1) window radius:      none")

    print("R(VIL) window radius:        %d" % r_vil_window_radius)

    if measure_time:
        starttime_init = time.time()

    m, n = vil.shape[1:]
    vil = vil.copy()

    if rainrate is None and apply_rainrate_mask:
        rainrate_mask = vil[-1, :] < 0.1

    if rainrate is not None:
        # determine the coefficients fields of the relation R=a*VIL+b by
        # localized linear regression
        r_vil_a, r_vil_b = _r_vil_regression(vil[-1, :], rainrate, r_vil_window_radius)

    # transform the input fields to Lagrangian coordinates by extrapolation
    extrapolator = extrapolation.get_method(extrap_method)
    res = list()

    def worker(vil, i):
        return (
            i,
            extrapolator(
                vil[i, :],
                velocity,
                vil.shape[0] - 1 - i,
                allow_nonfinite_values=True,
                **extrap_kwargs,
            )[-1],
        )

    for i in range(vil.shape[0] - 1):
        if not DASK_IMPORTED or num_workers == 1:
            vil[i, :, :] = worker(vil, i)[1]
        else:
            res.append(dask.delayed(worker)(vil, i))

    if DASK_IMPORTED and num_workers > 1:
        num_workers_ = len(res) if num_workers > len(res) else num_workers
        vil_e = dask.compute(*res, num_workers=num_workers_)
        for i in range(len(vil_e)):
            vil[vil_e[i][0], :] = vil_e[i][1]

    # compute the final mask as the intersection of the masks of the advected
    # fields
    mask = np.isfinite(vil[0, :])
    for i in range(1, vil.shape[0]):
        mask = np.logical_and(mask, np.isfinite(vil[i, :]))

    if rainrate is None and apply_rainrate_mask:
        rainrate_mask = np.logical_and(rainrate_mask, mask)

    # apply cascade decomposition to the advected input fields
    bp_filter_method = cascade.get_method("gaussian")
    bp_filter = bp_filter_method((m, n), n_cascade_levels, **filter_kwargs)

    fft = utils.get_method(fft_method, shape=vil.shape[1:], n_threads=num_workers)

    decomp_method, recomp_method = cascade.get_method("fft")

    vil_dec = np.empty((n_cascade_levels, vil.shape[0], m, n))
    for i in range(vil.shape[0]):
        vil_ = vil[i, :].copy()
        vil_[~np.isfinite(vil_)] = 0.0
        vil_dec_i = decomp_method(vil_, bp_filter, fft_method=fft)
        for j in range(n_cascade_levels):
            vil_dec[j, i, :] = vil_dec_i["cascade_levels"][j, :]

    # compute time-lagged correlation coefficients for the cascade levels of
    # the advected and differenced input fields
    gamma = np.empty((n_cascade_levels, ar_order, m, n))
    for i in range(n_cascade_levels):
        vil_diff = np.diff(vil_dec[i, :], axis=0)
        vil_diff[~np.isfinite(vil_diff)] = 0.0
        for j in range(ar_order):
            gamma[i, j, :] = _moving_window_corrcoef(
                vil_diff[-1, :], vil_diff[-(j + 2), :], ar_window_radius
            )

    if ar_order == 2:
        # if the order of the ARI model is 2, adjust the correlation coefficients
        # so that the resulting process is stationary
        for i in range(n_cascade_levels):
            gamma[i, 1, :] = autoregression.adjust_lag2_corrcoef2(
                gamma[i, 0, :], gamma[i, 1, :]
            )

    # estimate the parameters of the ARI models
    phi = []
    for i in range(n_cascade_levels):
        if ar_order > 2:
            phi_ = autoregression.estimate_ar_params_yw_localized(gamma[i, :], d=1)
        elif ar_order == 2:
            phi_ = _estimate_ar2_params(gamma[i, :])
        else:
            phi_ = _estimate_ar1_params(gamma[i, :])
        phi.append(phi_)

    vil_dec = vil_dec[:, -(ar_order + 1) :, :]

    if measure_time:
        init_time = time.time() - starttime_init

    print("Starting nowcast computation.")

    if measure_time:
        starttime_mainloop = time.time()

    r_f = []

    if isinstance(timesteps, int):
        timesteps = range(timesteps + 1)
        timestep_type = "int"
    else:
        original_timesteps = [0] + list(timesteps)
        timesteps = nowcast_utils.binned_timesteps(original_timesteps)
        timestep_type = "list"

    if rainrate is not None:
        r_f_prev = r_vil_a * vil[-1, :] + r_vil_b
    else:
        r_f_prev = vil[-1, :]
    extrap_kwargs["return_displacement"] = True

    dp = None
    t_prev = 0.0

    for t, subtimestep_idx in enumerate(timesteps):
        if timestep_type == "list":
            subtimesteps = [original_timesteps[t_] for t_ in subtimestep_idx]
        else:
            subtimesteps = [t]

        if (timestep_type == "list" and subtimesteps) or (
            timestep_type == "int" and t > 0
        ):
            is_nowcast_time_step = True
        else:
            is_nowcast_time_step = False

        if is_nowcast_time_step:
            print(
                "Computing nowcast for time step %d... " % t,
                end="",
                flush=True,
            )

        if measure_time:
            starttime = time.time()

        # iterate the ARI models for each cascade level
        for i in range(n_cascade_levels):
            vil_dec[i, :] = autoregression.iterate_ar_model(vil_dec[i, :], phi[i])

        # recompose the cascade to obtain the forecast field
        vil_dec_dict = {}
        vil_dec_dict["cascade_levels"] = vil_dec[:, -1, :]
        vil_dec_dict["domain"] = "spatial"
        vil_dec_dict["normalized"] = False
        vil_f = recomp_method(vil_dec_dict)
        vil_f[~mask] = np.nan

        if rainrate is not None:
            # convert VIL to rain rate
            r_f_new = r_vil_a * vil_f + r_vil_b
        else:
            r_f_new = vil_f
            if apply_rainrate_mask:
                r_f_new[rainrate_mask] = 0.0

        r_f_new[r_f_new < 0.0] = 0.0

        # advect the recomposed field to obtain the forecast for the current
        # time step (or subtimesteps if non-integer time steps are given)
        for t_sub in subtimesteps:
            if t_sub > 0:
                t_diff_prev_int = t_sub - int(t_sub)
                if t_diff_prev_int > 0.0:
                    r_f_ip = (
                        1.0 - t_diff_prev_int
                    ) * r_f_prev + t_diff_prev_int * r_f_new
                else:
                    r_f_ip = r_f_prev

                t_diff_prev = t_sub - t_prev
                extrap_kwargs["displacement_prev"] = dp
                r_f_ep, dp = extrapolator(
                    r_f_ip,
                    velocity,
                    [t_diff_prev],
                    allow_nonfinite_values=True,
                    **extrap_kwargs,
                )
                r_f.append(r_f_ep[0])
                t_prev = t_sub

        # advect the forecast field by one time step if no subtimesteps in the
        # current interval were found
        if not subtimesteps:
            t_diff_prev = t + 1 - t_prev
            extrap_kwargs["displacement_prev"] = dp
            _, dp = extrapolator(
                None,
                velocity,
                [t_diff_prev],
                allow_nonfinite_values=True,
                **extrap_kwargs,
            )
            t_prev = t + 1

        r_f_prev = r_f_new

        if is_nowcast_time_step:
            if measure_time:
                print("%.2f seconds." % (time.time() - starttime))
            else:
                print("done.")

    if measure_time:
        mainloop_time = time.time() - starttime_mainloop

    if measure_time:
        return np.stack(r_f), init_time, mainloop_time
    else:
        return np.stack(r_f)


def _check_inputs(vil, rainrate, velocity, timesteps, ar_order):
    if vil.ndim != 3:
        raise ValueError(
            "vil.shape = %s, but a three-dimensional array expected" % str(vil.shape)
        )
    if rainrate is not None:
        if rainrate.ndim != 2:
            raise ValueError(
                "rainrate.shape = %s, but a two-dimensional array expected"
                % str(rainrate.shape)
            )
    if vil.shape[0] != ar_order + 2:
        raise ValueError(
            "vil.shape[0] = %d, but vil.shape[0] = ar_order + 2 = %d required"
            % (vil.shape[0], ar_order + 2)
        )
    if velocity.ndim != 3:
        raise ValueError(
            "velocity.shape = %s, but a three-dimensional array expected"
            % str(velocity.shape)
        )
    if isinstance(timesteps, list) and not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")


# optimized version of timeseries.autoregression.estimate_ar_params_yw_localized
# for an ARI(1,1) model
def _estimate_ar1_params(gamma):
    phi = []
    phi.append(1 + gamma[0, :])
    phi.append(-gamma[0, :])
    phi.append(np.zeros(gamma[0, :].shape))

    return phi


# optimized version of timeseries.autoregression.estimate_ar_params_yw_localized
# for an ARI(2,1) model
def _estimate_ar2_params(gamma):
    phi_diff = []
    phi_diff.append(gamma[0, :] * (1 - gamma[1, :]) / (1 - gamma[0, :] * gamma[0, :]))
    phi_diff.append(
        (gamma[1, :] - gamma[0, :] * gamma[0, :]) / (1 - gamma[0, :] * gamma[0, :])
    )

    phi = []
    phi.append(1 + phi_diff[0])
    phi.append(-phi_diff[0] + phi_diff[1])
    phi.append(-phi_diff[1])
    phi.append(np.zeros(phi_diff[0].shape))

    return phi


# Compute correlation coefficients of two 2d fields in a moving window with
# a Gaussian weight function. See Section II.G of PCLH2020. Differently to the
# standard formula for the Pearson correlation coefficient, the mean value of
# the inputs is assumed to be zero.
def _moving_window_corrcoef(x, y, window_radius):
    mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x.copy()
    x[~mask] = 0.0
    y = y.copy()
    y[~mask] = 0.0
    mask = mask.astype(float)

    if window_radius is not None:
        n = gaussian_filter(mask, window_radius, mode="constant")

        ssx = gaussian_filter(x**2, window_radius, mode="constant")
        ssy = gaussian_filter(y**2, window_radius, mode="constant")
        sxy = gaussian_filter(x * y, window_radius, mode="constant")
    else:
        n = np.mean(mask)

        ssx = np.mean(x**2)
        ssy = np.mean(y**2)
        sxy = np.mean(x * y)

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


# Determine the coefficients of the regression R=a*VIL+b.
# See Section II.G of PCLH2020.
# The parameters a and b are estimated in a localized fashion for each pixel
# in the input grid. This is done using a window specified by window_radius.
# Zero and non-finite values are not included. In addition, the regression is
# done by using a Gaussian weight function depending on the distance to the
# current grid point.
def _r_vil_regression(vil, r, window_radius):
    vil = vil.copy()
    vil[~np.isfinite(vil)] = 0.0

    r = r.copy()
    r[~np.isfinite(r)] = 0.0

    mask_vil = vil > 10.0
    mask_r = r > 0.1
    mask_obs = np.logical_and(mask_vil, mask_r)
    vil[~mask_obs] = 0.0
    r[~mask_obs] = 0.0

    n = gaussian_filter(mask_obs.astype(float), window_radius, mode="constant")

    sx = gaussian_filter(vil, window_radius, mode="constant")
    sx2 = gaussian_filter(vil * vil, window_radius, mode="constant")
    sxy = gaussian_filter(vil * r, window_radius, mode="constant")
    sy = gaussian_filter(r, window_radius, mode="constant")

    rhs1 = sxy
    rhs2 = sy

    m1 = sx2
    m2 = sx
    m3 = sx
    m4 = n

    c = 1.0 / (m1 * m4 - m2 * m3)

    m_inv_11 = c * m4
    m_inv_12 = -c * m2
    m_inv_21 = -c * m3
    m_inv_22 = c * m1

    mask = np.abs(m1 * m4 - m2 * m3) > 1e-8
    mask = np.logical_and(mask, n > 0.01)
    a = np.empty(vil.shape)
    a[mask] = m_inv_11[mask] * rhs1[mask] + m_inv_12[mask] * rhs2[mask]
    a[~mask] = 0.0
    a[~mask_vil] = 0.0
    b = np.empty(vil.shape)
    b[mask] = m_inv_21[mask] * rhs1[mask] + m_inv_22[mask] * rhs2[mask]
    b[~mask] = 0.0
    b[~mask_vil] = 0.0

    return a, b
