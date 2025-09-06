"""
pysteps.nowcasts.sprogloc
=========================

Implementation of the S-PROG Localized method described in :cite:`RRR2022`,
based in S-PROG method described in :cite:`Seed2003`

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import time

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from pysteps import cascade, extrapolation, utils
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.nowcasts.utils import compute_percentile_mask, nowcast_main_loop
from pysteps.postprocessing import probmatching
from pysteps.timeseries import autoregression, correlation
from pysteps.utils.check_norain import check_norain
from pysteps.utils import spectral

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


def forecast(
    precip,
    velocity,
    timesteps,
    precip_thr=None,
    norain_thr=0.0,
    n_cascade_levels=6,
    extrap_method="semilagrangian",
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    gamma_filter_method="uniform",
    ar_order=2,
    d_order=0,
    ar_window_radius=None,
    gamma_factor=1.0,
    conditional=False,
    probmatching_method="cdf",
    num_workers=1,
    fft_method="numpy",
    domain="spatial",
    extrap_kwargs=None,
    filter_kwargs=None,
    measure_time=False,
):
    """
    Generate a nowcast by using the Spectral Prognosis Localized (S-PROG-LOC) method.

    Parameters
    ----------
    precip: array-like
        Array of shape (ar_order+d_order+1,m,n) containing the input precipitation fields
        ordered by timestamp from oldest to newest. The time steps between
        the inputs are assumed to be regular.
    velocity: array-like
        Array of shape (2,m,n) containing the x- and y-components of the
        advection field.
        The velocities are assumed to represent one time step between the
        inputs. All values are required to be finite.
    timesteps: int or list of floats
        Number of time steps to forecast or a list of time steps for which the
        forecasts are computed (relative to the input time step). The elements
        of the list are required to be in ascending order.
    precip_thr: float, required
        The threshold value for minimum observable precipitation intensity.
    norain_thr: float
      Specifies the threshold value for the fraction of rainy (see above) pixels
      in the radar rainfall field below which we consider there to be no rain.
      Depends on the amount of clutter typically present.
      Standard set to 0.0
    n_cascade_levels: int, optional
        The number of cascade levels to use. Defaults to 6, see issue #385
        on GitHub.
    extrap_method: str, optional
        Name of the extrapolation method to use. See the documentation of
        pysteps.extrapolation.interface.
    decomp_method: {'fft'}, optional
        Name of the cascade decomposition method to use. See the documentation
        of pysteps.cascade.interface.
    bandpass_filter_method: {'gaussian', 'uniform'}, optional
        Name of the bandpass filter method to use with the cascade decomposition.
        See the documentation of pysteps.cascade.interface.
    ar_order: int, optional
        The order of the autoregressive integrated model to use. Must be >= 1.
    d_order: int, optional
        The differencing order of the autoregressive integrated model to use.
        Must be >= 0. If d == 0, then ARI(p,0) == AR(p), as used in S-PROG model.
    ar_window_radius: int or list, optional
        The radius of the window to use for determining the parameters of the
        autoregressive integrated model. Set to None to disable localization.
    conditional: bool, optional
        If set to True, compute the statistics of the precipitation field
        conditionally by excluding pixels where the values are
        below the threshold precip_thr.
    probmatching_method: {'cdf','mean',None}, optional
        Method for matching the conditional statistics of the forecast field
        (areas with precipitation intensity above the threshold precip_thr) with
        those of the most recently observed one. 'cdf'=map the forecast CDF to the
        observed one, 'mean'=adjust only the mean value,
        None=no matching applied.
    num_workers: int, optional
        The number of workers to use for parallel computation. Applicable if dask
        is enabled or pyFFTW is used for computing the FFT.
        When num_workers>1, it is advisable to disable OpenMP by setting
        the environment variable OMP_NUM_THREADS to 1.
        This avoids slowdown caused by too many simultaneous threads.
    fft_method: str, optional
        A string defining the FFT method to use (see utils.fft.get_method).
        Defaults to 'numpy' for compatibility reasons. If pyFFTW is installed,
        the recommended method is 'pyfftw'.
    domain: {"spatial", "spectral"}
        If "spatial", all computations are done in the spatial domain (the
        classical S-PROG model). If "spectral", the ARI(p,d) models are applied
        directly in the spectral domain to reduce memory footprint and improve
        performance :cite:`PCH2019a`.
    extrap_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the extrapolation
        method. See the documentation of pysteps.extrapolation.
    filter_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the filter method.
        See the documentation of pysteps.cascade.bandpass_filters.py.
    measure_time: bool
        If set to True, measure, print and return the computation time.

    Returns
    -------
    out: ndarray
        A three-dimensional array of shape (num_timesteps,m,n) containing a time
        series of forecast precipitation fields. The time series starts from
        t0+timestep, where timestep is taken from the input precipitation fields
        precip. If measure_time is True, the return value is a three-element
        tuple containing the nowcast array, the initialization time of the
        nowcast generator and the time used in the main loop (seconds).

    See also
    --------
    pysteps.extrapolation.interface, pysteps.cascade.interface

    References
    ----------
    :cite:`Seed2003`, :cite:`PCH2019a`, :cite:`RRR2022`
    """

    _check_inputs(precip, velocity, timesteps, ar_order, d_order, ar_window_radius)

    if extrap_kwargs is None:
        extrap_kwargs = dict()

    if filter_kwargs is None:
        filter_kwargs = dict()

    if np.any(~np.isfinite(velocity)):
        raise ValueError("velocity contains non-finite values")

    if precip_thr is None:
        raise ValueError("precip_thr required but not specified")

    # Fix length of ar_windows_radius for localization
    if ar_window_radius is None:
        ar_window_radius = np.full(n_cascade_levels, np.inf)
    elif isinstance(ar_window_radius, int):
        ar_window_radius = np.full(n_cascade_levels, ar_window_radius)
    elif isinstance(ar_window_radius, list):
        # Use user-defined window sizes
        ar_window_radius = np.array(ar_window_radius)

    print("Computing S-PROG-LOC nowcast")
    print("------------------------")
    print("")

    print("Inputs")
    print("------")
    print(f"input dimensions: {precip.shape[1]}x{precip.shape[2]}")
    print("")

    print("Methods")
    print("-------")
    print(f"extrapolation:          {extrap_method}")
    print(f"bandpass filter:        {bandpass_filter_method}")
    print(f"decomposition:          {decomp_method}")
    print("conditional statistics: {}".format("yes" if conditional else "no"))
    print(f"probability matching:   {probmatching_method}")
    print(f"FFT method:             {fft_method}")
    print(f"domain:                 {domain}")
    print("")

    print("Parameters")
    print("----------")
    if isinstance(timesteps, int):
        print(f"number of time steps:     {timesteps}")
    else:
        print(f"time steps:               {timesteps}")
    print(f"parallel threads:         {num_workers}")
    print(f"number of cascade levels: {n_cascade_levels}")
    print(f"order of the ARI(p,d) model: {ar_order}")
    print(f"differencing order of the ARI(p,d) model: {d_order}")
    print(f"ARI(p,d) window radius:      {ar_window_radius}")
    print(f"precip. intensity threshold: {precip_thr}")

    if measure_time:
        starttime_init = time.time()
    else:
        starttime_init = None

    fft = utils.get_method(fft_method, shape=precip.shape[1:], n_threads=num_workers)

    m, n = precip.shape[1:]

    # initialize the band-pass filter
    filter_method = cascade.get_method(bandpass_filter_method)
    bp_filter = filter_method((m, n), n_cascade_levels, **filter_kwargs)

    decomp_method, recomp_method = cascade.get_method(decomp_method)

    extrapolator_method = extrapolation.get_method(extrap_method)

    precip = precip[-(ar_order + d_order + 1) :, :, :].copy()
    precip_min = np.nanmin(precip)

    # determine the domain mask from non-finite values
    domain_mask = np.logical_or.reduce(
        [~np.isfinite(precip[i, :]) for i in range(precip.shape[0])]
    )

    if check_norain(precip, precip_thr, norain_thr, None):
        return nowcast_utils.zero_precipitation_forecast(
            None, timesteps, precip, None, True, measure_time, starttime_init
        )

    # determine the precipitation threshold mask
    if conditional:
        mask_thr = np.logical_and.reduce(
            [precip[i, :, :] >= precip_thr for i in range(precip.shape[0])]
        )
    else:
        mask_thr = None

    # initialize the extrapolator
    x_values, y_values = np.meshgrid(
        np.arange(precip.shape[2]), np.arange(precip.shape[1])
    )

    xy_coords = np.stack([x_values, y_values])

    extrap_kwargs = extrap_kwargs.copy()
    extrap_kwargs["xy_coords"] = xy_coords
    extrap_kwargs["allow_nonfinite_values"] = (
        True if np.any(~np.isfinite(precip)) else False
    )

    # advect the previous precipitation fields to the same position with the
    # most recent one (i.e. transform them into the Lagrangian coordinates)
    res = list()

    def f(precip, i):
        return extrapolator_method(
            precip[i, :], velocity, ar_order + d_order - i, "min", **extrap_kwargs
        )[-1]

    for i in range(ar_order + d_order):
        if not DASK_IMPORTED:
            precip[i, :, :] = f(precip, i)
        else:
            res.append(dask.delayed(f)(precip, i))

    if DASK_IMPORTED:
        num_workers_ = len(res) if num_workers > len(res) else num_workers
        precip = np.stack(
            list(dask.compute(*res, num_workers=num_workers_)) + [precip[-1, :, :]]
        )

    # replace non-finite values with the minimum value
    precip = precip.copy()
    for i in range(precip.shape[0]):
        precip[i, ~np.isfinite(precip[i, :])] = np.nanmin(precip[i, :])

    # compute the cascade decompositions of the input precipitation fields
    precip_decomp = []
    for i in range(ar_order + d_order + 1):
        precip_ = decomp_method(
            precip[i, :, :],
            bp_filter,
            mask=mask_thr,
            fft_method=fft,
            output_domain=domain,
            normalize=True,
            compute_stats=True,
            compact_output=True,
        )
        precip_decomp.append(precip_)

    # rearrange the cascade levels into a four-dimensional array of shape
    # (n_cascade_levels,ar_order+d_order+1,m,n) for the autoregressive integrated model
    precip_cascades = nowcast_utils.stack_cascades(
        precip_decomp, n_cascade_levels, convert_to_full_arrays=True
    )

    # compute localized lag-l temporal autocorrelation coefficients for each cascade level
    gamma = np.empty((n_cascade_levels, ar_order, m, n))
    for i in range(n_cascade_levels):
        gamma_ = np.array(
            _temporal_autocorrelation(
                precip_cascades[i],
                d=d_order,
                domain=domain,
                x_shape=(m, n),
                mask=mask_thr,
                window=gamma_filter_method,
                window_radius=ar_window_radius[i],
            )
        )
        # Adjust shape if no localization
        if gamma_.ndim == 1:
            for j in range(len(gamma_)):
                gamma[i, j] = np.full((m, n), gamma_[j])
        # Assign values if localization
        elif gamma_.ndim == 3:
            gamma[i, :] = gamma_

    # Adjust autocorrelation coefficients by the coefficient factor
    gamma *= gamma_factor
    gamma[gamma >= 1] = 0.999999

    if ar_order == 2:
        # adjust the lag-2 correlation coefficient to ensure that the ARI(p,d)
        # process is stationary
        for i in range(n_cascade_levels):
            gamma[i, 1] = autoregression.adjust_lag2_corrcoef2(gamma[i, 0], gamma[i, 1])

    nowcast_utils.print_corrcoefs(np.nanmean(gamma, axis=(-2, -1)))

    precip_cascades = nowcast_utils.stack_cascades(
        precip_decomp, n_cascade_levels, convert_to_full_arrays=False
    )

    precip_decomp = precip_decomp[-1]

    # estimate the parameters of the ARI(p,d) model from the autocorrelation
    # coefficients
    phi = np.empty((n_cascade_levels, ar_order + d_order + 1, m, n))
    for i in range(n_cascade_levels):
        if ar_order > 2 or d_order > 1:
            phi[i, :] = autoregression.estimate_ar_params_yw_localized(
                gamma[i], d=d_order
            )
        elif ar_order == 2:
            phi[i, :] = _estimate_ar2_params(gamma[i], d=d_order)
        elif ar_order == 1:
            phi[i, :] = _estimate_ar1_params(gamma[i], d=d_order)

    nowcast_utils.print_ar_params(np.nanmean(np.array(phi), axis=(-2, -1)))

    # discard all except the p+d-1 last cascades because they are not needed for
    # the ARI(p,d) model
    precip_cascades = [
        precip_cascades[i][-(ar_order + d_order) :] for i in range(n_cascade_levels)
    ]

    if probmatching_method == "mean":
        mu_0 = np.mean(precip[-1, :, :][precip[-1, :, :] >= precip_thr])
    else:
        mu_0 = None

    # compute precipitation mask and wet area ratio
    mask_p = precip[-1, :, :] >= precip_thr
    war = 1.0 * np.sum(mask_p) / (precip.shape[1] * precip.shape[2])

    if measure_time:
        init_time = time.time() - starttime_init

    precip = precip[-1, :, :]

    print("Starting nowcast computation.")

    precip_forecast = []

    state = {"precip_cascades": precip_cascades, "precip_decomp": precip_decomp}
    params = {
        "domain": domain,
        "domain_mask": domain_mask,
        "fft": fft,
        "mu_0": mu_0,
        "n_cascade_levels": n_cascade_levels,
        "phi": phi,
        "precip_0": precip,
        "precip_min": precip_min,
        "probmatching_method": probmatching_method,
        "recomp_method": recomp_method,
        "war": war,
    }

    precip_forecast = nowcast_main_loop(
        precip,
        velocity,
        state,
        timesteps,
        extrap_method,
        _update,
        extrap_kwargs=extrap_kwargs,
        params=params,
        measure_time=measure_time,
    )
    if measure_time:
        precip_forecast, mainloop_time = precip_forecast

    precip_forecast = np.stack(precip_forecast)

    if measure_time:
        return precip_forecast, init_time, mainloop_time
    else:
        return precip_forecast


def _temporal_autocorrelation(
    x,
    d=0,
    domain="spatial",
    x_shape=None,
    mask=None,
    use_full_fft=False,
    window="gaussian",
    window_radius=np.inf,
):
    r"""
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
                ccg = np.corrcoef(x[-1, :][mask], x[-(k + 2), :][mask])[0, 1]
                cc = _moving_window_corrcoef(
                    x[-1, :], x[-(k + 2), :], window_radius, window=window, mask=mask
                )
                cc[~np.isfinite(cc)] = ccg
        else:
            cc = spectral.corrcoef(
                x[-1, :, :], x[-(k + 2), :], x_shape, use_full_fft=use_full_fft
            )
        gamma.append(cc)

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
        # mask = np.logical_and(np.isfinite(x), np.isfinite(y))
        x = x.copy()
        x[~mask] = 0.0
        y = y.copy()
        y[~mask] = 0.0
        mask = mask.astype(float)

    if window == "gaussian":
        convol_filter = gaussian_filter
        window_size = window_radius
    else:
        convol_filter = uniform_filter
        window_size = 2 * window_radius + 1

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


# optimized version of timeseries.autoregression.estimate_ar_params_yw_localized
# for an ARI(1,1) model
def _estimate_ar1_params(gamma, d):
    """
    Estimate AR(1) parameters for a given autocorrelation structure.

    Parameters:
    - gamma (np.ndarray): The autocorrelation coefficients for the time series.
    - d (int): The differencing order (0 for AR, 1 for ARI).

    Returns:
    - phi (np.ndarray): Estimated AR(1) parameters.
    """
    phi = []
    phi1 = gamma[0, :]
    phi0 = np.sqrt(np.maximum(1.0 - phi1**2, 0))

    if d == 0:
        # AR(1,0) model (no differencing)
        phi.append(phi1)
        # Noise term
        phi.append(phi0)

    elif d == 1:
        # ARI(1,1) model (first differenced)
        phi1_d = 1 + phi1
        phi2_d = -phi1
        phi.append(phi1_d)
        phi.append(phi2_d)
        # Noise term
        phi.append(phi0)  ## Check this adjusting for phi0

    return np.array(phi)


def _estimate_ar2_params(gamma, d):
    """
    Estimate AR(2) parameters for a given autocorrelation structure.

    Parameters:
    - gamma (np.ndarray): The autocorrelation coefficients for the time series.
    - d (int): The differencing order (0 for AR, 1 for ARI).

    Returns:
    - phi (np.ndarray): Estimated AR(2) parameters.
    """
    phi = []
    g1 = gamma[0, :]
    g2 = gamma[1, :]
    phi1 = (g1 * (1.0 - g2)) / (1.0 - g1**2)
    phi2 = (g2 - g1**2) / (1.0 - g1**2)
    phi0 = np.sqrt(np.maximum(1.0 - phi1 * g1 - phi2 * g2, 0))

    if d == 0:
        # AR(2,0) model (no differencing)
        phi.append(phi1)
        phi.append(phi2)
        # Noise term
        phi.append(phi0)
    elif d == 1:
        # ARI(2,1) model (first differenced)
        phi1_d = 1 + phi1  # Correctly adjusting phi1 before modifying phi2
        phi2_d = phi2 - phi1  # Ensure correct order of modification
        phi3_d = -phi2  # Standard approach for phi3
        phi.append(phi1_d)
        phi.append(phi2_d)
        phi.append(phi3_d)
        # Noise term
        phi.append(phi0)

    return np.array(phi)


def _check_inputs(precip, velocity, timesteps, ar_order, d_order, ar_window_radius):
    if precip.ndim != 3:
        raise ValueError("precip must be a three-dimensional array")
    if precip.shape[0] < ar_order + d_order + 1:
        raise ValueError("precip.shape[0] < ar_order+d_order+1")
    if velocity.ndim != 3:
        raise ValueError("velocity must be a three-dimensional array")
    if precip.shape[1:3] != velocity.shape[1:3]:
        raise ValueError(
            "dimension mismatch between precip and velocity: shape(precip)=%s, shape(velocity)=%s"
            % (str(precip.shape), str(velocity.shape))
        )
    if isinstance(timesteps, list) and not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")
    if ar_window_radius is not None:
        if not isinstance(ar_window_radius, (int, list)):
            raise ValueError("ar_window_radius type must be None, int or list")


def _update(state, params):
    for i in range(params["n_cascade_levels"]):
        state["precip_cascades"][i] = autoregression.iterate_ar_model(
            state["precip_cascades"][i], params["phi"][i, :]
        )

    state["precip_decomp"]["cascade_levels"] = [
        state["precip_cascades"][i][-1, :] for i in range(params["n_cascade_levels"])
    ]
    if params["domain"] == "spatial":
        state["precip_decomp"]["cascade_levels"] = np.stack(
            state["precip_decomp"]["cascade_levels"]
        )

    precip_forecast_recomp = params["recomp_method"](state["precip_decomp"])

    if params["domain"] == "spectral":
        precip_forecast_recomp = params["fft"].irfft2(precip_forecast_recomp)

    mask = compute_percentile_mask(precip_forecast_recomp, params["war"])
    precip_forecast_recomp[~mask] = params["precip_min"]

    if params["probmatching_method"] == "cdf":
        # adjust the CDF of the forecast to match the most recently
        # observed precipitation field
        precip_forecast_recomp = probmatching.nonparam_match_empirical_cdf(
            precip_forecast_recomp, params["precip_0"]
        )
    elif params["probmatching_method"] == "mean":
        mu_fct = np.mean(precip_forecast_recomp[mask])
        precip_forecast_recomp[mask] = (
            precip_forecast_recomp[mask] - mu_fct + params["mu_0"]
        )

    precip_forecast_recomp[params["domain_mask"]] = np.nan

    return precip_forecast_recomp, state
