"""
pysteps.nowcasts.sprog
======================

Implementation of the S-PROG method described in :cite:`Seed2003`

.. autosummary::
    :toctree: ../generated/
    
    forecast
"""

import sys
import time

import numpy as np

from pysteps import cascade
from pysteps import extrapolation
from pysteps import utils
from pysteps.postprocessing import probmatching
from pysteps.timeseries import autoregression, correlation
from pysteps.nowcasts import utils as nowcast_utils

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


def forecast(R, V, n_timesteps, n_cascade_levels=6, R_thr=None,
             extrap_method="semilagrangian", decomp_method="fft",
             bandpass_filter_method="gaussian", ar_order=2, conditional=False,
             probmatching_method="mean", num_workers=1, fft_method="numpy",
             extrap_kwargs=None, filter_kwargs=None, measure_time=False):
    """Generate a nowcast by using the Spectral Prognosis (S-PROG) method.

    Parameters
    ----------
    R : array-like
      Array of shape (ar_order+1,m,n) containing the input precipitation fields
      ordered by timestamp from oldest to newest. The time steps between the inputs
      are assumed to be regular, and the inputs are required to have finite values.
    V : array-like
      Array of shape (2,m,n) containing the x- and y-components of the advection
      field. The velocities are assumed to represent one time step between the
      inputs. All values are required to be finite.
    n_timesteps : int
      Number of time steps to forecast.
    n_cascade_levels : int, optional
      The number of cascade levels to use.
    R_thr : float, optional
      Specifies the threshold value for minimum observable precipitation
      intensity. Required if mask_method is not None or conditional is True.
    extrap_method : str, optional
      Name of the extrapolation method to use. See the documentation of
      pysteps.extrapolation.interface.
    decomp_method : {'fft'}, optional
      Name of the cascade decomposition method to use. See the documentation
      of pysteps.cascade.interface.
    bandpass_filter_method : {'gaussian', 'uniform'}, optional
      Name of the bandpass filter method to use with the cascade decomposition.
      See the documentation of pysteps.cascade.interface.
    ar_order : int, optional
      The order of the autoregressive model to use. Must be >= 1.
    conditional : bool, optional
      If set to True, compute the statistics of the precipitation field
      conditionally by excluding pixels where the values are below the threshold
      R_thr.
    probmatching_method : {'cdf','mean',None}, optional
      Method for matching the conditional statistics of the forecast field
      (areas with precipitation intensity above the threshold R_thr) with those
      of the most recently observed one. 'cdf'=map the forecast CDF to the
      observed one, 'mean'=adjust only the mean value, None=no matching applied.
    num_workers : int, optional
      The number of workers to use for parallel computation. Applicable if dask
      is enabled or pyFFTW is used for computing the FFT. When num_workers>1, it
      is advisable to disable OpenMP by setting the environment variable
      OMP_NUM_THREADS to 1. This avoids slowdown caused by too many simultaneous
      threads.
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
    measure_time : bool
      If set to True, measure, print and return the computation time.

    Returns
    -------
    out : ndarray
      A three-dimensional array of shape (n_timesteps,m,n) containing a time
      series of forecast precipitation fields. The time series starts from
      t0+timestep, where timestep is taken from the input precipitation fields
      R. If measure_time is True, the return value is a three-element tuple
      containing the nowcast array, the initialization time of the nowcast
      generator and the time used in the main loop (seconds).

    See also
    --------
    pysteps.extrapolation.interface, pysteps.cascade.interface

    References
    ----------
    :cite:`Seed2003`

    """
    _check_inputs(R, V, ar_order)

    if extrap_kwargs is None:
        extrap_kwargs = dict()

    if filter_kwargs is None:
        filter_kwargs = dict()

    if np.any(~np.isfinite(R)):
        raise ValueError("R contains non-finite values")

    if np.any(~np.isfinite(V)):
        raise ValueError("V contains non-finite values")

    if conditional and R_thr is None:
        raise ValueError("conditional=True but R_thr is not set")

    print("Computing S-PROG nowcast:")
    print("-------------------------")
    print("")

    print("Inputs:")
    print("-------")
    print("input dimensions: %dx%d" % (R.shape[1], R.shape[2]))
    print("")

    print("Methods:")
    print("--------")
    print("extrapolation:          %s" % extrap_method)
    print("bandpass filter:        %s" % bandpass_filter_method)
    print("decomposition:          %s" % decomp_method)
    print("conditional statistics: %s" % ("yes" if conditional else "no"))
    print("probability matching:   %s" % probmatching_method)
    print("FFT method:             %s" % fft_method)
    print("")

    print("Parameters:")
    print("-----------")
    print("number of time steps:     %d" % n_timesteps)
    print("parallel threads:         %d" % num_workers)
    print("number of cascade levels: %d" % n_cascade_levels)
    print("order of the AR(p) model: %d" % ar_order)

    if conditional:
        print("precip. intensity threshold: %g" % R_thr)

    if measure_time:
        starttime_init = time.time()

    fft = utils.get_method(fft_method, shape=R.shape[1:], n_threads=num_workers)

    M, N = R.shape[1:]

    # initialize the band-pass filter
    filter_method = cascade.get_method(bandpass_filter_method)
    filter = filter_method((M, N), n_cascade_levels, **filter_kwargs)

    decomp_method = cascade.get_method(decomp_method)

    extrapolator_method = extrapolation.get_method(extrap_method)

    R = R[-(ar_order + 1):, :, :].copy()
    R_min = R.min()

    if conditional:
        MASK_thr = np.logical_and.reduce([R[i, :, :] >= R_thr for i in range(R.shape[0])])
    else:
        MASK_thr = None

    # initialize the extrapolator
    x_values, y_values = np.meshgrid(np.arange(R.shape[2]),
                                     np.arange(R.shape[1]))

    xy_coords = np.stack([x_values, y_values])

    extrap_kwargs = extrap_kwargs.copy()
    extrap_kwargs['xy_coords'] = xy_coords

    # advect the previous precipitation fields to the same position with the
    # most recent one (i.e. transform them into the Lagrangian coordinates)
    res = list()

    def f(R, i):
        return extrapolator_method(R[i, :, :], V, ar_order - i,
                                   "min",
                                   **extrap_kwargs)[-1]

    for i in range(ar_order):
        if not DASK_IMPORTED:
            R[i, :, :] = f(R, i)
        else:
            res.append(dask.delayed(f)(R, i))

    if DASK_IMPORTED:
        num_workers_ = len(res) if num_workers > len(res) else num_workers
        R = np.stack(list(dask.compute(*res, num_workers=num_workers_)) + [R[-1, :, :]])

    # compute the cascade decompositions of the input precipitation fields
    R_d = []
    for i in range(ar_order + 1):
        R_ = decomp_method(R[i, :, :], filter, MASK=MASK_thr, fft_method=fft)
        R_d.append(R_)

    # normalize the cascades and rearrange them into a four-dimensional array
    # of shape (n_cascade_levels,ar_order+1,m,n) for the autoregressive model
    R_c, mu, sigma = nowcast_utils.stack_cascades(R_d, n_cascade_levels)

    # compute lag-l temporal autocorrelation coefficients for each cascade level
    GAMMA = np.empty((n_cascade_levels, ar_order))
    for i in range(n_cascade_levels):
        R_c_ = np.stack([R_c[i, j, :, :] for j in range(ar_order + 1)])
        GAMMA[i, :] = correlation.temporal_autocorrelation(R_c_, MASK=MASK_thr)

    nowcast_utils.print_corrcoefs(GAMMA)

    if ar_order == 2:
        # adjust the lag-2 correlation coefficient to ensure that the AR(p)
        # process is stationary
        for i in range(n_cascade_levels):
            GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef2(GAMMA[i, 0], GAMMA[i, 1])

    # estimate the parameters of the AR(p) model from the autocorrelation
    # coefficients
    PHI = np.empty((n_cascade_levels, ar_order + 1))
    for i in range(n_cascade_levels):
        PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])

    nowcast_utils.print_ar_params(PHI)

    # discard all except the p-1 last cascades because they are not needed for
    # the AR(p) model
    R_c = R_c[:, -ar_order:, :, :]

    D = None

    if probmatching_method == "mean":
        mu_0 = np.mean(R[-1, :, :][R[-1, :, :] >= R_thr])

    # compute precipitation mask and wet area ratio
    MASK_p = R[-1, :, :] >= R_thr
    war = 1.0 * np.sum(MASK_p) / (R.shape[1] * R.shape[2])

    if measure_time:
        init_time = time.time() - starttime_init

    R = R[-1, :, :]

    print("Starting nowcast computation.")

    if measure_time:
        starttime_mainloop = time.time()

    R_f = []

    # iterate each time step
    for t in range(n_timesteps):
        print("Computing nowcast for time step %d... " % (t + 1), end="")
        sys.stdout.flush()
        if measure_time:
            starttime = time.time()

        for i in range(n_cascade_levels):
            # use a separate AR(p) model for the non-perturbed forecast,
            # from which the mask is obtained
            R_c[i, :, :, :] = \
                autoregression.iterate_ar_model(R_c[i, :, :, :], PHI[i, :])

        R_c_ = nowcast_utils.recompose_cascade(R_c[:, -1, :, :], mu, sigma)

        MASK = _compute_sprog_mask(R_c_, war)
        R_c_[~MASK] = R_min

        if probmatching_method == "cdf":
            # adjust the CDF of the forecast to match the most recently
            # observed precipitation field
            R_c_ = probmatching.nonparam_match_empirical_cdf(R_c_, R)
        elif probmatching_method == "mean":
            mu_fct = np.mean(R_c_[MASK])
            R_c_[MASK] = R_c_[MASK] - mu_fct + mu_0

        # advect the recomposed precipitation field to obtain the forecast for
        # time step t
        extrap_kwargs.update({"D_prev": D, "return_displacement": True})
        R_f_, D_ = extrapolator_method(R_c_, V, 1, **extrap_kwargs)
        D = D_
        R_f_ = R_f_[0]
        R_f.append(R_f_)

        if measure_time:
            print("%.2f seconds." % (time.time() - starttime))
        else:
            print("done.")

    if measure_time:
        mainloop_time = time.time() - starttime_mainloop

    R_f = np.stack(R_f)

    if measure_time:
        return R_f, init_time, mainloop_time
    else:
        return R_f


def _check_inputs(R, V, ar_order):
    if len(R.shape) != 3:
        raise ValueError("R must be a three-dimensional array")
    if R.shape[0] < ar_order + 1:
        raise ValueError("R.shape[0] < ar_order+1")
    if len(V.shape) != 3:
        raise ValueError("V must be a three-dimensional array")
    if R.shape[1:3] != V.shape[1:3]:
        raise ValueError("dimension mismatch between R and V: shape(R)=%s, shape(V)=%s" % \
                         (str(R.shape), str(V.shape)))


def _compute_sprog_mask(R, war):
    # obtain the CDF from the non-perturbed forecast that is
    # scale-filtered by the AR(p) model
    R_s = R.flatten()

    # compute the threshold value R_pct_thr corresponding to the
    # same fraction of precipitation pixels (forecast values above
    # R_thr) as in the most recently observed precipitation field
    R_s.sort(kind="quicksort")
    x = 1.0 * np.arange(1, len(R_s) + 1)[::-1] / len(R_s)
    i = np.argmin(abs(x - war))
    # handle ties
    if R_s[i] == R_s[i + 1]:
        i = np.where(R_s == R_s[i])[0][-1] + 1
    R_pct_thr = R_s[i]

    # determine a mask using the above threshold value to preserve the
    # wet-area ratio
    return R >= R_pct_thr
