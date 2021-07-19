# -*- coding: utf-8 -*-
"""
pysteps.nowcasts.sseps
======================

Implementation of the Short-space ensemble prediction system (SSEPS) method.
Essentially, SSEPS is a localized version of STEPS.

For localization we intend the use of a subset of the observations in order to
estimate model parameters that are distributed in space. The short-space
approach used in :cite:`NBSG2017` is generalized to the whole nowcasting system.
This essentially boils down to a moving window localization of the nowcasting
procedure, whereby all parameters are estimated over a subdomain of prescribed
size.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import numpy as np
import scipy.ndimage
import time

from .. import cascade
from .. import extrapolation
from .. import noise
from ..nowcasts import utils as nowcast_utils
from ..postprocessing import probmatching
from ..timeseries import autoregression, correlation

try:
    import dask

    dask_imported = True
except ImportError:
    dask_imported = False


def forecast(
    R,
    metadata,
    V,
    timesteps,
    n_ens_members=24,
    n_cascade_levels=6,
    win_size=256,
    overlap=0.1,
    war_thr=0.1,
    extrap_method="semilagrangian",
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    noise_method="ssft",
    ar_order=2,
    vel_pert_method=None,
    probmatching_method="cdf",
    mask_method="incremental",
    callback=None,
    fft_method="numpy",
    return_output=True,
    seed=None,
    num_workers=1,
    extrap_kwargs=None,
    filter_kwargs=None,
    noise_kwargs=None,
    vel_pert_kwargs=None,
    mask_kwargs=None,
    measure_time=False,
):
    """
    Generate a nowcast ensemble by using the Short-space ensemble prediction
    system (SSEPS) method.
    This is an experimental version of STEPS which allows for localization
    by means of a window function.

    Parameters
    ----------
    R: array-like
        Array of shape (ar_order+1,m,n) containing the input precipitation fields
        ordered by timestamp from oldest to newest. The time steps between the inputs
        are assumed to be regular, and the inputs are required to have finite values.
    metadata: dict
        Metadata dictionary containing the accutime, xpixelsize, threshold and
        zerovalue attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`. xpixelsize is assumed to be in meters.
    V: array-like
        Array of shape (2,m,n) containing the x- and y-components of the advection
        field. The velocities are assumed to represent one time step between the
        inputs. All values are required to be finite.
    win_size: int or two-element sequence of ints
        Size-length of the localization window.
    overlap: float [0,1[
        A float between 0 and 1 prescribing the level of overlap between
        successive windows. If set to 0, no overlap is used.
    war_thr: float
        Threshold for the minimum fraction of rain in a given window.
    timesteps: int or list of floats
        Number of time steps to forecast or a list of time steps for which the
        forecasts are computed (relative to the input time step). The elements
        of the list are required to be in ascending order.
    n_ens_members: int
        The number of ensemble members to generate.
    n_cascade_levels: int
        The number of cascade levels to use.
    extrap_method: {'semilagrangian'}
        Name of the extrapolation method to use. See the documentation of
        pysteps.extrapolation.interface.
    decomp_method: {'fft'}
        Name of the cascade decomposition method to use. See the documentation
        of pysteps.cascade.interface.
    bandpass_filter_method: {'gaussian', 'uniform'}
        Name of the bandpass filter method to use with the cascade
        decomposition.
    noise_method: {'parametric','nonparametric','ssft','nested',None}
        Name of the noise generator to use for perturbating the precipitation
        field. See the documentation of pysteps.noise.interface. If set to None,
        no noise is generated.
    ar_order: int
        The order of the autoregressive model to use. Must be >= 1.
    vel_pert_method: {'bps',None}
        Name of the noise generator to use for perturbing the advection field.
        See the documentation of pysteps.noise.interface. If set to None,
        the advection field is not perturbed.
    mask_method: {'incremental', None}
        The method to use for masking no precipitation areas in the forecast
        field. The masked pixels are set to the minimum value of the
        observations. 'incremental' = iteratively buffer the mask with a
        certain rate (currently it is 1 km/min), None=no masking.
    probmatching_method: {'cdf', None}
        Method for matching the statistics of the forecast field with those of
        the most recently observed one. 'cdf'=map the forecast CDF to the
        observed one, None=no matching applied. Using 'mean' requires
        that mask_method is not None.
    callback: function
        Optional function that is called after computation of each time step of
        the nowcast. The function takes one argument: a three-dimensional array
        of shape (n_ens_members,h,w), where h and w are the height and width
        of the input field R, respectively. This can be used, for instance,
        writing the outputs into files.
    return_output: bool
        Set to False to disable returning the outputs as numpy arrays. This can
        save memory if the intermediate results are written to output files
        using the callback function.
    seed: int
        Optional seed number for the random generators.
    num_workers: int
        The number of workers to use for parallel computation. Applicable if
        dask is enabled or pyFFTW is used for computing the FFT.
        When num_workers>1, it is advisable to disable OpenMP by setting the
        environment variable OMP_NUM_THREADS to 1.
        This avoids slowdown caused by too many simultaneous threads.
    fft_method: str
        A string defining the FFT method to use (see utils.fft.get_method).
        Defaults to 'numpy' for compatibility reasons. If pyFFTW is installed,
        the recommended method is 'pyfftw'.
    extrap_kwargs: dict
        Optional dictionary containing keyword arguments for the extrapolation
        method. See the documentation of pysteps.extrapolation.
    filter_kwargs: dict
        Optional dictionary containing keyword arguments for the filter method.
        See the documentation of pysteps.cascade.bandpass_filters.py.
    noise_kwargs: dict
        Optional dictionary containing keyword arguments for the initializer of
        the noise generator. See the documentation of
        pysteps.noise.fftgenerators.
    vel_pert_kwargs: dict
        Optional dictionary containing keyword arguments "p_pert_par" and
        "p_pert_perp" for the initializer of the velocity perturbator.
        See the documentation of pysteps.noise.motion.
    mask_kwargs: dict
        Optional dictionary containing mask keyword arguments 'mask_f' and
        'mask_rim', the factor defining the the mask increment and the rim size,
        respectively.
        The mask increment is defined as mask_f*timestep/kmperpixel.
    measure_time: bool
        If set to True, measure, print and return the computation time.

    Returns
    -------
    out: ndarray
        If return_output is True, a four-dimensional array of shape
        (n_ens_members,num_timesteps,m,n) containing a time series of forecast
        precipitation fields for each ensemble member. Otherwise, a None value
        is returned. The time series starts from t0+timestep, where timestep is
        taken from the input precipitation fields R.

    See also
    --------
    pysteps.extrapolation.interface, pysteps.cascade.interface,
    pysteps.noise.interface, pysteps.noise.utils.compute_noise_stddev_adjs

    Notes
    -----
    Please be aware that this represents a (very) experimental implementation.

    References
    ----------
    :cite:`Seed2003`, :cite:`BPS2006`, :cite:`SPN2013`, :cite:`NBSG2017`
    """

    _check_inputs(R, V, timesteps, ar_order)

    if extrap_kwargs is None:
        extrap_kwargs = dict()
    else:
        extrap_kwargs = extrap_kwargs.copy()

    if filter_kwargs is None:
        filter_kwargs = dict()

    if noise_kwargs is None:
        noise_kwargs = dict()

    if vel_pert_kwargs is None:
        vel_pert_kwargs = dict()

    if mask_kwargs is None:
        mask_kwargs = dict()

    if np.any(~np.isfinite(R)):
        raise ValueError("R contains non-finite values")

    if np.any(~np.isfinite(V)):
        raise ValueError("V contains non-finite values")

    if mask_method not in ["incremental", None]:
        raise ValueError(
            "unknown mask method %s: must be 'incremental' or None" % mask_method
        )

    if np.isscalar(win_size):
        win_size = (int(win_size), int(win_size))
    else:
        win_size = tuple([int(win_size[i]) for i in range(2)])

    timestep = metadata["accutime"]
    kmperpixel = metadata["xpixelsize"] / 1000

    print("Computing SSEPS nowcast:")
    print("------------------------")
    print("")

    print("Inputs:")
    print("-------")
    print("input dimensions: %dx%d" % (R.shape[1], R.shape[2]))
    print("km/pixel:         %g" % kmperpixel)
    print("time step:        %d minutes" % timestep)
    print("")

    print("Methods:")
    print("--------")
    print("extrapolation:          %s" % extrap_method)
    print("bandpass filter:        %s" % bandpass_filter_method)
    print("decomposition:          %s" % decomp_method)
    print("noise generator:        %s" % noise_method)
    print("velocity perturbator:   %s" % vel_pert_method)
    print("precip. mask method:    %s" % mask_method)
    print("probability matching:   %s" % probmatching_method)
    print("FFT method:             %s" % fft_method)
    print("")

    print("Parameters:")
    print("-----------")
    print("localization window:      %dx%d" % (win_size[0], win_size[1]))
    print("overlap:                  %.1f" % overlap)
    print("war thr:                  %.2f" % war_thr)
    if isinstance(timesteps, int):
        print("number of time steps:     %d" % timesteps)
    else:
        print("time steps:               %s" % timesteps)
    print("ensemble size:            %d" % n_ens_members)
    print("number of cascade levels: %d" % n_cascade_levels)
    print("order of the AR(p) model: %d" % ar_order)
    print("dask imported:            %s" % ("yes" if dask_imported else "no"))
    print("num workers:              %d" % num_workers)

    if vel_pert_method == "bps":
        vp_par = vel_pert_kwargs.get(
            "p_pert_par", noise.motion.get_default_params_bps_par()
        )
        vp_perp = vel_pert_kwargs.get(
            "p_pert_perp", noise.motion.get_default_params_bps_perp()
        )
        print(
            "velocity perturbations, parallel:      %g,%g,%g"
            % (vp_par[0], vp_par[1], vp_par[2])
        )
        print(
            "velocity perturbations, perpendicular: %g,%g,%g"
            % (vp_perp[0], vp_perp[1], vp_perp[2])
        )

    R_thr = metadata["threshold"]
    R_min = metadata["zerovalue"]

    num_ensemble_workers = n_ens_members if num_workers > n_ens_members else num_workers

    if measure_time:
        starttime_init = time.time()

    # get methods
    extrapolator_method = extrapolation.get_method(extrap_method)

    x_values, y_values = np.meshgrid(np.arange(R.shape[2]), np.arange(R.shape[1]))

    xy_coords = np.stack([x_values, y_values])

    decomp_method, __ = cascade.get_method(decomp_method)
    filter_method = cascade.get_method(bandpass_filter_method)
    if noise_method is not None:
        init_noise, generate_noise = noise.get_method(noise_method)

    # advect the previous precipitation fields to the same position with the
    # most recent one (i.e. transform them into the Lagrangian coordinates)
    R = R[-(ar_order + 1) :, :, :].copy()
    extrap_kwargs = extrap_kwargs.copy()
    extrap_kwargs["xy_coords"] = xy_coords
    res = []
    f = lambda R, i: extrapolator_method(
        R[i, :, :], V, ar_order - i, "min", **extrap_kwargs
    )[-1]
    for i in range(ar_order):
        if not dask_imported:
            R[i, :, :] = f(R, i)
        else:
            res.append(dask.delayed(f)(R, i))

    if dask_imported:
        num_workers_ = len(res) if num_workers > len(res) else num_workers
        R = np.stack(list(dask.compute(*res, num_workers=num_workers_)) + [R[-1, :, :]])

    if mask_method == "incremental":
        # get mask parameters
        mask_rim = mask_kwargs.get("mask_rim", 10)
        mask_f = mask_kwargs.get("mask_f", 1.0)
        # initialize the structuring element
        struct = scipy.ndimage.generate_binary_structure(2, 1)
        # iterate it to expand it nxn
        n = mask_f * timestep / kmperpixel
        struct = scipy.ndimage.iterate_structure(struct, int((n - 1) / 2.0))

    noise_kwargs.update(
        {
            "win_size": win_size,
            "overlap": overlap,
            "war_thr": war_thr,
            "rm_rdisc": True,
            "donorm": True,
        }
    )

    print("Estimating nowcast parameters...", end="")

    def estimator(R, parsglob=None, idxm=None, idxn=None):

        pars = {}

        # initialize the perturbation generator for the precipitation field
        if noise_method is not None and parsglob is None:
            P = init_noise(R, fft_method=fft_method, **noise_kwargs)
        else:
            P = None
        pars["P"] = P

        # initialize the band-pass filter
        if parsglob is None:
            filter = filter_method(R.shape[1:], n_cascade_levels, **filter_kwargs)
            pars["filter"] = filter
        else:
            pars["filter"] = None

        # compute the cascade decompositions of the input precipitation fields
        if parsglob is None:
            R_d = []
            for i in range(ar_order + 1):
                R_d_ = decomp_method(
                    R[i, :, :],
                    filter,
                    fft_method=fft_method,
                    normalize=True,
                    compute_stats=True,
                )
                R_d.append(R_d_)
            R_d_ = None

        # normalize the cascades and rearrange them into a four-dimensional array
        # of shape (n_cascade_levels,ar_order+1,m,n) for the autoregressive model
        if parsglob is None:
            R_c = nowcast_utils.stack_cascades(R_d, n_cascade_levels)
            mu = R_d[-1]["means"]
            sigma = R_d[-1]["stds"]
            R_d = None

        else:
            R_c = parsglob["R_c"][0][
                :, :, idxm.item(0) : idxm.item(1), idxn.item(0) : idxn.item(1)
            ].copy()
            mu = np.mean(R_c, axis=(2, 3))
            sigma = np.std(R_c, axis=(2, 3))

            R_c = (R_c - mu[:, :, None, None]) / sigma[:, :, None, None]

            mu = mu[:, -1]
            sigma = sigma[:, -1]

        pars["mu"] = mu
        pars["sigma"] = sigma

        # compute lag-l temporal autocorrelation coefficients for each cascade level
        GAMMA = np.empty((n_cascade_levels, ar_order))
        for i in range(n_cascade_levels):
            R_c_ = np.stack([R_c[i, j, :, :] for j in range(ar_order + 1)])
            GAMMA[i, :] = correlation.temporal_autocorrelation(R_c_)
        R_c_ = None

        if ar_order == 2:
            # adjust the local lag-2 correlation coefficient to ensure that the AR(p)
            # process is stationary
            for i in range(n_cascade_levels):
                GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef2(
                    GAMMA[i, 0], GAMMA[i, 1]
                )

        # estimate the parameters of the AR(p) model from the autocorrelation
        # coefficients
        PHI = np.empty((n_cascade_levels, ar_order + 1))
        for i in range(n_cascade_levels):
            PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])
        pars["PHI"] = PHI

        # stack the cascades into a five-dimensional array containing all ensemble
        # members
        R_c = [R_c.copy() for i in range(n_ens_members)]
        pars["R_c"] = R_c

        if mask_method is not None and parsglob is None:
            MASK_prec = R[-1, :, :] >= R_thr
            if mask_method == "incremental":
                # initialize precip mask for each member
                MASK_prec = _compute_incremental_mask(MASK_prec, struct, mask_rim)
                MASK_prec = [MASK_prec.copy() for j in range(n_ens_members)]
        else:
            MASK_prec = None
        pars["MASK_prec"] = MASK_prec

        return pars

    # prepare windows
    M, N = R.shape[1:]
    n_windows_M = np.ceil(1.0 * M / win_size[0]).astype(int)
    n_windows_N = np.ceil(1.0 * N / win_size[1]).astype(int)
    idxm = np.zeros((2, 1), dtype=int)
    idxn = np.zeros((2, 1), dtype=int)

    if measure_time:
        starttime = time.time()

    # compute global parameters to be used as defaults
    parsglob = estimator(R)

    # loop windows
    if n_windows_M > 1 or n_windows_N > 1:
        war = np.empty((n_windows_M, n_windows_N))
        PHI = np.empty((n_windows_M, n_windows_N, n_cascade_levels, ar_order + 1))
        mu = np.empty((n_windows_M, n_windows_N, n_cascade_levels))
        sigma = np.empty((n_windows_M, n_windows_N, n_cascade_levels))
        ff = []
        rc = []
        pp = []
        mm = []
        for m in range(n_windows_M):
            ff_ = []
            pp_ = []
            rc_ = []
            mm_ = []
            for n in range(n_windows_N):

                # compute indices of local window
                idxm[0] = int(np.max((m * win_size[0] - overlap * win_size[0], 0)))
                idxm[1] = int(
                    np.min((idxm[0] + win_size[0] + overlap * win_size[0], M))
                )
                idxn[0] = int(np.max((n * win_size[1] - overlap * win_size[1], 0)))
                idxn[1] = int(
                    np.min((idxn[0] + win_size[1] + overlap * win_size[1], N))
                )

                mask = np.zeros((M, N), dtype=bool)
                mask[idxm.item(0) : idxm.item(1), idxn.item(0) : idxn.item(1)] = True

                R_ = R[:, idxm.item(0) : idxm.item(1), idxn.item(0) : idxn.item(1)]

                war[m, n] = np.sum(R_[-1, :, :] >= R_thr) / R_[-1, :, :].size
                if war[m, n] > war_thr:

                    # estimate local parameters
                    pars = estimator(R, parsglob, idxm, idxn)
                    ff_.append(pars["filter"])
                    pp_.append(pars["P"])
                    rc_.append(pars["R_c"])
                    mm_.append(pars["MASK_prec"])
                    mu[m, n, :] = pars["mu"]
                    sigma[m, n, :] = pars["sigma"]
                    PHI[m, n, :, :] = pars["PHI"]

                else:
                    # dry window
                    ff_.append(None)
                    pp_.append(None)
                    rc_.append(None)
                    mm_.append(None)

            ff.append(ff_)
            pp.append(pp_)
            rc.append(rc_)
            mm.append(mm_)

        # remove unnecessary variables
        ff_ = None
        pp_ = None
        rc_ = None
        mm_ = None
        pars = None

    if measure_time:
        print("%.2f seconds." % (time.time() - starttime))
    else:
        print(" done.")

    # initialize the random generators
    if noise_method is not None:
        randgen_prec = []
        randgen_motion = []
        np.random.seed(seed)
        for j in range(n_ens_members):
            rs = np.random.RandomState(seed)
            randgen_prec.append(rs)
            seed = rs.randint(0, high=1e9)
            rs = np.random.RandomState(seed)
            randgen_motion.append(rs)
            seed = rs.randint(0, high=1e9)

    if vel_pert_method is not None:
        init_vel_noise, generate_vel_noise = noise.get_method(vel_pert_method)

        # initialize the perturbation generators for the motion field
        vps = []
        for j in range(n_ens_members):
            kwargs = {
                "randstate": randgen_motion[j],
                "p_par": vp_par,
                "p_perp": vp_perp,
            }
            vp_ = init_vel_noise(V, 1.0 / kmperpixel, timestep, **kwargs)
            vps.append(vp_)

    D = [None for j in range(n_ens_members)]
    R_f = [[] for j in range(n_ens_members)]

    if measure_time:
        init_time = time.time() - starttime_init

    R = R[-1, :, :]

    print("Starting nowcast computation.")

    if measure_time:
        starttime_mainloop = time.time()

    if isinstance(timesteps, int):
        timesteps = range(timesteps + 1)
        timestep_type = "int"
    else:
        original_timesteps = [0] + list(timesteps)
        timesteps = nowcast_utils.binned_timesteps(original_timesteps)
        timestep_type = "list"

    extrap_kwargs["return_displacement"] = True
    R_f_prev = [R for i in range(n_ens_members)]
    t_prev = [0.0 for j in range(n_ens_members)]
    t_total = [0.0 for j in range(n_ens_members)]

    # iterate each time step
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

        # iterate each ensemble member
        def worker(j):

            # first the global step

            if noise_method is not None:
                # generate noise field
                EPS = generate_noise(
                    parsglob["P"], randstate=randgen_prec[j], fft_method=fft_method
                )
                # decompose the noise field into a cascade
                EPS_d = decomp_method(
                    EPS,
                    parsglob["filter"],
                    fft_method=fft_method,
                    normalize=True,
                    compute_stats=True,
                )
            else:
                EPS_d = None

            # iterate the AR(p) model for each cascade level
            R_c = parsglob["R_c"][j].copy()
            if R_c.shape[1] >= ar_order:
                R_c = R_c[:, -ar_order:, :, :].copy()
            for i in range(n_cascade_levels):
                # normalize the noise cascade
                if EPS_d is not None:
                    EPS_ = (
                        EPS_d["cascade_levels"][i, :, :] - EPS_d["means"][i]
                    ) / EPS_d["stds"][i]
                else:
                    EPS_ = None
                # apply AR(p) process to cascade level
                R_c[i, :, :, :] = autoregression.iterate_ar_model(
                    R_c[i, :, :, :], parsglob["PHI"][i, :], eps=EPS_
                )
                EPS_ = None
            parsglob["R_c"][j] = R_c.copy()
            EPS = None

            # compute the recomposed precipitation field(s) from the cascades
            # obtained from the AR(p) model(s)
            R_f_new = _recompose_cascade(R_c, parsglob["mu"], parsglob["sigma"])
            R_c = None

            # then the local steps
            if n_windows_M > 1 or n_windows_N > 1:
                idxm = np.zeros((2, 1), dtype=int)
                idxn = np.zeros((2, 1), dtype=int)
                R_l = np.zeros((M, N), dtype=float)
                M_s = np.zeros((M, N), dtype=float)
                for m in range(n_windows_M):
                    for n in range(n_windows_N):

                        # compute indices of local window
                        idxm[0] = int(
                            np.max((m * win_size[0] - overlap * win_size[0], 0))
                        )
                        idxm[1] = int(
                            np.min((idxm[0] + win_size[0] + overlap * win_size[0], M))
                        )
                        idxn[0] = int(
                            np.max((n * win_size[1] - overlap * win_size[1], 0))
                        )
                        idxn[1] = int(
                            np.min((idxn[0] + win_size[1] + overlap * win_size[1], N))
                        )

                        # build localization mask
                        mask = _get_mask((M, N), idxm, idxn)
                        mask_l = mask[
                            idxm.item(0) : idxm.item(1), idxn.item(0) : idxn.item(1)
                        ]
                        M_s += mask

                        # skip if dry
                        if war[m, n] > war_thr:

                            R_c = rc[m][n][j].copy()
                            if R_c.shape[1] >= ar_order:
                                R_c = R_c[:, -ar_order:, :, :]
                            if noise_method is not None:
                                # extract noise field
                                EPS_d_l = EPS_d["cascade_levels"][
                                    :,
                                    idxm.item(0) : idxm.item(1),
                                    idxn.item(0) : idxn.item(1),
                                ].copy()
                                mu_ = np.mean(EPS_d_l, axis=(1, 2))
                                sigma_ = np.std(EPS_d_l, axis=(1, 2))
                            else:
                                EPS_d_l = None

                            # iterate the AR(p) model for each cascade level
                            for i in range(n_cascade_levels):
                                # normalize the noise cascade
                                if EPS_d_l is not None:
                                    EPS_ = (
                                        EPS_d_l[i, :, :] - mu_[i, None, None]
                                    ) / sigma_[i, None, None]
                                else:
                                    EPS_ = None
                                # apply AR(p) process to cascade level
                                R_c[i, :, :, :] = autoregression.iterate_ar_model(
                                    R_c[i, :, :, :], PHI[m, n, i, :], eps=EPS_
                                )
                                EPS_ = None
                            rc[m][n][j] = R_c.copy()
                            EPS_d_l = mu_ = sigma_ = None

                            # compute the recomposed precipitation field(s) from the cascades
                            # obtained from the AR(p) model(s)
                            mu_ = mu[m, n, :]
                            sigma_ = sigma[m, n, :]
                            R_c = [
                                ((R_c[i, -1, :, :] * sigma_[i]) + mu_[i])
                                * parsglob["sigma"][i]
                                + parsglob["mu"][i]
                                for i in range(len(mu_))
                            ]
                            R_l_ = np.sum(np.stack(R_c), axis=0)
                            R_c = mu_ = sigma_ = None
                            # R_l_ = _recompose_cascade(R_c[:, :, :], mu[m, n, :], sigma[m, n, :])
                        else:
                            R_l_ = R_f_new[
                                idxm.item(0) : idxm.item(1), idxn.item(0) : idxn.item(1)
                            ].copy()

                        if probmatching_method == "cdf":
                            # adjust the CDF of the forecast to match the most recently
                            # observed precipitation field
                            R_ = R[
                                idxm.item(0) : idxm.item(1), idxn.item(0) : idxn.item(1)
                            ].copy()
                            R_l_ = probmatching.nonparam_match_empirical_cdf(R_l_, R_)
                            R_ = None

                        R_l[
                            idxm.item(0) : idxm.item(1), idxn.item(0) : idxn.item(1)
                        ] += (R_l_ * mask_l)
                        R_l_ = None

                ind = M_s > 0
                R_l[ind] *= 1 / M_s[ind]
                R_l[~ind] = R_min

                R_f_new = R_l.copy()
                R_l = None

            if probmatching_method == "cdf":
                # adjust the CDF of the forecast to match the most recently
                # observed precipitation field
                R_f_new[R_f_new < R_thr] = R_min
                R_f_new = probmatching.nonparam_match_empirical_cdf(R_f_new, R)

            if mask_method is not None:
                # apply the precipitation mask to prevent generation of new
                # precipitation into areas where it was not originally
                # observed
                if mask_method == "incremental":
                    MASK_prec = parsglob["MASK_prec"][j].copy()
                    R_f_new = R_f_new.min() + (R_f_new - R_f_new.min()) * MASK_prec
                    MASK_prec = None

            if mask_method == "incremental":
                parsglob["MASK_prec"][j] = _compute_incremental_mask(
                    R_f_new >= R_thr, struct, mask_rim
                )

            R_f_out = []
            extrap_kwargs_ = extrap_kwargs.copy()
            extrap_kwargs_["xy_coords"] = xy_coords
            extrap_kwargs_["return_displacement"] = True

            V_pert = V

            # advect the recomposed precipitation field to obtain the forecast for
            # the current time step (or subtimesteps if non-integer time steps are
            # given)
            for t_sub in subtimesteps:
                if t_sub > 0:
                    t_diff_prev_int = t_sub - int(t_sub)
                    if t_diff_prev_int > 0.0:
                        R_f_ip = (1.0 - t_diff_prev_int) * R_f_prev[
                            j
                        ] + t_diff_prev_int * R_f_new
                    else:
                        R_f_ip = R_f_prev[j]

                    t_diff_prev = t_sub - t_prev[j]
                    t_total[j] += t_diff_prev

                    # compute the perturbed motion field
                    if vel_pert_method is not None:
                        V_pert = V + generate_vel_noise(vps[j], t_total[j] * timestep)

                    extrap_kwargs_["displacement_prev"] = D[j]
                    R_f_ep, D[j] = extrapolator_method(
                        R_f_ip,
                        V_pert,
                        [t_diff_prev],
                        **extrap_kwargs_,
                    )
                    R_f_ep[0][R_f_ep[0] < R_thr] = R_min
                    R_f_out.append(R_f_ep[0])
                    t_prev[j] = t_sub

            # advect the forecast field by one time step if no subtimesteps in the
            # current interval were found
            if not subtimesteps:
                t_diff_prev = t + 1 - t_prev[j]
                t_total[j] += t_diff_prev

                # compute the perturbed motion field
                if vel_pert_method is not None:
                    V_pert = V + generate_vel_noise(vps[j], t_total[j] * timestep)

                extrap_kwargs_["displacement_prev"] = D[j]
                _, D[j] = extrapolator_method(
                    None,
                    V_pert,
                    [t_diff_prev],
                    **extrap_kwargs_,
                )
                t_prev[j] = t + 1

            R_f_prev[j] = R_f_new

            return R_f_out

        res = []
        for j in range(n_ens_members):
            if not dask_imported or n_ens_members == 1:
                res.append(worker(j))
            else:
                res.append(dask.delayed(worker)(j))

        R_f_ = (
            dask.compute(*res, num_workers=num_ensemble_workers)
            if dask_imported and n_ens_members > 1
            else res
        )
        res = None

        if is_nowcast_time_step:
            if measure_time:
                print("%.2f seconds." % (time.time() - starttime))
            else:
                print("done.")

        if callback is not None:
            R_f_stacked = np.stack(R_f_)
            if R_f_stacked.shape[1] > 0:
                callback(R_f_stacked.squeeze())
            R_f_ = None

        if return_output:
            for j in range(n_ens_members):
                R_f[j].extend(R_f_[j])

    if measure_time:
        mainloop_time = time.time() - starttime_mainloop

    if return_output:
        outarr = np.stack([np.stack(R_f[j]) for j in range(n_ens_members)])
        if measure_time:
            return outarr, init_time, mainloop_time
        else:
            return outarr
    else:
        return None


def _check_inputs(R, V, timesteps, ar_order):
    if R.ndim != 3:
        raise ValueError("R must be a three-dimensional array")
    if R.shape[0] < ar_order + 1:
        raise ValueError("R.shape[0] < ar_order+1")
    if V.ndim != 3:
        raise ValueError("V must be a three-dimensional array")
    if R.shape[1:3] != V.shape[1:3]:
        raise ValueError(
            "dimension mismatch between R and V: shape(R)=%s, shape(V)=%s"
            % (str(R.shape), str(V.shape))
        )
    if isinstance(timesteps, list) and not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")


def _compute_incremental_mask(Rbin, kr, r):
    # buffer the observation mask Rbin using the kernel kr
    # add a grayscale rim r (for smooth rain/no-rain transition)

    # buffer observation mask
    Rbin = np.ndarray.astype(Rbin.copy(), "uint8")
    Rd = scipy.ndimage.morphology.binary_dilation(Rbin, kr)

    # add grayscale rim
    kr1 = scipy.ndimage.generate_binary_structure(2, 1)
    mask = Rd.astype(int)
    for n in range(r):
        Rd = scipy.ndimage.morphology.binary_dilation(Rd, kr1)
        mask += Rd
    # normalize between 0 and 1
    return mask / mask.max()


# TODO: Use the recomponse_cascade method in the cascade.decomposition module
def _recompose_cascade(R, mu, sigma):
    R_rc = [(R[i, -1, :, :] * sigma[i]) + mu[i] for i in range(len(mu))]
    R_rc = np.sum(np.stack(R_rc), axis=0)

    return R_rc


def _build_2D_tapering_function(win_size, win_type="flat-hanning"):
    """Produces two-dimensional tapering function for rectangular fields.

    Parameters
    ----------
    win_size: tuple of int
        Size of the tapering window as two-element tuple of integers.
    win_type: str
        Name of the tapering window type (hanning, flat-hanning)

    Returns
    -------
    w2d: array-like
        A two-dimensional numpy array containing the 2D tapering function.
    """

    if len(win_size) != 2:
        raise ValueError("win_size is not a two-element tuple")

    if win_type == "hanning":
        w1dr = np.hanning(win_size[0])
        w1dc = np.hanning(win_size[1])

    elif win_type == "flat-hanning":

        T = win_size[0] / 4.0
        W = win_size[0] / 2.0
        B = np.linspace(-W, W, int(2 * W))
        R = np.abs(B) - T
        R[R < 0] = 0.0
        A = 0.5 * (1.0 + np.cos(np.pi * R / T))
        A[np.abs(B) > (2 * T)] = 0.0
        w1dr = A

        T = win_size[1] / 4.0
        W = win_size[1] / 2.0
        B = np.linspace(-W, W, int(2 * W))
        R = np.abs(B) - T
        R[R < 0] = 0.0
        A = 0.5 * (1.0 + np.cos(np.pi * R / T))
        A[np.abs(B) > (2 * T)] = 0.0
        w1dc = A

    elif win_type == "rectangular":

        w1dr = np.ones(win_size[0])
        w1dc = np.ones(win_size[1])

    else:
        raise ValueError("unknown win_type %s" % win_type)

    # Expand to 2-D
    # w2d = np.sqrt(np.outer(w1dr,w1dc))
    w2d = np.outer(w1dr, w1dc)

    # Set nans to zero
    if np.any(np.isnan(w2d)):
        w2d[np.isnan(w2d)] = np.min(w2d[w2d > 0])

    w2d[w2d < 1e-3] = 1e-3

    return w2d


def _get_mask(Size, idxi, idxj, win_type="flat-hanning"):
    """Compute a mask of zeros with a window at a given position."""

    idxi = np.array(idxi).astype(int)
    idxj = np.array(idxj).astype(int)

    win_size = (idxi[1] - idxi[0], idxj[1] - idxj[0])
    wind = _build_2D_tapering_function(win_size, win_type)

    mask = np.zeros(Size)
    mask[idxi.item(0) : idxi.item(1), idxj.item(0) : idxj.item(1)] = wind

    return mask
