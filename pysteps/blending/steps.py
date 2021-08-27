# -*- coding: utf-8 -*-
"""
pysteps.blending.steps
======================

Implementation of the STEPS stochastic blending method as described in
:cite:`BPS2006`. The method assumes the presence of one NWP model or ensemble
member to be blended with one nowcast. More models, such as in :cite:`SPN2013`
is possible with this code, but we recommend the use of just two models.

.. autosummary::
    :toctree: ../generated/

    forecast
    calculate_ratios
    calculate_weights
    blend_means_sigmas
    _check_inputs
    _compute_incremental_mask

References
----------
:cite:`BPS2004`
:cite:`BPS2006`
:cite:`SPN2013`
"""

import numpy as np
import scipy.ndimage
import time

from pysteps import cascade
from pysteps import extrapolation
from pysteps import noise
from pysteps import utils
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.postprocessing import probmatching
from pysteps.timeseries import autoregression, correlation
from pysteps import blending

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


def forecast(
    R,
    R_d_models,
    V,
    V_models,
    timesteps,
    timestep,
    n_ens_members=24,
    n_cascade_levels=8,
    blend_nwp_members=False,
    R_thr=None,
    kmperpixel=None,
    extrap_method="semilagrangian",
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    noise_method="nonparametric",
    noise_stddev_adj=None,
    ar_order=2,
    vel_pert_method="bps",
    conditional=False,
    probmatching_method="cdf",
    mask_method="incremental",
    callback=None,
    return_output=True,
    seed=None,
    num_workers=1,
    fft_method="numpy",
    domain="spatial",
    extrap_kwargs=None,
    filter_kwargs=None,
    noise_kwargs=None,
    vel_pert_kwargs=None,
    clim_kwargs=None,
    mask_kwargs=None,
    measure_time=False,
):
    """Generate a blended nowcast ensemble by using the Short-Term Ensemble
    Prediction System (STEPS) method.

    Parameters
    ----------
    R: array-like
      Array of shape (ar_order+1,m,n) containing the input precipitation fields
      ordered by timestamp from oldest to newest. The time steps between the
      inputs are assumed to be regular.
    R_d_models: array-like
        Array of shape (n_models,timesteps+1) containing, per timestep (t=0 to
        lead time here) and per (NWP) model or model ensemble member, a
        dictionary with a list of cascades obtained by calling a method
        implemented in pysteps.cascade.decomposition. In case of one
        (deterministic) model as input, add an extra dimension to make sure
        R_models is five dimensional prior to calling this function.
    V: array-like
      Array of shape (2,m,n) containing the x- and y-components of the advection
      field. The velocities are assumed to represent one time step between the
      inputs. All values are required to be finite.
    V_models: array-like
      Array of shape (n_models,timestep,2,m,n) containing the x- and y-components
      of the advection field for the (NWP) model field per forecast lead time.
      All values are required to be finite.
    timesteps: int or list of floats
      Number of time steps to forecast or a list of time steps for which the
      forecasts are computed (relative to the input time step). The elements of
      the list are required to be in ascending order.
    timestep: float
      Time step of the motion vectors (minutes). Required if vel_pert_method is
      not None or mask_method is 'incremental'.
    n_ens_members: int, optional
      The number of ensemble members to generate.
    n_cascade_levels: int, optional
      The number of cascade levels to use. Default set to 8 due to default
      climatological skill values on 8 levels.
    blend_nwp_members: bool
        Check if NWP models/members should be used individually, or if all of
        them are blended together per nowcast ensemble member. Standard set to
        false.
    R_thr: float, optional
      Specifies the threshold value for minimum observable precipitation
      intensity. Required if mask_method is not None or conditional is True.
    kmperpixel: float, optional
      Spatial resolution of the input data (kilometers/pixel). Required if
      vel_pert_method is not None or mask_method is 'incremental'.
    extrap_method: str, optional
      Name of the extrapolation method to use. See the documentation of
      pysteps.extrapolation.interface.
    decomp_method: {'fft'}, optional
      Name of the cascade decomposition method to use. See the documentation
      of pysteps.cascade.interface.
    bandpass_filter_method: {'gaussian', 'uniform'}, optional
      Name of the bandpass filter method to use with the cascade decomposition.
      See the documentation of pysteps.cascade.interface.
    noise_method: {'parametric','nonparametric','ssft','nested',None}, optional
      Name of the noise generator to use for perturbating the precipitation
      field. See the documentation of pysteps.noise.interface. If set to None,
      no noise is generated.
    noise_stddev_adj: {'auto','fixed',None}, optional
      Optional adjustment for the standard deviations of the noise fields added
      to each cascade level. This is done to compensate incorrect std. dev.
      estimates of casace levels due to presence of no-rain areas. 'auto'=use
      the method implemented in pysteps.noise.utils.compute_noise_stddev_adjs.
      'fixed'= use the formula given in :cite:`BPS2006` (eq. 6), None=disable
      noise std. dev adjustment.
    ar_order: int, optional
      The order of the autoregressive model to use. Must be >= 1.
    vel_pert_method: {'bps',None}, optional
      Name of the noise generator to use for perturbing the advection field. See
      the documentation of pysteps.noise.interface. If set to None, the advection
      field is not perturbed.
    conditional: bool, optional
      If set to True, compute the statistics of the precipitation field
      conditionally by excluding pixels where the values are below the threshold
      R_thr.
    mask_method: {'obs','incremental',None}, optional
      The method to use for masking no precipitation areas in the forecast field.
      The masked pixels are set to the minimum value of the observations.
      'obs' = apply R_thr to the most recently observed precipitation intensity
      field, 'incremental' = iteratively buffer the mask with a certain rate
      (currently it is 1 km/min), None=no masking.
    probmatching_method: {'cdf','mean',None}, optional
      Method for matching the statistics of the forecast field with those of
      the most recently observed one. 'cdf'=map the forecast CDF to the observed
      one, 'mean'=adjust only the conditional mean value of the forecast field
      in precipitation areas, None=no matching applied. Using 'mean' requires
      that mask_method is not None.
    callback: function, optional
      Optional function that is called after computation of each time step of
      the nowcast. The function takes one argument: a three-dimensional array
      of shape (n_ens_members,h,w), where h and w are the height and width
      of the input field R, respectively. This can be used, for instance,
      writing the outputs into files.
    return_output: bool, optional
      Set to False to disable returning the outputs as numpy arrays. This can
      save memory if the intermediate results are written to output files using
      the callback function.
    seed: int, optional
      Optional seed number for the random generators.
    num_workers: int, optional
      The number of workers to use for parallel computation. Applicable if dask
      is enabled or pyFFTW is used for computing the FFT. When num_workers>1, it
      is advisable to disable OpenMP by setting the environment variable
      OMP_NUM_THREADS to 1. This avoids slowdown caused by too many simultaneous
      threads.
    fft_method: str, optional
      A string defining the FFT method to use (see utils.fft.get_method).
      Defaults to 'numpy' for compatibility reasons. If pyFFTW is installed,
      the recommended method is 'pyfftw'.
    domain: {"spatial", "spectral"}
      If "spatial", all computations are done in the spatial domain (the
      classical STEPS model). If "spectral", the AR(2) models and stochastic
      perturbations are applied directly in the spectral domain to reduce
      memory footprint and improve performance :cite:`PCH2019b`.
    extrap_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the extrapolation
      method. See the documentation of pysteps.extrapolation.
    filter_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the filter method.
      See the documentation of pysteps.cascade.bandpass_filters.py.
    noise_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the initializer of
      the noise generator. See the documentation of pysteps.noise.fftgenerators.
    vel_pert_kwargs: dict, optional
      Optional dictionary containing keyword arguments 'p_par' and 'p_perp' for
      the initializer of the velocity perturbator. The choice of the optimal
      parameters depends on the domain and the used optical flow method.

      Default parameters from :cite:`BPS2006`:
      p_par  = [10.88, 0.23, -7.68]
      p_perp = [5.76, 0.31, -2.72]

      Parameters fitted to the data (optical flow/domain):

      darts/fmi:
      p_par  = [13.71259667, 0.15658963, -16.24368207]
      p_perp = [8.26550355, 0.17820458, -9.54107834]

      darts/mch:
      p_par  = [24.27562298, 0.11297186, -27.30087471]
      p_perp = [-7.80797846e+01, -3.38641048e-02, 7.56715304e+01]

      darts/fmi+mch:
      p_par  = [16.55447057, 0.14160448, -19.24613059]
      p_perp = [14.75343395, 0.11785398, -16.26151612]

      lucaskanade/fmi:
      p_par  = [2.20837526, 0.33887032, -2.48995355]
      p_perp = [2.21722634, 0.32359621, -2.57402761]

      lucaskanade/mch:
      p_par  = [2.56338484, 0.3330941, -2.99714349]
      p_perp = [1.31204508, 0.3578426, -1.02499891]

      lucaskanade/fmi+mch:
      p_par  = [2.31970635, 0.33734287, -2.64972861]
      p_perp = [1.90769947, 0.33446594, -2.06603662]

      vet/fmi:
      p_par  = [0.25337388, 0.67542291, 11.04895538]
      p_perp = [0.02432118, 0.99613295, 7.40146505]

      vet/mch:
      p_par  = [0.5075159, 0.53895212, 7.90331791]
      p_perp = [0.68025501, 0.41761289, 4.73793581]

      vet/fmi+mch:
      p_par  = [0.29495222, 0.62429207, 8.6804131 ]
      p_perp = [0.23127377, 0.59010281, 5.98180004]

      fmi=Finland, mch=Switzerland, fmi+mch=both pooled into the same data set

      The above parameters have been fitten by using run_vel_pert_analysis.py
      and fit_vel_pert_params.py located in the scripts directory.

      See pysteps.noise.motion for additional documentation.
    clim_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the climatological
        skill file. Arguments can consist of: 'outdir_path', 'n_models'
        (the number of NWP models) and 'window_length' (the minimum number of
        days the clim file should have, otherwise the default is used).
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
      taken from the input precipitation fields R. If measure_time is True, the
      return value is a three-element tuple containing the nowcast array, the
      initialization time of the nowcast generator and the time used in the
      main loop (seconds).

    See also
    --------
    pysteps.extrapolation.interface, pysteps.cascade.interface,
    pysteps.noise.interface, pysteps.noise.utils.compute_noise_stddev_adjs

    References
    ----------
    :cite:`Seed2003`, :cite:`BPS2006`, :cite:`SPN2013`, :cite:`PCH2019b`

    Notes
    -----
    1. The blending currently does not blend the beta-parameters in the parametric
    noise method. It is recommended to use the non-parameteric noise method.

    2. If blend_nwp_members is True, the BPS2006 method for the weights is
    suboptimal. It is recommended to change this method to SPN2013 later.
    """

    # 0.1 Start with some checks
    _check_inputs(R, R_d_models, V, V_models, timesteps, ar_order)

    if extrap_kwargs is None:
        extrap_kwargs = dict()

    if filter_kwargs is None:
        filter_kwargs = dict()

    if noise_kwargs is None:
        noise_kwargs = dict()

    if vel_pert_kwargs is None:
        vel_pert_kwargs = dict()

    if clim_kwargs is None:
        clim_kwargs = dict()

    if mask_kwargs is None:
        mask_kwargs = dict()

    if np.any(~np.isfinite(V)):
        raise ValueError("V contains non-finite values")

    if mask_method not in ["obs", "incremental", None]:
        raise ValueError(
            "unknown mask method %s: must be 'obs', 'incremental' or None" % mask_method
        )

    if conditional and R_thr is None:
        raise ValueError("conditional=True but R_thr is not set")

    if mask_method is not None and R_thr is None:
        raise ValueError("mask_method!=None but R_thr=None")

    if noise_stddev_adj not in ["auto", "fixed", None]:
        raise ValueError(
            "unknown noise_std_dev_adj method %s: must be 'auto', 'fixed', or None"
            % noise_stddev_adj
        )

    if kmperpixel is None:
        if vel_pert_method is not None:
            raise ValueError("vel_pert_method is set but kmperpixel=None")
        if mask_method == "incremental":
            raise ValueError("mask_method='incremental' but kmperpixel=None")

    if timestep is None:
        if vel_pert_method is not None:
            raise ValueError("vel_pert_method is set but timestep=None")
        if mask_method == "incremental":
            raise ValueError("mask_method='incremental' but timestep=None")

    # 0.2 Log some settings
    print("Computing STEPS nowcast:")
    print("------------------------")
    print("")

    print("Inputs:")
    print("-------")
    print("input dimensions: %dx%d" % (R.shape[1], R.shape[2]))
    if kmperpixel is not None:
        print("km/pixel:         %g" % kmperpixel)
    if timestep is not None:
        print("time step:        %d minutes" % timestep)
    print("")

    print("Methods:")
    print("--------")
    print("extrapolation:          %s" % extrap_method)
    print("bandpass filter:        %s" % bandpass_filter_method)
    print("decomposition:          %s" % decomp_method)
    print("noise generator:        %s" % noise_method)
    print("noise adjustment:       %s" % ("yes" if noise_stddev_adj else "no"))
    print("velocity perturbator:   %s" % vel_pert_method)
    print("conditional statistics: %s" % ("yes" if conditional else "no"))
    print("precip. mask method:    %s" % mask_method)
    print("probability matching:   %s" % probmatching_method)
    print("FFT method:             %s" % fft_method)
    print("domain:                 %s" % domain)
    print("")

    print("Parameters:")
    print("-----------")
    if isinstance(timesteps, int):
        print("number of time steps:     %d" % timesteps)
    else:
        print("time steps:               %s" % timesteps)
    print("ensemble size:            %d" % n_ens_members)
    print("parallel threads:         %d" % num_workers)
    print("number of cascade levels: %d" % n_cascade_levels)
    print("order of the AR(p) model: %d" % ar_order)
    if vel_pert_method == "bps":
        vp_par = vel_pert_kwargs.get("p_par", noise.motion.get_default_params_bps_par())
        vp_perp = vel_pert_kwargs.get(
            "p_perp", noise.motion.get_default_params_bps_perp()
        )
        print(
            "velocity perturbations, parallel:      %g,%g,%g"
            % (vp_par[0], vp_par[1], vp_par[2])
        )
        print(
            "velocity perturbations, perpendicular: %g,%g,%g"
            % (vp_perp[0], vp_perp[1], vp_perp[2])
        )

    if conditional or mask_method is not None:
        print("precip. intensity threshold: %g" % R_thr)

    # 0.3 Get the methods that will be used
    num_ensemble_workers = n_ens_members if num_workers > n_ens_members else num_workers

    if measure_time:
        starttime_init = time.time()

    fft = utils.get_method(fft_method, shape=R.shape[1:], n_threads=num_workers)

    M, N = R.shape[1:]

    # initialize the band-pass filter
    filter_method = cascade.get_method(bandpass_filter_method)
    filter = filter_method((M, N), n_cascade_levels, **filter_kwargs)

    decomp_method, recomp_method = cascade.get_method(decomp_method)

    extrapolator_method = extrapolation.get_method(extrap_method)

    x_values, y_values = np.meshgrid(np.arange(R.shape[2]), np.arange(R.shape[1]))

    xy_coords = np.stack([x_values, y_values])

    R = R[-(ar_order + 1) :, :, :].copy()

    # determine the domain mask from non-finite values
    domain_mask = np.logical_or.reduce(
        [~np.isfinite(R[i, :]) for i in range(R.shape[0])]
    )

    # determine the precipitation threshold mask
    if conditional:
        MASK_thr = np.logical_and.reduce(
            [R[i, :, :] >= R_thr for i in range(R.shape[0])]
        )
    else:
        MASK_thr = None

    ###
    # 1. Start with the radar rainfall field extrapolation to get the
    # different time steps in a Lagrangian space
    ###
    # advect the previous precipitation fields to the same position with the
    # most recent one (i.e. transform them into the Lagrangian coordinates)
    extrap_kwargs = extrap_kwargs.copy()
    extrap_kwargs["xy_coords"] = xy_coords
    extrap_kwargs["allow_nonfinite_values"] = True
    res = list()

    def f(R, i):
        return extrapolator_method(R[i, :, :], V, ar_order - i, "min", **extrap_kwargs)[
            -1
        ]

    for i in range(ar_order):
        if not DASK_IMPORTED:
            R[i, :, :] = f(R, i)
        else:
            res.append(dask.delayed(f)(R, i))

    if DASK_IMPORTED:
        num_workers_ = len(res) if num_workers > len(res) else num_workers
        R = np.stack(list(dask.compute(*res, num_workers=num_workers_)) + [R[-1, :, :]])

    # replace non-finite values with the minimum value
    R = R.copy()
    for i in range(R.shape[0]):
        R[i, ~np.isfinite(R[i, :])] = np.nanmin(R[i, :])

    ###
    # 2. Initialize the noise method
    ###
    if noise_method is not None:
        # get methods for perturbations
        init_noise, generate_noise = noise.get_method(noise_method)

        # initialize the perturbation generator for the precipitation field
        pp = init_noise(R, fft_method=fft, **noise_kwargs)

        if noise_stddev_adj == "auto":
            print("Computing noise adjustment coefficients... ", end="", flush=True)
            if measure_time:
                starttime = time.time()

            R_min = np.min(R)
            noise_std_coeffs = noise.utils.compute_noise_stddev_adjs(
                R[-1, :, :],
                R_thr,
                R_min,
                filter,
                decomp_method,
                pp,
                generate_noise,
                20,
                conditional=True,
                num_workers=num_workers,
            )

            if measure_time:
                print("%.2f seconds." % (time.time() - starttime))
            else:
                print("done.")
        elif noise_stddev_adj == "fixed":
            f = lambda k: 1.0 / (0.75 + 0.09 * k)
            noise_std_coeffs = [f(k) for k in range(1, n_cascade_levels + 1)]
        else:
            noise_std_coeffs = np.ones(n_cascade_levels)

        if noise_stddev_adj is not None:
            print("noise std. dev. coeffs:   %s" % str(noise_std_coeffs))

    ###
    # 3. Perform the cascade decomposition for the input precip fields and
    # The decomposition for the (NWP) model fields is already present
    ###
    # compute the cascade decompositions of the input precipitation fields
    R_d = []
    for i in range(ar_order + 1):
        R_ = decomp_method(
            R[i, :, :],
            filter,
            mask=MASK_thr,
            fft_method=fft,
            output_domain=domain,
            normalize=True,
            compute_stats=True,
            compact_output=True,
        )
        R_d.append(R_)

    # normalize the cascades and rearrange them into a four-dimensional array
    # of shape (n_cascade_levels,ar_order+1,m,n) for the autoregressive model
    R_c = nowcast_utils.stack_cascades(R_d, n_cascade_levels)

    R_d = R_d[-1]
    mu_extrapolation = np.array(R_d["means"])
    sigma_extrapolation = np.array(R_d["stds"])
    R_d = [R_d.copy() for j in range(n_ens_members)]

    # Also stack the (NWP) model cascades in separate normalized cascades and
    # return the means and sigmas.
    # The normalized model cascade should have the shape:
    # [n_models, n_timesteps, n_cascade_levels, m, n]
    R_models = []
    mu_models = []
    sigma_models = []
    # Stack it per model and combine that
    for i in range(R_d_models.shape[0]):
        R_models_, mu_models_, sigma_models_ = [], [], []
        R_models_, mu_models_, sigma_models_ = blending.utils.stack_cascades(
            R_d=R_d_models[i, :], donorm=False
        )
        R_models.append(R_models_)
        mu_models.append(mu_models_)
        sigma_models.append(sigma_models_)

    R_models = np.stack(R_models)
    mu_models = np.stack(mu_models)
    sigma_models = np.stack(sigma_models)

    R_models_, mu_models_, sigma_models_ = None, None, None

    # Finally, recompose the (NWP) model cascades to have rainfall fields per
    # model and time step, which will be used in the probability matching steps.
    # Recomposed cascade will have shape: [n_models, n_timesteps, m, n]
    R_models_pm = []
    for i in range(R_d_models.shape[0]):
        R_models_pm_ = []
        for time_step in range(R_d_models.shape[1]):
            R_models_pm_.append(recomp_method(R_d_models[i, time_step]))
        R_models_pm.append(R_models_pm_)

    R_models_pm = np.stack(R_models_pm)
    R_models_pm_ = None

    ###
    # 4. Estimate AR parameters for the radar rainfall field
    ###
    # compute lag-l temporal autocorrelation coefficients for each cascade level
    GAMMA = np.empty((n_cascade_levels, ar_order))
    for i in range(n_cascade_levels):
        GAMMA[i, :] = correlation.temporal_autocorrelation(R_c[i], mask=MASK_thr)

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

    ###
    # 5. Before calling the worker for the forecast loop, determine which (NWP)
    # models will be combined with which nowcast ensemble members. With the
    # way it is implemented at this moment: n_ens_members of the output equals
    # the maximum number of (ensemble) members in the input
    # (either the nowcasts or NWP).
    ###
    # First, discard all except the p-1 last cascades because they are not needed
    # for the AR(p) model
    R_c = [R_c[i][-ar_order:] for i in range(n_cascade_levels)]

    # Check if NWP models/members should be used individually, or if all of
    # them are blended together per nowcast ensemble member.
    if blend_nwp_members == True:
        # stack the extrapolation cascades into a list containing all ensemble members
        R_c = [
            [R_c[j].copy() for j in range(n_cascade_levels)]
            for i in range(n_ens_members)
        ]
        R_c = np.stack(R_c)

    else:
        # Start with determining the maximum and mimimum number of members/models
        # in both input products
        n_model_members = R_models.shape[0]
        n_ens_members_max = max(n_ens_members, n_model_members)
        n_ens_members_min = min(n_ens_members, n_model_members)

        # Now, repeat the nowcast ensemble members or the nwp models/members until
        # it has the same amount of members as n_ens_members_max. For instance, if
        # you have 10 ensemble nowcasts members and 3 NWP members, the output will
        # be an ensemble of 10 members. Hence, the three NWP members are blended
        # with the first three members of the nowcast (member one with member one,
        # two with two, etc.), subsequently, the same NWP members are blended with
        # the next three members (NWP member one with member 4, NWP member 2 with
        # member 5, etc.), until 10 is reached.
        if n_ens_members_min != n_ens_members_max:
            if n_model_members == 1:
                R_models = np.repeat(R_models[:, :, :, :, :], n_ens_members_max, axis=0)
                mu_models = np.repeat(mu_models[:, :, :], n_ens_members_max, axis=0)
                sigma_models = np.repeat(
                    sigma_models[:, :, :], n_ens_members_max, axis=0
                )
                V_models = np.repeat(V_models[:, :, :, :], n_ens_members_max, axis=0)
                # For the prob. matching
                R_models_pm = np.repeat(
                    R_models_pm[:, :, :, :], n_ens_members_max, axis=0
                )

            elif n_model_members == n_ens_members_min:
                repeats = [
                    (n_ens_members_max + i) // n_ens_members_min
                    for i in range(n_ens_members_min)
                ]
                if n_model_members == n_ens_members_min:
                    R_models = np.repeat(R_models, repeats, axis=0)
                    mu_models = np.repeat(mu_models, repeats, axis=0)
                    sigma_models = np.repeat(sigma_models, repeats, axis=0)
                    V_models = np.repeat(V_models, repeats, axis=0)
                    # For the prob. matching
                    R_models_pm = np.repeat(R_models_pm, repeats, axis=0)

        R_c = [
            [R_c[j].copy() for j in range(n_cascade_levels)]
            for i in range(n_ens_members_max)
        ]
        R_c = np.stack(R_c)

        # Check if dimensions are correct
        assert (
            R_models.shape[0] == R_c.shape[0]
        ), "The number of members in the nowcast and nwp cascades need to be identical when blend_nwp_members is False: current dimension of R_models = {} and dimension of R_c = {}".format(
            R_models.shape[0], R_c.shape[0]
        )

        n_ens_members = n_ens_members_max

    # Also initialize the cascade of temporally correlated noise, which has the
    # same shape as R_c, but starts with value zero.
    Yn_c = np.zeros(R_c.shape)

    ###
    # 6. Initialize all the random generators and prepare for the forecast
    # loop
    ###
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

    D = np.stack([np.full(n_cascade_levels, None) for j in range(n_ens_members)])
    R_f = [[] for j in range(n_ens_members)]

    if mask_method == "incremental":
        # get mask parameters
        mask_rim = mask_kwargs.get("mask_rim", 10)
        mask_f = mask_kwargs.get("mask_f", 1.0)
        # initialize the structuring element
        struct = scipy.ndimage.generate_binary_structure(2, 1)
        # iterate it to expand it nxn
        n = mask_f * timestep / kmperpixel
        struct = scipy.ndimage.iterate_structure(struct, int((n - 1) / 2.0))

    if noise_method is None:
        R_m = [R_c[0][i].copy() for i in range(n_cascade_levels)]

    fft_objs = []
    for i in range(n_ens_members):
        fft_objs.append(utils.get_method(fft_method, shape=R.shape[1:]))

    if measure_time:
        init_time = time.time() - starttime_init

    R = R[-1, :, :]

    ###
    # 7. Calculate the initial skill of the (NWP) model forecasts at t=0
    ###
    rho_nwp_models = [
        blending.skill_scores.spatial_correlation(
            obs=R_c[0, :, -1, :, :], mod=R_models[n_model, 0, :, :, :]
        )
        for n_model in range(R_models.shape[0])
    ]
    rho_nwp_models = np.stack(rho_nwp_models)

    # Also initizalize the current and previous extrapolation forecast scale
    # for the nowcasting component
    rho_extr_prev = None
    rho_extr = None

    ###
    # 8. Start the forecasting loop
    ###
    print("Starting blended nowcast computation.")

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
    R_f_prev = R_c
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

        ###
        # 8.1.1 Determine the skill of the components for lead time (t0 + t)
        ###
        # First for the extrapolation component
        rho_extr, rho_extr_prev = blending.skill_scores.lt_dependent_cor_extrapolation(
            PHI=PHI, correlations=rho_extr, correlations_prev=rho_extr_prev
        )

        # the nowcast iteration for each ensemble member
        def worker(j):
            ###
            # 8.1.2 Determine the skill of the nwp components for lead time (t0 + t)
            ###
            # Then for the model components
            if blend_nwp_members == True:
                rho_nwp_fc = [
                    blending.skill_scores.lt_dependent_cor_nwp(
                        lt=(t * int(timestep)),
                        correlations=rho_nwp_models[n_model],
                        **clim_kwargs,
                    )
                    for n_model in range(rho_nwp_models.shape[0])
                ]
                rho_nwp_fc = np.stack(rho_nwp_fc)
                # Concatenate rho_extr and rho_nwp
                rho_fc = np.concatenate((rho_extr[None, :], rho_nwp_fc), axis=0)
            else:
                rho_nwp_fc = blending.skill_scores.lt_dependent_cor_nwp(
                    lt=(t * int(timestep)),
                    correlations=rho_nwp_models[j],
                    **clim_kwargs,
                )
                # Concatenate rho_extr and rho_nwp
                rho_fc = np.concatenate(
                    (rho_extr[None, :], rho_nwp_fc[None, :]), axis=0
                )

            ###
            # 8.2 Determine the weights per component
            ###
            # weight = [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
            weights = calculate_weights(rho_fc)
            # Also determine the weights of the components without the extrapolation
            # cascade, in case this is no data or outside the mask.
            weights_model_only = calculate_weights(rho_fc[1:, :])

            ###
            # 8.3 Determine the noise cascade and regress this to the subsequent
            # time step + regress the extrapolation component to the subsequent
            # time step
            ###
            # 8.3.1 Determine the epsilon, a cascade of temporally independent
            # but spatially correlated noise
            if noise_method is not None:
                # generate noise field
                EPS = generate_noise(
                    pp, randstate=randgen_prec[j], fft_method=fft_objs[j], domain=domain
                )

                # decompose the noise field into a cascade
                EPS = decomp_method(
                    EPS,
                    filter,
                    fft_method=fft_objs[j],
                    input_domain=domain,
                    output_domain=domain,
                    compute_stats=True,
                    normalize=True,
                    compact_output=True,
                )
            else:
                EPS = None

            # 8.3.2 regress the extrapolation component to the subsequent time
            # step
            # iterate the AR(p) model for each cascade level
            for i in range(n_cascade_levels):
                # apply AR(p) process to extrapolation cascade level
                if EPS is not None or vel_pert_method is not None:
                    R_c[j][i] = autoregression.iterate_ar_model(R_c[j][i], PHI[i, :])

                else:
                    # use the deterministic AR(p) model computed above if
                    # perturbations are disabled
                    R_c[j][i] = R_m[i]

            # 8.3.3 regress the noise component to the subsequent time step
            # iterate the AR(p) model for each cascade level
            for i in range(n_cascade_levels):
                # normalize the noise cascade
                if EPS is not None:
                    EPS_ = EPS["cascade_levels"][i]
                    EPS_ *= noise_std_coeffs[i]
                else:
                    EPS_ = None
                # apply AR(p) process to noise cascade level
                # (Returns zero noise if EPS is None)
                Yn_c[j][i] = autoregression.iterate_ar_model(
                    Yn_c[j][i], PHI[i, :], eps=EPS_
                )

            EPS = None
            EPS_ = None

            ###
            # 8.4 Perturb and blend the advection fields +
            # advect the extrapolation cascade to the current time step
            # (or subtimesteps if non-integer time steps are given)
            ###
            extrap_kwargs_ = extrap_kwargs.copy()
            V_pert = V
            R_f_ep_out = []
            R_pb_ep = []

            for t_sub in subtimesteps:
                if t_sub > 0:
                    t_diff_prev_int = t_sub - int(t_sub)
                    if t_diff_prev_int > 0.0:
                        R_f_ip = [
                            (1.0 - t_diff_prev_int) * R_f_prev[j][i][-1, :]
                            + t_diff_prev_int * R_c[j][i][-1, :]
                            for i in range(n_cascade_levels)
                        ]

                    else:
                        R_f_ip = [
                            R_f_prev[j][i][-1, :] for i in range(n_cascade_levels)
                        ]

                    R_f_ip = np.stack(R_f_ip)

                    t_diff_prev = t_sub - t_prev[j]
                    t_total[j] += t_diff_prev

                    # compute the perturbed motion field - include the NWP
                    # velocities and the weights
                    if vel_pert_method is not None:
                        V_pert = V + generate_vel_noise(vps[j], t_total[j] * timestep)

                    # Stack the perturbed extrapolation and the NWP velocities
                    if blend_nwp_members == True:
                        V_stack = np.concatenate(
                            (V_pert[None, :, :, :], V_models[:, t, :, :, :]), axis=0
                        )
                    else:
                        V_model_ = V_models[j, t, :, :, :]
                        V_stack = np.concatenate(
                            (V_pert[None, :, :, :], V_model_[None, :, :, :]), axis=0
                        )
                        V_model_ = None

                    # Obtain a blended optical flow, using the weights of the
                    # second cascade following eq. 24 in BPS2006
                    V_blended = blending.utils.blend_optical_flows(
                        flows=V_stack,
                        weights=weights[
                            :-1, 1
                        ],  # [(extr_field, n_model_fields), cascade_level=2]
                    )

                    # Extrapolate it to the next time step
                    R_f_ep = np.zeros(R_f_ip.shape)

                    min_R_f_ip = np.min(R_f_ip)
                    # min_Yn_ip = np.min(Yn_ip)
                    for i in range(n_cascade_levels):
                        extrap_kwargs_["displacement_prev"] = D[j][i]
                        # First, extrapolate the extrapolation component
                        R_f_ep_, D[j][i] = extrapolator_method(
                            R_f_ip[i],
                            V_blended,
                            [t_diff_prev],
                            **extrap_kwargs_,
                        )

                        R_f_ep[i] = R_f_ep_[0]

                    # Make sure we have no nans left
                    nan_indices = np.isnan(R_f_ep)
                    R_f_ep[nan_indices] = min_R_f_ip

                    R_f_ep_out.append(R_f_ep)

                    # Finally, also extrapolate the previous radar rainfall
                    # field. This will be blended with the rainfall field(s)
                    # of the (NWP) model(s) for Lagrangian blended prob. matching
                    min_R = np.min(R)
                    R_pb_ep_, __ = extrapolator_method(
                        R,
                        V_blended,
                        [t_diff_prev],
                        **extrap_kwargs_,
                    )

                    # Make sure we have no nans left
                    R_pb_ep__ = R_pb_ep_[0]
                    nan_indices = np.isnan(R_pb_ep__)
                    R_pb_ep__[nan_indices] = min_R
                    R_pb_ep.append(R_pb_ep__)

                    t_prev[j] = t_sub

                R_f_prev[j] = R_c[j]

            if len(R_f_ep_out) > 0:
                R_f_ep_out = np.stack(R_f_ep_out)
                R_pb_ep = np.stack(R_pb_ep)

            # advect the forecast field by one time step if no subtimesteps in the
            # current interval were found
            if not subtimesteps:
                t_diff_prev = t + 1 - t_prev[j]
                t_total[j] += t_diff_prev

                # compute the perturbed motion field - include the NWP
                # velocities and the weights
                if vel_pert_method is not None:
                    V_pert = V + generate_vel_noise(vps[j], t_total[j] * timestep)

                # Stack the perturbed extrapolation and the NWP velocities
                if blend_nwp_members == True:
                    V_stack = np.concatenate(
                        (V_pert[None, :, :, :], V_models[:, t, :, :, :]), axis=0
                    )
                else:
                    V_model_ = V_models[j, t, :, :, :]
                    V_stack = np.concatenate(
                        (V_pert[None, :, :, :], V_model_[None, :, :, :]), axis=0
                    )
                    V_model_ = None

                # Obtain a blended optical flow, using the weights of the
                # second cascade following eq. 24 in BPS2006
                V_blended = blending.utils.blend_optical_flows(
                    flows=V_stack,
                    weights=weights[
                        :-1, 1
                    ],  # [(extr_field, n_model_fields), cascade_level=2]
                )

                for i in range(n_cascade_levels):
                    extrap_kwargs_["displacement_prev"] = D[j][i]
                    _, D[j][i] = extrapolator_method(
                        None,
                        V_blended,
                        [t_diff_prev],
                        **extrap_kwargs_,
                    )
                t_prev[j] = t + 1

            ###
            # 8.5 Blend the cascades
            ###
            R_f_out = []
            for t_sub in subtimesteps:
                # TODO: does it make sense to use sub time steps - check if it works?
                if t_sub > 0:
                    t_index = np.where(np.array(subtimesteps) == t_sub)[0][0]
                    # First concatenate the cascades and the means and sigmas
                    # R_models = [n_models,timesteps,n_cascade_levels,m,n]
                    if blend_nwp_members == True:
                        cascades_stacked = np.concatenate(
                            (
                                R_f_ep_out[None, t_index],
                                R_models[:, t],
                                Yn_c[None, j, :, -1, :],
                            ),
                            axis=0,
                        )  # [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
                        means_stacked = np.concatenate(
                            (mu_extrapolation[None, :], mu_models[:, t]), axis=0
                        )
                        sigmas_stacked = np.concatenate(
                            (sigma_extrapolation[None, :], sigma_models[:, t]),
                            axis=0,
                        )
                    else:
                        cascades_stacked = np.concatenate(
                            (
                                R_f_ep_out[None, t_index],
                                R_models[None, j, t],
                                Yn_c[None, j, :, -1, :],
                            ),
                            axis=0,
                        )  # [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
                        means_stacked = np.concatenate(
                            (mu_extrapolation[None, :], mu_models[None, j, t]), axis=0
                        )
                        sigmas_stacked = np.concatenate(
                            (sigma_extrapolation[None, :], sigma_models[None, j, t]),
                            axis=0,
                        )

                    # Blend the extrapolation, (NWP) model(s) and noise cascades
                    R_f_blended = blending.utils.blend_cascades(
                        cascades_norm=cascades_stacked, weights=weights
                    )
                    # Also blend the cascade without the extrapolation component
                    R_f_blended_mod_only = blending.utils.blend_cascades(
                        cascades_norm=cascades_stacked[1:, :],
                        weights=weights_model_only,
                    )

                    # Blend the means and standard deviations
                    # Input is array of shape [number_components, scale_level, ...]
                    means_blended, sigmas_blended = blend_means_sigmas(
                        means=means_stacked, sigmas=sigmas_stacked, weights=weights
                    )
                    # Also blend the means and sigmas for the cascade without extrapolation
                    (
                        means_blended_mod_only,
                        sigmas_blended_mod_only,
                    ) = blend_means_sigmas(
                        means=means_stacked[1:, :],
                        sigmas=sigmas_stacked[1:, :],
                        weights=weights_model_only,
                    )

                    ###
                    # 8.6 Recompose the cascade to a precipitation field
                    # (The function first normalizes the blended cascade, R_f_blended
                    # again)
                    ###
                    R_f_new = blending.utils.recompose_cascade(
                        combined_cascade=R_f_blended,
                        combined_mean=means_blended,
                        combined_sigma=sigmas_blended,
                    )
                    # The recomposed cascade without the extrapolation (for NaN filling)
                    R_f_new_mod_only = blending.utils.recompose_cascade(
                        combined_cascade=R_f_blended_mod_only,
                        combined_mean=means_blended_mod_only,
                        combined_sigma=sigmas_blended_mod_only,
                    )
                    if domain == "spectral":
                        # TODO: Check this! (Only tested with domain == 'spatial')
                        R_f_new = fft_objs[j].irfft2(R_f_new)
                        R_f_new_mod_only = fft_objs[j].irfft2(R_f_new_mod_only)

                    ###
                    # 8.7 Post-processing steps - use the mask and fill no data with
                    # the blended NWP forecast. Probability matching following
                    # Lagrangian blended probability matching which uses the
                    # latest extrapolated radar rainfall field blended with the
                    # nwp model(s) rainfall forecast fields as 'benchmark'.
                    ###
                    # TODO: Check probability matching method
                    # 8.7.1 first blend the extrapolated rainfall field with
                    # the NWP rainfall forecast for this time step using the
                    # weights at scale level 2.
                    weights_pm = weights[:-1, 1]  # Weights without noise, level 2
                    weights_pm_normalized = weights_pm / np.sum(weights_pm)
                    weights_pm_mod_only = weights[
                        1:-1, 1
                    ]  # Weights without noise, level 2
                    weights_pm_normalized_mod_only = weights_pm_mod_only / np.sum(
                        weights_pm_mod_only
                    )
                    # Stack the fields
                    if blend_nwp_members == True:
                        R_pb_stacked = np.concatenate(
                            (
                                R_pb_ep[None, t_index],
                                R_models_pm[:, t],
                            ),
                            axis=0,
                        )
                    else:
                        R_pb_stacked = np.concatenate(
                            (
                                R_pb_ep[None, t_index],
                                R_models_pm[None, j, t],
                            ),
                            axis=0,
                        )
                    # Blend it
                    R_pb_blended = np.sum(
                        weights_pm_normalized.reshape(
                            weights_pm_normalized.shape[0], 1, 1
                        )
                        * R_pb_stacked,
                        axis=0,
                    )
                    if blend_nwp_members == True:
                        R_pb_blended_mod_only = np.sum(
                            weights_pm_normalized_mod_only.reshape(
                                weights_pm_normalized_mod_only.shape[0], 1, 1
                            )
                            * R_models_pm[:, t],
                            axis=0,
                        )
                    else:
                        R_pb_blended_mod_only = R_models_pm[j, t]

                    # Only keep the extrapolation component where radar data
                    # initially was present.
                    # Replace any NaN-values with the forecast without
                    # extrapolation
                    R_f_new[domain_mask] = np.nan
                    # TODO: Check if the domain masking below is necessary and how
                    # to implement it. Current commented implementation filters
                    # away too much when prob matching is used...
                    # R_pb_blended[domain_mask] = np.nan # Also make sure the 'benchmark''  only uses the NWP forecast outside the radar domain.

                    nan_indices = np.isnan(R_f_new)
                    R_f_new[nan_indices] = R_f_new_mod_only[nan_indices]
                    # nan_indices = np.isnan(R_pb_blended)
                    # R_pb_blended[nan_indices] = R_pb_blended_mod_only[nan_indices]

                    # 8.7.2. Apply the masking and prob. matching
                    if mask_method is not None:
                        # apply the precipitation mask to prevent generation of new
                        # precipitation into areas where it was not originally
                        # observed
                        R_cmin = R_f_new.min()
                        if mask_method == "incremental":
                            # The incremental mask is slightly different from
                            # the implementation in the non-blended steps.py, as
                            # it is not based on the last forecast, but instead
                            # on R_pb_blended. Therefore, the buffer does not
                            # increase over time.
                            # Get the mask for this forecast
                            MASK_prec = R_pb_blended >= R_thr
                            # Buffer the mask
                            MASK_prec = _compute_incremental_mask(
                                MASK_prec, struct, mask_rim
                            )
                            # Get the final mask
                            R_f_new = R_cmin + (R_f_new - R_cmin) * MASK_prec
                            MASK_prec_ = R_f_new > R_cmin
                        elif mask_method == "obs":
                            # The mask equals the most recent benchmark
                            # rainfall field
                            MASK_prec_ = R_pb_blended >= R_thr

                        # Set to min value outside of mask
                        R_f_new[~MASK_prec_] = R_cmin

                    if probmatching_method == "cdf":
                        # adjust the CDF of the forecast to match the most recent
                        # benchmark rainfall field (R_pb_blended)
                        R_f_new = probmatching.nonparam_match_empirical_cdf(
                            R_f_new, R_pb_blended
                        )
                    elif probmatching_method == "mean":
                        # Use R_pb_blended as benchmark field and
                        mu_0 = np.mean(R_pb_blended[R_pb_blended >= R_thr])
                        MASK = R_f_new >= R_thr
                        mu_fct = np.mean(R_f_new[MASK])
                        R_f_new[MASK] = R_f_new[MASK] - mu_fct + mu_0

                    R_f_out.append(R_f_new)

            return R_f_out

        res = []
        for j in range(n_ens_members):
            if not DASK_IMPORTED or n_ens_members == 1:
                res.append(worker(j))
            else:
                res.append(dask.delayed(worker)(j))

        R_f_ = (
            dask.compute(*res, num_workers=num_ensemble_workers)
            if DASK_IMPORTED and n_ens_members > 1
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

        if return_output:
            for j in range(n_ens_members):
                R_f[j].extend(R_f_[j])

        R_f_ = None

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


def calculate_ratios(correlations):
    """Calculate explained variance ratios from correlation.

    Parameters
    ----------
    Array of shape [component, scale_level, ...]
      containing correlation (skills) for each component (NWP and nowcast),
      scale level, and optionally along [y, x] dimensions.

    Returns
    -------
    out : numpy array
      An array containing the ratios of explain variance for each
      component, scale level, ...
    """
    # correlations: [component, scale, ...]
    square_corrs = np.square(correlations)
    # Calculate the ratio of the explained variance to the unexplained
    # variance of the nowcast and NWP model components
    out = square_corrs / (1 - square_corrs)
    # out: [component, scale, ...]
    return out


def calculate_weights(correlations):
    """Calculate blending weights for STEPS blending from correlation.

    Parameters
    ----------
    correlations : array-like
      Array of shape [component, scale_level, ...]
      containing correlation (skills) for each component (NWP and nowcast),
      scale level, and optionally along [y, x] dimensions.

    Returns
    -------
    weights : array-like
      Array of shape [component+1, scale_level, ...]
      containing the weights to be used in STEPS blending for
      each original component plus an addtional noise component, scale level,
      and optionally along [y, x] dimensions.

    Notes
    -----
    The weights in the BPS method can sum op to more than 1.0. Hence, the
    blended cascade has the be (re-)normalized (mu = 0, sigma = 1.0) first
    before the blended cascade can be recomposed.
    """
    # correlations: [component, scale, ...]
    # Check if the correlations are positive, otherwise rho = 10e-5
    correlations = np.where(correlations < 10e-5, 10e-5, correlations)
    # Calculate weights for each source
    ratios = calculate_ratios(correlations)
    # ratios: [component, scale, ...]
    total_ratios = np.sum(ratios, axis=0)
    # total_ratios: [scale, ...] - the denominator of eq. 11 & 12 in BPS2006
    weights = correlations * np.sqrt(ratios / total_ratios)
    # weights: [component, scale, ...]
    # Calculate the weight of the noise component.
    # Original BPS2006 method in the following two lines (eq. 13)
    total_square_weights = np.sum(np.square(weights), axis=0)
    noise_weight = np.sqrt(1.0 - total_square_weights)
    # TODO: determine the weights method and/or add different functions

    # Finally, add the noise_weights to the weights variable.
    weights = np.concatenate((weights, noise_weight[None, ...]), axis=0)
    return weights


def blend_means_sigmas(means, sigmas, weights):
    """Calculate the blended means and sigmas, the normalization parameters
    needed to recompose the cascade. This procedure uses the weights of the
    blending of the normalized cascades and follows eq. 32 and 33 in BPS2004.


    Parameters
    ----------
    means : array-like
      Array of shape [number_components, scale_level, ...]
      with the mean for each component (NWP, nowcasts, noise).
    sigmas : array-like
      Array of shape [number_components, scale_level, ...]
      with the standard deviation for each component.
    weights : array-like
      An array of shape [number_components + 1, scale_level, ...]
      containing the weights to be used in this routine
      for each component plus noise, scale level, and optionally [y, x]
      dimensions, obtained by calling a method implemented in
      pysteps.blending.steps.calculate_weights

    Returns
    -------
    combined_means : array-like
      An array of shape [scale_level, ...]
      containing per scale level (cascade) the weighted combination of
      means from multiple components (NWP, nowcasts and noise).
    combined_sigmas : array-like
      An array of shape [scale_level, ...]
      similar to combined_means, but containing the standard deviations.

    """
    # Check if the dimensions are the same
    diff_dims = weights.ndim - means.ndim
    if diff_dims:
        for i in range(diff_dims):
            means = np.expand_dims(means, axis=means.ndim)
    diff_dims = weights.ndim - sigmas.ndim
    if diff_dims:
        for i in range(diff_dims):
            sigmas = np.expand_dims(sigmas, axis=sigmas.ndim)
    # Weight should have one component more (the noise component) than the
    # means and sigmas. Check this
    if (
        weights.shape[0] - means.shape[0] != 1
        or weights.shape[0] - sigmas.shape[0] != 1
    ):
        raise ValueError(
            "The weights array does not have one (noise) component more than mu and sigma"
        )
    else:
        # Throw away the last component, which is the noise component
        weights = weights[:-1]

    # Combine (blend) the means and sigmas
    combined_means = np.zeros(weights.shape[1])
    combined_sigmas = np.zeros(weights.shape[1])
    total_weight = np.sum((weights), axis=0)
    for i in range(weights.shape[0]):
        combined_means += (weights[i] / total_weight) * means[i]
        combined_sigmas += (weights[i] / total_weight) * sigmas[i]
    # TODO: substract covariances to weigthed sigmas - still necessary?

    return combined_means, combined_sigmas


def _check_inputs(R, R_d_models, V, V_models, timesteps, ar_order):
    if R.ndim != 3:
        raise ValueError("R must be a three-dimensional array")
    if R.shape[0] < ar_order + 1:
        raise ValueError("R.shape[0] < ar_order+1")
    if R_d_models.ndim != 2:
        raise ValueError(
            "R_d_models must be a two-dimensional array containing dictionaries"
        )
    if V.ndim != 3:
        raise ValueError("V must be a three-dimensional array")
    if V_models.ndim != 5:
        raise ValueError("V_models must be a five-dimensional array")
    if V.shape[0] != 2 or V_models.shape[2] != 2:
        raise ValueError(
            "V and V_models must have an x- and y-component, check the shape"
        )
    if R.shape[1:3] != V.shape[1:3]:
        raise ValueError(
            "dimension mismatch between R and V: shape(R)=%s, shape(V)=%s"
            % (str(R.shape), str(V.shape))
        )
    if R_d_models.shape[0] != V_models.shape[0]:
        raise ValueError(
            "R_d_models and V_models must consist of the same number of models"
        )
    if isinstance(timesteps, list) and not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")
    if isinstance(timesteps, list):
        if R_d_models.shape[1] != len(timesteps) + 1:
            raise ValueError(
                "R_models does not contain sufficient lead times for this forecast"
            )
    else:
        if R_d_models.shape[1] != timesteps + 1:
            raise ValueError(
                "R_models does not contain sufficient lead times for this forecast"
            )


def _compute_incremental_mask(Rbin, kr, r):
    # buffer the observation mask Rbin using the kernel kr
    # add a grayscale rim r (for smooth rain/no-rain transition)

    # buffer observation mask
    Rbin = np.ndarray.astype(Rbin.copy(), "uint8")
    Rd = scipy.ndimage.morphology.binary_dilation(Rbin, kr)

    # add grayscale rim
    kr1 = scipy.ndimage.generate_binary_structure(2, 1)
    mask = Rd.astype(float)
    for n in range(r):
        Rd = scipy.ndimage.morphology.binary_dilation(Rd, kr1)
        mask += Rd
    # normalize between 0 and 1
    return mask / mask.max()
