# -*- coding: utf-8 -*-
"""
pysteps.blending.steps
======================

Implementation of the STEPS stochastic blending method as described in
:cite:`BPS2004`, :cite:`BPS2006` and :cite:`SPN2013`. The STEPS blending method
consists of the following main steps:

    #. Set the radar rainfall fields in a Lagrangian space.
    #. Perform the cascade decomposition for the input radar rainfall fields.
       The method assumes that the cascade decomposition of the NWP model fields is
       already done prior to calling the function, as the NWP model fields are
       generally not updated with the same frequency (which is more efficient). A
       method to decompose and store the NWP model fields whenever a new NWP model
       field is present, is present in pysteps.blending.utils.decompose_NWP.
    #. Initialize the noise method.
    #. Estimate AR parameters for the extrapolation nowcast and noise cascade.
    #. Initialize all the random generators.
    #. Calculate the initial skill of the NWP model forecasts at t=0.
    #. Start the forecasting loop:
        #. Determine which NWP models will be combined with which nowcast ensemble
           member. The number of output ensemble members equals the maximum number
           of (ensemble) members in the input, which can be either the defined
           number of (nowcast) ensemble members or the number of NWP models/members.
        #. Determine the skill and weights of the forecasting components
           (extrapolation, NWP and noise) for that lead time.
        #. Regress the extrapolation and noise cascades separately to the subsequent
           time step.
        #. Extrapolate the extrapolation and noise cascades to the current time step.
        #. Blend the cascades.
        #. Recompose the cascade to a rainfall field.
        #. Post-processing steps (masking and probability matching, which are
           different from the original blended STEPS implementation).

.. autosummary::
    :toctree: ../generated/

    forecast
    calculate_ratios
    calculate_weights_bps
    calculate_weights_spn
    blend_means_sigmas
"""

import math
import time
from copy import deepcopy
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
from scipy.linalg import inv
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure

from pysteps import blending, cascade, extrapolation, noise, utils
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.postprocessing import probmatching
from pysteps.timeseries import autoregression, correlation

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


def forecast(
    precip,
    precip_models,
    velocity,
    velocity_models,
    timesteps,
    timestep,
    issuetime,
    n_ens_members,
    n_cascade_levels=6,
    blend_nwp_members=False,
    precip_thr=None,
    norain_thr=0.0,
    kmperpixel=None,
    extrap_method="semilagrangian",
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    noise_method="nonparametric",
    noise_stddev_adj=None,
    ar_order=2,
    vel_pert_method="bps",
    weights_method="bps",
    conditional=False,
    probmatching_method="cdf",
    mask_method="incremental",
    resample_distribution=True,
    smooth_radar_mask_range=0,
    callback=None,
    return_output=True,
    seed=None,
    num_workers=1,
    fft_method="numpy",
    domain="spatial",
    outdir_path_skill="./tmp/",
    extrap_kwargs=None,
    filter_kwargs=None,
    noise_kwargs=None,
    vel_pert_kwargs=None,
    clim_kwargs=None,
    mask_kwargs=None,
    measure_time=False,
):
    """
    Generate a blended nowcast ensemble by using the Short-Term Ensemble
    Prediction System (STEPS) method.

    Parameters
    ----------
    precip: array-like
      Array of shape (ar_order+1,m,n) containing the input precipitation fields
      ordered by timestamp from oldest to newest. The time steps between the
      inputs are assumed to be regular.
    precip_models: array-like
      Either raw (NWP) model forecast data or decomposed (NWP) model forecast data.
      If you supply decomposed data, it needs to be an array of shape
      (n_models,timesteps+1) containing, per timestep (t=0 to lead time here) and
      per (NWP) model or model ensemble member, a dictionary with a list of cascades
      obtained by calling a method implemented in :py:mod:`pysteps.cascade.decomposition`.
      If you supply the original (NWP) model forecast data, it needs to be an array of shape
      (n_models,timestep+1,m,n) containing precipitation (or other) fields, which will
      then be decomposed in this function.

      Depending on your use case it can be advantageous to decompose the model
      forecasts outside beforehand, as this slightly reduces calculation times.
      This is possible with :py:func:`pysteps.blending.utils.decompose_NWP`,
      :py:func:`pysteps.blending.utils.compute_store_nwp_motion`, and
      :py:func:`pysteps.blending.utils.load_NWP`. However, if you have a lot of (NWP) model
      members (e.g. 1 model member per nowcast member), this can lead to excessive memory
      usage.

      To further reduce memory usage, both this array and the ``velocity_models`` array
      can be given as float32. They will then be converted to float64 before computations
      to minimize loss in precision.

      In case of one (deterministic) model as input, add an extra dimension to make sure
      precip_models is four dimensional prior to calling this function.
    velocity: array-like
      Array of shape (2,m,n) containing the x- and y-components of the advection
      field. The velocities are assumed to represent one time step between the
      inputs. All values are required to be finite.
    velocity_models: array-like
      Array of shape (n_models,timestep,2,m,n) containing the x- and y-components
      of the advection field for the (NWP) model field per forecast lead time.
      All values are required to be finite.

      To reduce memory usage, this array
      can be given as float32. They will then be converted to float64 before computations
      to minimize loss in precision.
    timesteps: int or list of floats
      Number of time steps to forecast or a list of time steps for which the
      forecasts are computed (relative to the input time step). The elements of
      the list are required to be in ascending order.
    timestep: float
      Time step of the motion vectors (minutes). Required if vel_pert_method is
      not None or mask_method is 'incremental'.
    issuetime: datetime
      Datetime object containing the date and time for which the forecast
      is issued.
    n_ens_members: int
      The number of ensemble members to generate. This number should always be
      equal to or larger than the number of NWP ensemble members / number of
      NWP models.
    n_cascade_levels: int, optional
      The number of cascade levels to use. Defaults to 6,
      see issue #385 on GitHub.
    blend_nwp_members: bool
      Check if NWP models/members should be used individually, or if all of
      them are blended together per nowcast ensemble member. Standard set to
      false.
    precip_thr: float, optional
      Specifies the threshold value for minimum observable precipitation
      intensity. Required if mask_method is not None or conditional is True.
    norain_thr: float, optional
      Specifies the threshold value for the fraction of rainy (see above) pixels
      in the radar rainfall field below which we consider there to be no rain.
      Depends on the amount of clutter typically present.
      Standard set to 0.0
    kmperpixel: float, optional
      Spatial resolution of the input data (kilometers/pixel). Required if
      vel_pert_method is not None or mask_method is 'incremental'.
    extrap_method: str, optional
      Name of the extrapolation method to use. See the documentation of
      :py:mod:`pysteps.extrapolation.interface`.
    decomp_method: {'fft'}, optional
      Name of the cascade decomposition method to use. See the documentation
      of :py:mod:`pysteps.cascade.interface`.
    bandpass_filter_method: {'gaussian', 'uniform'}, optional
      Name of the bandpass filter method to use with the cascade decomposition.
      See the documentation of :py:mod:`pysteps.cascade.interface`.
    noise_method: {'parametric','nonparametric','ssft','nested',None}, optional
      Name of the noise generator to use for perturbating the precipitation
      field. See the documentation of :py:mod:`pysteps.noise.interface`. If set to None,
      no noise is generated.
    noise_stddev_adj: {'auto','fixed',None}, optional
      Optional adjustment for the standard deviations of the noise fields added
      to each cascade level. This is done to compensate incorrect std. dev.
      estimates of casace levels due to presence of no-rain areas. 'auto'=use
      the method implemented in :py:func:`pysteps.noise.utils.compute_noise_stddev_adjs`.
      'fixed'= use the formula given in :cite:`BPS2006` (eq. 6), None=disable
      noise std. dev adjustment.
    ar_order: int, optional
      The order of the autoregressive model to use. Must be >= 1.
    vel_pert_method: {'bps',None}, optional
      Name of the noise generator to use for perturbing the advection field. See
      the documentation of :py:mod:`pysteps.noise.interface`. If set to None, the advection
      field is not perturbed.
    weights_method: {'bps','spn'}, optional
      The calculation method of the blending weights. Options are the method
      by :cite:`BPS2006` and the covariance-based method by :cite:`SPN2013`.
      Defaults to bps.
    conditional: bool, optional
      If set to True, compute the statistics of the precipitation field
      conditionally by excluding pixels where the values are below the threshold
      precip_thr.
    probmatching_method: {'cdf','mean',None}, optional
      Method for matching the statistics of the forecast field with those of
      the most recently observed one. 'cdf'=map the forecast CDF to the observed
      one, 'mean'=adjust only the conditional mean value of the forecast field
      in precipitation areas, None=no matching applied. Using 'mean' requires
      that mask_method is not None.
    mask_method: {'obs','incremental',None}, optional
      The method to use for masking no precipitation areas in the forecast field.
      The masked pixels are set to the minimum value of the observations.
      'obs' = apply precip_thr to the most recently observed precipitation intensity
      field, 'incremental' = iteratively buffer the mask with a certain rate
      (currently it is 1 km/min), None=no masking.
    resample_distribution: bool, optional
        Method to resample the distribution from the extrapolation and NWP cascade as input
        for the probability matching. Not resampling these distributions may lead to losing
        some extremes when the weight of both the extrapolation and NWP cascade is similar.
        Defaults to True.
    smooth_radar_mask_range: int, Default is 0.
      Method to smooth the transition between the radar-NWP-noise blend and the NWP-noise
      blend near the edge of the radar domain (radar mask), where the radar data is either
      not present anymore or is not reliable. If set to 0 (grid cells), this generates a
      normal forecast without smoothing. To create a smooth mask, this range should be a
      positive value, representing a buffer band of a number of pixels by which the mask
      is cropped and smoothed. The smooth radar mask removes the hard edges between NWP
      and radar in the final blended product. Typically, a value between 50 and 100 km
      can be used. 80 km generally gives good results.
    callback: function, optional
      Optional function that is called after computation of each time step of
      the nowcast. The function takes one argument: a three-dimensional array
      of shape (n_ens_members,h,w), where h and w are the height and width
      of the input field precip, respectively. This can be used, for instance,
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
      A string defining the FFT method to use (see FFT methods in
      :py:func:`pysteps.utils.interface.get_method`).
      Defaults to 'numpy' for compatibility reasons. If pyFFTW is installed,
      the recommended method is 'pyfftw'.
    domain: {"spatial", "spectral"}
      If "spatial", all computations are done in the spatial domain (the
      classical STEPS model). If "spectral", the AR(2) models and stochastic
      perturbations are applied directly in the spectral domain to reduce
      memory footprint and improve performance :cite:`PCH2019b`.
    outdir_path_skill: string, optional
      Path to folder where the historical skill are stored. Defaults to
      path_workdir from rcparams. If no path is given, './tmp' will be used.
    extrap_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the extrapolation
      method. See the documentation of :py:func:`pysteps.extrapolation.interface`.
    filter_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the filter method.
      See the documentation of :py:mod:`pysteps.cascade.bandpass_filters`.
    noise_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the initializer of
      the noise generator. See the documentation of :py:mod:`pysteps.noise.fftgenerators`.
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

      See :py:mod:`pysteps.noise.motion` for additional documentation.
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
      taken from the input precipitation fields precip. If measure_time is True, the
      return value is a three-element tuple containing the nowcast array, the
      initialization time of the nowcast generator and the time used in the
      main loop (seconds).

    See also
    --------
    :py:mod:`pysteps.extrapolation.interface`, :py:mod:`pysteps.cascade.interface`,
    :py:mod:`pysteps.noise.interface`, :py:func:`pysteps.noise.utils.compute_noise_stddev_adjs`

    References
    ----------
    :cite:`Seed2003`, :cite:`BPS2004`, :cite:`BPS2006`, :cite:`SPN2013`, :cite:`PCH2019b`

    Notes
    -----
    1. The blending currently does not blend the beta-parameters in the parametric
    noise method. It is recommended to use the non-parameteric noise method.

    2. If blend_nwp_members is True, the BPS2006 method for the weights is
    suboptimal. It is recommended to use the SPN2013 method instead.

    3. Not yet implemented (and neither in the steps nowcasting module): The regression
    of the lag-1 and lag-2 parameters to their climatological values. See also eq.
    12 - 19 in :cite: `BPS2004`. By doing so, the Phi parameters change over time,
    which enhances the AR process. This can become a future development if this
    turns out to be a warranted functionality.
    """

    # 0.1 Start with some checks
    _check_inputs(precip, precip_models, velocity, velocity_models, timesteps, ar_order)

    if extrap_kwargs is None:
        extrap_kwargs = dict()

    if filter_kwargs is None:
        filter_kwargs = dict()

    if noise_kwargs is None:
        noise_kwargs = dict()

    if vel_pert_kwargs is None:
        vel_pert_kwargs = dict()

    if clim_kwargs is None:
        # Make sure clim_kwargs at least contains the number of models
        clim_kwargs = dict({"n_models": precip_models.shape[0]})

    if mask_kwargs is None:
        mask_kwargs = dict()

    if np.any(~np.isfinite(velocity)):
        raise ValueError("velocity contains non-finite values")

    if mask_method not in ["obs", "incremental", None]:
        raise ValueError(
            "unknown mask method %s: must be 'obs', 'incremental' or None" % mask_method
        )

    if conditional and precip_thr is None:
        raise ValueError("conditional=True but precip_thr is not set")

    if mask_method is not None and precip_thr is None:
        raise ValueError("mask_method!=None but precip_thr=None")

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
    print("STEPS blending")
    print("==============")
    print("")

    print("Inputs")
    print("------")
    print(f"forecast issue time:         {issuetime.isoformat()}")
    print(f"input dimensions:            {precip.shape[1]}x{precip.shape[2]}")
    if kmperpixel is not None:
        print(f"km/pixel:                    {kmperpixel}")
    if timestep is not None:
        print(f"time step:                   {timestep} minutes")
    print("")

    print("NWP and blending inputs")
    print("-----------------------")
    print(f"number of (NWP) models:      {precip_models.shape[0]}")
    print(f"blend (NWP) model members:   {blend_nwp_members}")
    print(f"decompose (NWP) models:      {'yes' if precip_models.ndim == 4 else 'no'}")
    print("")

    print("Methods")
    print("-------")
    print(f"extrapolation:               {extrap_method}")
    print(f"bandpass filter:             {bandpass_filter_method}")
    print(f"decomposition:               {decomp_method}")
    print(f"noise generator:             {noise_method}")
    print(f"noise adjustment:            {'yes' if noise_stddev_adj else 'no'}")
    print(f"velocity perturbator:        {vel_pert_method}")
    print(f"blending weights method:     {weights_method}")
    print(f"conditional statistics:      {'yes' if conditional else 'no'}")
    print(f"precip. mask method:         {mask_method}")
    print(f"probability matching:        {probmatching_method}")
    print(f"FFT method:                  {fft_method}")
    print(f"domain:                      {domain}")
    print("")

    print("Parameters")
    print("----------")
    if isinstance(timesteps, int):
        print(f"number of time steps:        {timesteps}")
    else:
        print(f"time steps:                  {timesteps}")
    print(f"ensemble size:               {n_ens_members}")
    print(f"parallel threads:            {num_workers}")
    print(f"number of cascade levels:    {n_cascade_levels}")
    print(f"order of the AR(p) model:    {ar_order}")
    if vel_pert_method == "bps":
        vp_par = vel_pert_kwargs.get("p_par", noise.motion.get_default_params_bps_par())
        vp_perp = vel_pert_kwargs.get(
            "p_perp", noise.motion.get_default_params_bps_perp()
        )
        print(f"vel. pert., parallel:        {vp_par[0]},{vp_par[1]},{vp_par[2]}")
        print(f"vel. pert., perpendicular:   {vp_perp[0]},{vp_perp[1]},{vp_perp[2]}")
    else:
        vp_par, vp_perp = None, None

    if conditional or mask_method is not None:
        print(f"precip. intensity threshold: {precip_thr}")
    print(f"no-rain fraction threshold for radar: {norain_thr}")
    print("")

    # 0.3 Get the methods that will be used
    num_ensemble_workers = n_ens_members if num_workers > n_ens_members else num_workers

    if measure_time:
        starttime_init = time.time()

    fft = utils.get_method(fft_method, shape=precip.shape[1:], n_threads=num_workers)

    precip_shape = precip.shape[1:]

    # initialize the band-pass filter
    filter_method = cascade.get_method(bandpass_filter_method)
    bp_filter = filter_method(precip_shape, n_cascade_levels, **filter_kwargs)

    decompositor, recompositor = cascade.get_method(decomp_method)

    extrapolator = extrapolation.get_method(extrap_method)

    x_values, y_values = np.meshgrid(
        np.arange(precip.shape[2]), np.arange(precip.shape[1])
    )

    xy_coords = np.stack([x_values, y_values])

    precip = precip[-(ar_order + 1) :, :, :].copy()

    # determine the domain mask from non-finite values
    domain_mask = np.logical_or.reduce(
        [~np.isfinite(precip[i, :]) for i in range(precip.shape[0])]
    )

    # determine the precipitation threshold mask
    if conditional:
        MASK_thr = np.logical_and.reduce(
            [precip[i, :, :] >= precip_thr for i in range(precip.shape[0])]
        )
    else:
        MASK_thr = None

    # we need to know the zerovalue of precip to replace the mask when decomposing after extrapolation
    zerovalue = np.nanmin(precip)

    # 1. Start with the radar rainfall fields. We want the fields in a
    # Lagrangian space
    precip = _transform_to_lagrangian(
        precip, velocity, ar_order, xy_coords, extrapolator, extrap_kwargs, num_workers
    )

    # 2. Perform the cascade decomposition for the input precip fields and
    # and, if necessary, for the (NWP) model fields
    # 2.1 Compute the cascade decompositions of the input precipitation fields
    (
        precip_cascade,
        mu_extrapolation,
        sigma_extrapolation,
    ) = _compute_cascade_decomposition_radar(
        precip,
        ar_order,
        n_cascade_levels,
        n_ens_members,
        MASK_thr,
        domain,
        bp_filter,
        decompositor,
        fft,
    )

    # 2.2 If necessary, recompose (NWP) model forecasts
    precip_models_cascade = None

    if precip_models.ndim != 4:
        precip_models_cascade = precip_models
        precip_models = _compute_cascade_recomposition_nwp(precip_models, recompositor)

    # 2.3 Check for zero input fields in the radar and NWP data.
    zero_precip_radar = blending.utils.check_norain(precip, precip_thr, norain_thr)
    # The norain fraction threshold used for nwp is the default value of 0.0,
    # since nwp does not suffer from clutter.
    zero_model_fields = blending.utils.check_norain(
        precip_models, precip_thr, norain_thr
    )

    if isinstance(timesteps, int):
        timesteps = list(range(timesteps + 1))
        timestep_type = "int"
    else:
        original_timesteps = [0] + list(timesteps)
        timesteps = nowcast_utils.binned_timesteps(original_timesteps)
        timestep_type = "list"

    # 2.3.1 If precip is below the norain threshold and precip_models is zero,
    # we consider it as no rain in the domain.
    # The forecast will directly return an array filled with the minimum
    # value present in precip (which equals zero rainfall in the used
    # transformation)
    if zero_precip_radar and zero_model_fields:
        print(
            "No precipitation above the threshold found in both the radar and NWP fields"
        )
        print("The resulting forecast will contain only zeros")
        # Create the output list
        R_f = [[] for j in range(n_ens_members)]

        # Save per time step to ensure the array does not become too large if
        # no return_output is requested and callback is not None.
        for t, subtimestep_idx in enumerate(timesteps):
            # If the timestep is not the first one, we need to provide the zero forecast
            if t > 0:
                # Create an empty np array with shape [n_ens_members, rows, cols]
                # and fill it with the minimum value from precip (corresponding to
                # zero precipitation)
                R_f_ = np.full(
                    (n_ens_members, precip_shape[0], precip_shape[1]), np.nanmin(precip)
                )
                if subtimestep_idx:
                    if callback is not None:
                        if R_f_.shape[1] > 0:
                            callback(R_f_.squeeze())
                    if return_output:
                        for j in range(n_ens_members):
                            R_f[j].append(R_f_[j])

                R_f_ = None

        if measure_time:
            zero_precip_time = time.time() - starttime_init

        if return_output:
            outarr = np.stack([np.stack(R_f[j]) for j in range(n_ens_members)])
            if measure_time:
                return outarr, zero_precip_time, zero_precip_time
            else:
                return outarr
        else:
            return None

    else:
        # 2.3.3 If zero_precip_radar, make sure that precip_cascade does not contain
        # only nans or infs. If so, fill it with the zero value.
        if zero_precip_radar:
            # Look for a timestep and member with rain so that we have a sensible decomposition
            done = False
            for t in timesteps:
                if done:
                    break
                for j in range(precip_models.shape[0]):
                    if not blending.utils.check_norain(
                        precip_models[j, t], precip_thr, norain_thr
                    ):
                        if precip_models_cascade is not None:
                            precip_cascade[~np.isfinite(precip_cascade)] = np.nanmin(
                                precip_models_cascade[j, t]["cascade_levels"]
                            )
                            continue
                        precip_models_cascade_temp = decompositor(
                            precip_models[j, t, :, :],
                            bp_filter=bp_filter,
                            fft_method=fft,
                            output_domain=domain,
                            normalize=True,
                            compute_stats=True,
                            compact_output=True,
                        )["cascade_levels"]
                        precip_cascade[~np.isfinite(precip_cascade)] = np.nanmin(
                            precip_models_cascade_temp
                        )
                        done = True
                        break

        # 2.3.5 If zero_precip_radar is True, only use the velocity field of the NWP
        # forecast. I.e., velocity (radar) equals velocity_model at the first time
        # step.
        if zero_precip_radar:
            # Use the velocity from velocity_models at time step 0
            velocity = velocity_models[:, 0, :, :, :].astype(np.float64, copy=False)
            # Take the average over the first axis, which corresponds to n_models
            # (hence, the model average)
            velocity = np.mean(velocity, axis=0)

        # 3. Initialize the noise method.
        # If zero_precip_radar is True, initialize noise based on the NWP field time
        # step where the fraction of rainy cells is highest (because other lead times
        # might be zero as well). Else, initialize the noise with the radar
        # rainfall data
        if zero_precip_radar:
            precip_noise_input = _determine_max_nr_rainy_cells_nwp(
                precip_models, precip_thr, precip_models.shape[0], timesteps
            )
            # Make sure precip_noise_input is three dimensional
            if len(precip_noise_input.shape) != 3:
                precip_noise_input = precip_noise_input[np.newaxis, :, :]
        else:
            precip_noise_input = precip.copy()

        pp, generate_noise, noise_std_coeffs = _init_noise(
            precip_noise_input,
            precip_thr,
            n_cascade_levels,
            bp_filter,
            decompositor,
            fft,
            noise_method,
            noise_kwargs,
            noise_stddev_adj,
            measure_time,
            num_workers,
            seed,
        )
        precip_noise_input = None

        # 4. Estimate AR parameters for the radar rainfall field
        PHI = _estimate_ar_parameters_radar(
            precip_cascade,
            ar_order,
            n_cascade_levels,
            MASK_thr,
            zero_precip_radar,
        )

        # 5. Repeat precip_cascade for n ensemble members
        # First, discard all except the p-1 last cascades because they are not needed
        # for the AR(p) model

        precip_cascade = np.stack(
            [[precip_cascade[i][-ar_order:].copy() for i in range(n_cascade_levels)]]
            * n_ens_members
        )

        # 6. Initialize all the random generators and prepare for the forecast loop
        randgen_prec, vps, generate_vel_noise = _init_random_generators(
            velocity,
            noise_method,
            vel_pert_method,
            vp_par,
            vp_perp,
            seed,
            n_ens_members,
            kmperpixel,
            timestep,
        )
        D, D_Yn, D_pb, R_f, R_m, mask_rim, struct, fft_objs = _prepare_forecast_loop(
            precip_cascade,
            noise_method,
            fft_method,
            n_cascade_levels,
            n_ens_members,
            mask_method,
            mask_kwargs,
            timestep,
            kmperpixel,
        )

        # Also initialize the cascade of temporally correlated noise, which has the
        # same shape as precip_cascade, but starts random noise.
        noise_cascade, mu_noise, sigma_noise = _init_noise_cascade(
            shape=precip_cascade.shape,
            n_ens_members=n_ens_members,
            n_cascade_levels=n_cascade_levels,
            generate_noise=generate_noise,
            decompositor=decompositor,
            pp=pp,
            randgen_prec=randgen_prec,
            fft_objs=fft_objs,
            bp_filter=bp_filter,
            domain=domain,
            noise_method=noise_method,
            noise_std_coeffs=noise_std_coeffs,
            ar_order=ar_order,
        )

        precip = precip[-1, :, :]

        # 7. initizalize the current and previous extrapolation forecast scale
        # for the nowcasting component
        rho_extr_prev = np.repeat(1.0, PHI.shape[0])
        rho_extr = PHI[:, 0] / (1.0 - PHI[:, 1])  # phi1 / (1 - phi2), see BPS2004

        if measure_time:
            init_time = time.time() - starttime_init

        ###
        # 8. Start the forecasting loop
        ###
        print("Starting blended nowcast computation.")

        if measure_time:
            starttime_mainloop = time.time()

        extrap_kwargs["return_displacement"] = True

        forecast_prev = deepcopy(precip_cascade)
        noise_prev = deepcopy(noise_cascade)

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
                    f"Computing nowcast for time step {t}... ",
                    end="",
                    flush=True,
                )

            if measure_time:
                starttime = time.time()

            if precip_models_cascade is not None:
                decomp_precip_models = list(precip_models_cascade[:, t])
            else:
                if precip_models.shape[0] == 1:
                    decomp_precip_models = [
                        decompositor(
                            precip_models[0, t, :, :],
                            bp_filter=bp_filter,
                            fft_method=fft,
                            output_domain=domain,
                            normalize=True,
                            compute_stats=True,
                            compact_output=True,
                        )
                    ]
                else:
                    with ThreadPool(num_workers) as pool:
                        decomp_precip_models = pool.map(
                            partial(
                                decompositor,
                                bp_filter=bp_filter,
                                fft_method=fft,
                                output_domain=domain,
                                normalize=True,
                                compute_stats=True,
                                compact_output=True,
                            ),
                            list(precip_models[:, t, :, :]),
                        )

            precip_models_cascade_temp = np.array(
                [decomp["cascade_levels"] for decomp in decomp_precip_models]
            )
            mu_models_temp = np.array(
                [decomp["means"] for decomp in decomp_precip_models]
            )
            sigma_models_temp = np.array(
                [decomp["stds"] for decomp in decomp_precip_models]
            )

            # 2.3.4 Check if the NWP fields contain nans or infinite numbers. If so,
            # fill these with the minimum value present in precip (corresponding to
            # zero rainfall in the radar observations)
            (
                precip_models_cascade_temp,
                precip_models_temp,
                mu_models_temp,
                sigma_models_temp,
            ) = _fill_nans_infs_nwp_cascade(
                precip_models_cascade_temp,
                precip_models[:, t, :, :].astype(np.float64, copy=False),
                precip_cascade,
                precip,
                mu_models_temp,
                sigma_models_temp,
            )

            # 8.1.1 Before calling the worker for the forecast loop, determine which (NWP)
            # models will be combined with which nowcast ensemble members. With the
            # way it is implemented at this moment: n_ens_members of the output equals
            # the maximum number of (ensemble) members in the input (either the nowcasts or NWP).
            (
                precip_models_cascade_temp,
                precip_models_temp,
                velocity_models_temp,
                mu_models_temp,
                sigma_models_temp,
                n_model_indices,
            ) = _find_nwp_combination(
                precip_models_cascade_temp,
                precip_models_temp,
                velocity_models[:, t, :, :, :].astype(np.float64, copy=False),
                mu_models_temp,
                sigma_models_temp,
                n_ens_members,
                ar_order,
                n_cascade_levels,
                blend_nwp_members,
            )

            # If zero_precip_radar is True, set the velocity field equal to the NWP
            # velocity field for the current time step (velocity_models_temp).
            if zero_precip_radar:
                # Use the velocity from velocity_models and take the average over
                # n_models (axis=0)
                velocity = np.mean(velocity_models_temp, axis=0)

            if t == 0:
                # 8.1.2 Calculate the initial skill of the (NWP) model forecasts at t=0
                rho_nwp_models = _compute_initial_nwp_skill(
                    precip_cascade,
                    precip_models_cascade_temp,
                    domain_mask,
                    issuetime,
                    outdir_path_skill,
                    clim_kwargs,
                )

            if t > 0:
                # 8.1.3 Determine the skill of the components for lead time (t0 + t)
                # First for the extrapolation component. Only calculate it when t > 0.
                (
                    rho_extr,
                    rho_extr_prev,
                ) = blending.skill_scores.lt_dependent_cor_extrapolation(
                    PHI=PHI, correlations=rho_extr, correlations_prev=rho_extr_prev
                )

            # the nowcast iteration for each ensemble member
            R_f_ = [None for _ in range(n_ens_members)]

            def worker(j):
                # 8.1.2 Determine the skill of the nwp components for lead time (t0 + t)
                # Then for the model components
                if blend_nwp_members:
                    rho_nwp_fc = [
                        blending.skill_scores.lt_dependent_cor_nwp(
                            lt=(t * int(timestep)),
                            correlations=rho_nwp_models[n_model],
                            outdir_path=outdir_path_skill,
                            n_model=n_model,
                            skill_kwargs=clim_kwargs,
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
                        outdir_path=outdir_path_skill,
                        n_model=n_model_indices[j],
                        skill_kwargs=clim_kwargs,
                    )
                    # Concatenate rho_extr and rho_nwp
                    rho_fc = np.concatenate(
                        (rho_extr[None, :], rho_nwp_fc[None, :]), axis=0
                    )

                # 8.2 Determine the weights per component

                # Weights following the bps method. These are needed for the velocity
                # weights prior to the advection step. If weights method spn is
                # selected, weights will be overwritten with those weights prior to
                # blending step.
                # weight = [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
                weights = calculate_weights_bps(rho_fc)

                # The model only weights
                if weights_method == "bps":
                    # Determine the weights of the components without the extrapolation
                    # cascade, in case this is no data or outside the mask.
                    weights_model_only = calculate_weights_bps(rho_fc[1:, :])
                elif weights_method == "spn":
                    # Only the weights of the components without the extrapolation
                    # cascade will be determined here. The full set of weights are
                    # determined after the extrapolation step in this method.
                    if blend_nwp_members and precip_models_cascade_temp.shape[0] > 1:
                        weights_model_only = np.zeros(
                            (precip_models_cascade_temp.shape[0] + 1, n_cascade_levels)
                        )
                        for i in range(n_cascade_levels):
                            # Determine the normalized covariance matrix (containing)
                            # the cross-correlations between the models
                            cov = np.corrcoef(
                                np.stack(
                                    [
                                        precip_models_cascade_temp[
                                            n_model, i, :, :
                                        ].flatten()
                                        for n_model in range(
                                            precip_models_cascade_temp.shape[0]
                                        )
                                    ]
                                )
                            )
                            # Determine the weights for this cascade level
                            weights_model_only[:, i] = calculate_weights_spn(
                                correlations=rho_fc[1:, i], cov=cov
                            )
                    else:
                        # Same as correlation and noise is 1 - correlation
                        weights_model_only = calculate_weights_bps(rho_fc[1:, :])
                else:
                    raise ValueError(
                        "Unknown weights method %s: must be 'bps' or 'spn'"
                        % weights_method
                    )

                # 8.3 Determine the noise cascade and regress this to the subsequent
                # time step + regress the extrapolation component to the subsequent
                # time step

                # 8.3.1 Determine the epsilon, a cascade of temporally independent
                # but spatially correlated noise
                if noise_method is not None:
                    # generate noise field
                    EPS = generate_noise(
                        pp,
                        randstate=randgen_prec[j],
                        fft_method=fft_objs[j],
                        domain=domain,
                    )

                    # decompose the noise field into a cascade
                    EPS = decompositor(
                        EPS,
                        bp_filter,
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
                        precip_cascade[j][i] = autoregression.iterate_ar_model(
                            precip_cascade[j][i], PHI[i, :]
                        )
                        # Renormalize the cascade
                        precip_cascade[j][i][1] /= np.std(precip_cascade[j][i][1])
                    else:
                        # use the deterministic AR(p) model computed above if
                        # perturbations are disabled
                        precip_cascade[j][i] = R_m[i]

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
                    noise_cascade[j][i] = autoregression.iterate_ar_model(
                        noise_cascade[j][i], PHI[i, :], eps=EPS_
                    )

                EPS = None
                EPS_ = None

                # 8.4 Perturb and blend the advection fields + advect the
                # extrapolation and noise cascade to the current time step
                # (or subtimesteps if non-integer time steps are given)

                # Settings and initialize the output
                extrap_kwargs_ = extrap_kwargs.copy()
                extrap_kwargs_noise = extrap_kwargs.copy()
                extrap_kwargs_pb = extrap_kwargs.copy()
                velocity_pert = velocity
                R_f_ep_out = []
                Yn_ep_out = []
                R_pm_ep = []

                # Extrapolate per sub time step
                for t_sub in subtimesteps:
                    if t_sub > 0:
                        t_diff_prev_int = t_sub - int(t_sub)
                        if t_diff_prev_int > 0.0:
                            R_f_ip = [
                                (1.0 - t_diff_prev_int) * forecast_prev[j][i][-1, :]
                                + t_diff_prev_int * precip_cascade[j][i][-1, :]
                                for i in range(n_cascade_levels)
                            ]
                            Yn_ip = [
                                (1.0 - t_diff_prev_int) * noise_prev[j][i][-1, :]
                                + t_diff_prev_int * noise_cascade[j][i][-1, :]
                                for i in range(n_cascade_levels)
                            ]

                        else:
                            R_f_ip = [
                                forecast_prev[j][i][-1, :]
                                for i in range(n_cascade_levels)
                            ]
                            Yn_ip = [
                                noise_prev[j][i][-1, :] for i in range(n_cascade_levels)
                            ]

                        R_f_ip = np.stack(R_f_ip)
                        Yn_ip = np.stack(Yn_ip)

                        t_diff_prev = t_sub - t_prev[j]
                        t_total[j] += t_diff_prev

                        # compute the perturbed motion field - include the NWP
                        # velocities and the weights. Note that we only perturb
                        # the extrapolation velocity field, as the NWP velocity
                        # field is present per time step
                        if vel_pert_method is not None:
                            velocity_pert = velocity + generate_vel_noise(
                                vps[j], t_total[j] * timestep
                            )

                        # Stack the perturbed extrapolation and the NWP velocities
                        if blend_nwp_members:
                            V_stack = np.concatenate(
                                (
                                    velocity_pert[None, :, :, :],
                                    velocity_models_temp,
                                ),
                                axis=0,
                            )
                        else:
                            V_model_ = velocity_models_temp[j]
                            V_stack = np.concatenate(
                                (velocity_pert[None, :, :, :], V_model_[None, :, :, :]),
                                axis=0,
                            )
                            V_model_ = None

                        # Obtain a blended optical flow, using the weights of the
                        # second cascade following eq. 24 in BPS2006
                        velocity_blended = blending.utils.blend_optical_flows(
                            flows=V_stack,
                            weights=weights[
                                :-1, 1
                            ],  # [(extr_field, n_model_fields), cascade_level=2]
                        )

                        # Extrapolate both cascades to the next time step
                        # First recompose the cascade, advect it and decompose it again
                        # This is needed to remove the interpolation artifacts.
                        # In addition, the number of extrapolations is greatly reduced
                        # A. Radar Rain
                        R_f_ip_recomp = blending.utils.recompose_cascade(
                            combined_cascade=R_f_ip,
                            combined_mean=mu_extrapolation,
                            combined_sigma=sigma_extrapolation,
                        )
                        # Make sure we have values outside the mask
                        if zero_precip_radar:
                            R_f_ip_recomp = np.nan_to_num(
                                R_f_ip_recomp,
                                copy=True,
                                nan=zerovalue,
                                posinf=zerovalue,
                                neginf=zerovalue,
                            )
                        # Put back the mask
                        R_f_ip_recomp[domain_mask] = np.nan
                        extrap_kwargs["displacement_prev"] = D[j]
                        R_f_ep_recomp_, D[j] = extrapolator(
                            R_f_ip_recomp,
                            velocity_blended,
                            [t_diff_prev],
                            allow_nonfinite_values=True,
                            **extrap_kwargs,
                        )
                        R_f_ep_recomp = R_f_ep_recomp_[0].copy()
                        temp_mask = ~np.isfinite(R_f_ep_recomp)
                        # TODO WHERE DO CAN I FIND THIS -15.0
                        R_f_ep_recomp[~np.isfinite(R_f_ep_recomp)] = zerovalue
                        R_f_ep = decompositor(
                            R_f_ep_recomp,
                            bp_filter,
                            mask=MASK_thr,
                            fft_method=fft,
                            output_domain=domain,
                            normalize=True,
                            compute_stats=True,
                            compact_output=True,
                        )["cascade_levels"]
                        # Make sure we have values outside the mask
                        if zero_precip_radar:
                            R_f_ep = np.nan_to_num(
                                R_f_ep,
                                copy=True,
                                nan=np.nanmin(R_f_ip),
                                posinf=np.nanmin(R_f_ip),
                                neginf=np.nanmin(R_f_ip),
                            )
                        for i in range(n_cascade_levels):
                            R_f_ep[i][temp_mask] = np.nan
                        # B. Noise
                        Yn_ip_recomp = blending.utils.recompose_cascade(
                            combined_cascade=Yn_ip,
                            combined_mean=mu_noise[j],
                            combined_sigma=sigma_noise[j],
                        )
                        extrap_kwargs_noise["displacement_prev"] = D_Yn[j]
                        extrap_kwargs_noise["map_coordinates_mode"] = "wrap"
                        Yn_ep_recomp_, D_Yn[j] = extrapolator(
                            Yn_ip_recomp,
                            velocity_blended,
                            [t_diff_prev],
                            allow_nonfinite_values=True,
                            **extrap_kwargs_noise,
                        )
                        Yn_ep_recomp = Yn_ep_recomp_[0].copy()
                        Yn_ep = decompositor(
                            Yn_ep_recomp,
                            bp_filter,
                            mask=MASK_thr,
                            fft_method=fft,
                            output_domain=domain,
                            normalize=True,
                            compute_stats=True,
                            compact_output=True,
                        )["cascade_levels"]
                        for i in range(n_cascade_levels):
                            Yn_ep[i] *= noise_std_coeffs[i]

                        # Append the results to the output lists
                        R_f_ep_out.append(R_f_ep.copy())
                        Yn_ep_out.append(Yn_ep.copy())
                        R_f_ip = None
                        R_f_ip_recomp = None
                        R_f_ep_recomp_ = None
                        R_f_ep_recomp = None
                        R_f_ep = None
                        Yn_ip = None
                        Yn_ip_recomp = None
                        Yn_ep_recomp_ = None
                        Yn_ep_recomp = None
                        Yn_ep = None

                        # Finally, also extrapolate the initial radar rainfall
                        # field. This will be blended with the rainfall field(s)
                        # of the (NWP) model(s) for Lagrangian blended prob. matching
                        # min_R = np.min(precip)
                        extrap_kwargs_pb["displacement_prev"] = D_pb[j]
                        # Apply the domain mask to the extrapolation component
                        R_ = precip.copy()
                        R_[domain_mask] = np.nan
                        R_pm_ep_, D_pb[j] = extrapolator(
                            R_,
                            velocity_blended,
                            [t_diff_prev],
                            allow_nonfinite_values=True,
                            **extrap_kwargs_pb,
                        )
                        R_pm_ep.append(R_pm_ep_[0])

                        t_prev[j] = t_sub

                if len(R_f_ep_out) > 0:
                    R_f_ep_out = np.stack(R_f_ep_out)
                    Yn_ep_out = np.stack(Yn_ep_out)
                    R_pm_ep = np.stack(R_pm_ep)

                # advect the forecast field by one time step if no subtimesteps in the
                # current interval were found
                if not subtimesteps:
                    t_diff_prev = t + 1 - t_prev[j]
                    t_total[j] += t_diff_prev

                    # compute the perturbed motion field - include the NWP
                    # velocities and the weights
                    if vel_pert_method is not None:
                        velocity_pert = velocity + generate_vel_noise(
                            vps[j], t_total[j] * timestep
                        )

                    # Stack the perturbed extrapolation and the NWP velocities
                    if blend_nwp_members:
                        V_stack = np.concatenate(
                            (velocity_pert[None, :, :, :], velocity_models_temp),
                            axis=0,
                        )
                    else:
                        V_model_ = velocity_models_temp[j]
                        V_stack = np.concatenate(
                            (velocity_pert[None, :, :, :], V_model_[None, :, :, :]),
                            axis=0,
                        )
                        V_model_ = None

                    # Obtain a blended optical flow, using the weights of the
                    # second cascade following eq. 24 in BPS2006
                    velocity_blended = blending.utils.blend_optical_flows(
                        flows=V_stack,
                        weights=weights[
                            :-1, 1
                        ],  # [(extr_field, n_model_fields), cascade_level=2]
                    )

                    # Extrapolate the extrapolation and noise cascade

                    extrap_kwargs_["displacement_prev"] = D[j]
                    extrap_kwargs_noise["displacement_prev"] = D_Yn[j]
                    extrap_kwargs_noise["map_coordinates_mode"] = "wrap"

                    _, D[j] = extrapolator(
                        None,
                        velocity_blended,
                        [t_diff_prev],
                        allow_nonfinite_values=True,
                        **extrap_kwargs_,
                    )

                    _, D_Yn[j] = extrapolator(
                        None,
                        velocity_blended,
                        [t_diff_prev],
                        allow_nonfinite_values=True,
                        **extrap_kwargs_noise,
                    )

                    # Also extrapolate the radar observation, used for the probability
                    # matching and post-processing steps
                    extrap_kwargs_pb["displacement_prev"] = D_pb[j]
                    _, D_pb[j] = extrapolator(
                        None,
                        velocity_blended,
                        [t_diff_prev],
                        allow_nonfinite_values=True,
                        **extrap_kwargs_pb,
                    )

                    t_prev[j] = t + 1

                forecast_prev[j] = precip_cascade[j]
                noise_prev[j] = noise_cascade[j]

                # 8.5 Blend the cascades
                R_f_out = []

                for t_sub in subtimesteps:
                    # TODO: does it make sense to use sub time steps - check if it works?
                    if t_sub > 0:
                        t_index = np.where(np.array(subtimesteps) == t_sub)[0][0]
                        # First concatenate the cascades and the means and sigmas
                        # precip_models = [n_models,timesteps,n_cascade_levels,m,n]
                        if blend_nwp_members:
                            cascades_stacked = np.concatenate(
                                (
                                    R_f_ep_out[None, t_index],
                                    precip_models_cascade_temp,
                                    Yn_ep_out[None, t_index],
                                ),
                                axis=0,
                            )  # [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
                            means_stacked = np.concatenate(
                                (mu_extrapolation[None, :], mu_models_temp), axis=0
                            )
                            sigmas_stacked = np.concatenate(
                                (sigma_extrapolation[None, :], sigma_models_temp),
                                axis=0,
                            )
                        else:
                            cascades_stacked = np.concatenate(
                                (
                                    R_f_ep_out[None, t_index],
                                    precip_models_cascade_temp[None, j],
                                    Yn_ep_out[None, t_index],
                                ),
                                axis=0,
                            )  # [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
                            means_stacked = np.concatenate(
                                (mu_extrapolation[None, :], mu_models_temp[None, j]),
                                axis=0,
                            )
                            sigmas_stacked = np.concatenate(
                                (
                                    sigma_extrapolation[None, :],
                                    sigma_models_temp[None, j],
                                ),
                                axis=0,
                            )

                        # First determine the blending weights if method is spn. The
                        # weights for method bps have already been determined.
                        if weights_method == "spn":
                            weights = np.zeros(
                                (cascades_stacked.shape[0], n_cascade_levels)
                            )
                            for i in range(n_cascade_levels):
                                # Determine the normalized covariance matrix (containing)
                                # the cross-correlations between the models
                                cascades_stacked_ = np.stack(
                                    [
                                        cascades_stacked[n_model, i, :, :].flatten()
                                        for n_model in range(
                                            cascades_stacked.shape[0] - 1
                                        )
                                    ]
                                )  # -1 to exclude the noise component
                                cov = np.ma.corrcoef(
                                    np.ma.masked_invalid(cascades_stacked_)
                                )
                                # Determine the weights for this cascade level
                                weights[:, i] = calculate_weights_spn(
                                    correlations=rho_fc[:, i], cov=cov
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

                        # 8.6 Recompose the cascade to a precipitation field
                        # (The function first normalizes the blended cascade, R_f_blended
                        # again)
                        R_f_new = blending.utils.recompose_cascade(
                            combined_cascade=R_f_blended,
                            combined_mean=means_blended,
                            combined_sigma=sigmas_blended,
                        )
                        # The recomposed cascade without the extrapolation (for NaN filling
                        # outside the radar domain)
                        R_f_new_mod_only = blending.utils.recompose_cascade(
                            combined_cascade=R_f_blended_mod_only,
                            combined_mean=means_blended_mod_only,
                            combined_sigma=sigmas_blended_mod_only,
                        )
                        if domain == "spectral":
                            # TODO: Check this! (Only tested with domain == 'spatial')
                            R_f_new = fft_objs[j].irfft2(R_f_new)
                            R_f_new_mod_only = fft_objs[j].irfft2(R_f_new_mod_only)

                        # 8.7 Post-processing steps - use the mask and fill no data with
                        # the blended NWP forecast. Probability matching following
                        # Lagrangian blended probability matching which uses the
                        # latest extrapolated radar rainfall field blended with the
                        # nwp model(s) rainfall forecast fields as 'benchmark'.

                        # 8.7.1 first blend the extrapolated rainfall field (the field
                        # that is only used for post-processing steps) with the NWP
                        # rainfall forecast for this time step using the weights
                        # at scale level 2.
                        weights_pm = weights[:-1, 1]  # Weights without noise, level 2
                        weights_pm_normalized = weights_pm / np.sum(weights_pm)
                        # And the weights for outside the radar domain
                        weights_pm_mod_only = weights_model_only[
                            :-1, 1
                        ]  # Weights without noise, level 2
                        weights_pm_normalized_mod_only = weights_pm_mod_only / np.sum(
                            weights_pm_mod_only
                        )
                        # Stack the fields
                        if blend_nwp_members:
                            R_pm_stacked = np.concatenate(
                                (
                                    R_pm_ep[None, t_index],
                                    precip_models_temp,
                                ),
                                axis=0,
                            )
                        else:
                            R_pm_stacked = np.concatenate(
                                (
                                    R_pm_ep[None, t_index],
                                    precip_models_temp[None, j],
                                ),
                                axis=0,
                            )
                        # Blend it
                        R_pm_blended = np.sum(
                            weights_pm_normalized.reshape(
                                weights_pm_normalized.shape[0], 1, 1
                            )
                            * R_pm_stacked,
                            axis=0,
                        )
                        if blend_nwp_members:
                            R_pm_blended_mod_only = np.sum(
                                weights_pm_normalized_mod_only.reshape(
                                    weights_pm_normalized_mod_only.shape[0], 1, 1
                                )
                                * precip_models_temp,
                                axis=0,
                            )
                        else:
                            R_pm_blended_mod_only = precip_models_temp[j]

                        # The extrapolation components are NaN outside the advected
                        # radar domain. This results in NaN values in the blended
                        # forecast outside the radar domain. Therefore, fill these
                        # areas with the "..._mod_only" blended forecasts, consisting
                        # of the NWP and noise components.

                        nan_indices = np.isnan(R_f_new)
                        if smooth_radar_mask_range != 0:
                            # Compute the smooth dilated mask
                            new_mask = blending.utils.compute_smooth_dilated_mask(
                                nan_indices,
                                max_padding_size_in_px=smooth_radar_mask_range,
                            )

                            # Ensure mask values are between 0 and 1
                            mask_model = np.clip(new_mask, 0, 1)
                            mask_radar = np.clip(1 - new_mask, 0, 1)

                            # Handle NaNs in R_f_new and R_f_new_mod_only by setting NaNs to 0 in the blending step
                            R_f_new_mod_only_no_nan = np.nan_to_num(
                                R_f_new_mod_only, nan=0
                            )
                            R_f_new_no_nan = np.nan_to_num(R_f_new, nan=0)

                            # Perform the blending of radar and model inside the radar domain using a weighted combination
                            R_f_new = np.nansum(
                                [
                                    mask_model * R_f_new_mod_only_no_nan,
                                    mask_radar * R_f_new_no_nan,
                                ],
                                axis=0,
                            )

                            nan_indices = np.isnan(R_pm_blended)
                            R_pm_blended = np.nansum(
                                [
                                    R_pm_blended * mask_radar,
                                    R_pm_blended_mod_only * mask_model,
                                ],
                                axis=0,
                            )
                        else:
                            R_f_new[nan_indices] = R_f_new_mod_only[nan_indices]
                            nan_indices = np.isnan(R_pm_blended)
                            R_pm_blended[nan_indices] = R_pm_blended_mod_only[
                                nan_indices
                            ]

                        # Finally, fill the remaining nan values, if present, with
                        # the minimum value in the forecast
                        nan_indices = np.isnan(R_f_new)
                        R_f_new[nan_indices] = np.nanmin(R_f_new)
                        nan_indices = np.isnan(R_pm_blended)
                        R_pm_blended[nan_indices] = np.nanmin(R_pm_blended)

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
                                # on R_pm_blended. Therefore, the buffer does not
                                # increase over time.
                                # Get the mask for this forecast
                                MASK_prec = R_pm_blended >= precip_thr
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
                                MASK_prec_ = R_pm_blended >= precip_thr

                            # Set to min value outside of mask
                            R_f_new[~MASK_prec_] = R_cmin

                        # If probmatching_method is not None, resample the distribution from
                        # both the extrapolation cascade and the model (NWP) cascade and use
                        # that for the probability matching.
                        if probmatching_method is not None and resample_distribution:
                            arr1 = R_pm_ep[t_index]
                            arr2 = precip_models_temp[j]
                            # resample weights based on cascade level 2.
                            # Areas where one of the fields is nan are not included.
                            R_pm_resampled = probmatching.resample_distributions(
                                first_array=arr1,
                                second_array=arr2,
                                probability_first_array=weights_pm_normalized[0],
                            )
                        else:
                            R_pm_resampled = R_pm_blended.copy()

                        if probmatching_method == "cdf":
                            # nan indices in the extrapolation nowcast
                            nan_indices = np.isnan(R_pm_ep[t_index])
                            # Adjust the CDF of the forecast to match the resampled distribution combined from
                            # extrapolation and model fields.
                            # Rainfall outside the pure extrapolation domain is not taken into account.
                            if np.any(np.isfinite(R_f_new)):
                                R_f_new = probmatching.nonparam_match_empirical_cdf(
                                    R_f_new, R_pm_resampled, nan_indices
                                )
                                R_pm_resampled = None
                        elif probmatching_method == "mean":
                            # Use R_pm_blended as benchmark field and
                            mu_0 = np.mean(R_pm_resampled[R_pm_resampled >= precip_thr])
                            MASK = R_f_new >= precip_thr
                            mu_fct = np.mean(R_f_new[MASK])
                            R_f_new[MASK] = R_f_new[MASK] - mu_fct + mu_0
                            R_pm_resampled = None

                        R_f_out.append(R_f_new)

                R_f_[j] = R_f_out

            res = []

            if DASK_IMPORTED and n_ens_members > 1:
                for j in range(n_ens_members):
                    res.append(dask.delayed(worker)(j))
                dask.compute(*res, num_workers=num_ensemble_workers)
            else:
                for j in range(n_ens_members):
                    worker(j)

            res = None

            if is_nowcast_time_step:
                if measure_time:
                    print(f"{time.time() - starttime:.2f} seconds.")
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


def calculate_weights_bps(correlations):
    """Calculate BPS blending weights for STEPS blending from correlation.

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

    References
    ----------
    :cite:`BPS2006`

    Notes
    -----
    The weights in the BPS method can sum op to more than 1.0.
    """
    # correlations: [component, scale, ...]
    # Check if the correlations are positive, otherwise rho = 10e-5
    correlations = np.where(correlations < 10e-5, 10e-5, correlations)

    # If we merge more than one component with the noise cascade, we follow
    # the weights impolementation in either :cite:`BPS2006` or :cite:`SPN2013`.
    if correlations.shape[0] > 1:
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
        # Finally, add the noise_weights to the weights variable.
        weights = np.concatenate((weights, noise_weight[None, ...]), axis=0)

    # Otherwise, the weight equals the correlation on that scale level and
    # the noise component weight equals 1 - this weight. This only occurs for
    # the weights calculation outside the radar domain where in the case of 1
    # NWP model or ensemble member, no blending of multiple models has to take
    # place
    else:
        noise_weight = 1.0 - correlations
        weights = np.concatenate((correlations, noise_weight), axis=0)

    return weights


def calculate_weights_spn(correlations, cov):
    """Calculate SPN blending weights for STEPS blending from correlation.

    Parameters
    ----------
    correlations : array-like
      Array of shape [n_components]
      containing correlation (skills) for each component (NWP models and nowcast).
    cov : array-like
        Array of shape [n_components, n_components] containing the covariance
        matrix of the models that will be blended. If cov is set to None and
        correlations only contains one model, the weight equals the correlation
        on that scale level and the noise component weight equals 1 - this weight.

    Returns
    -------
    weights : array-like
      Array of shape [component+1]
      containing the weights to be used in STEPS blending for each original
      component plus an addtional noise component.

    References
    ----------
    :cite:`SPN2013`
    """
    # Check if the correlations are positive, otherwise rho = 10e-5
    correlations = np.where(correlations < 10e-5, 10e-5, correlations)

    if correlations.shape[0] > 1 and len(cov) > 1:
        if isinstance(cov, type(None)):
            raise ValueError("cov must contain a covariance matrix")
        else:
            # Make a numpy array out of cov and get the inverse
            cov = np.where(cov == 0.0, 10e-5, cov)
            # Make sure the determinant of the matrix is not zero, otherwise
            # subtract 10e-5 from the cross-correlations between the models
            if np.linalg.det(cov) == 0.0:
                cov = cov - 10e-5
            # Ensure the correlation of the model with itself is always 1.0
            for i, _ in enumerate(cov):
                cov[i][i] = 1.0
            # Use a numpy array instead of a matrix
            cov_matrix = np.array(cov)
            # Get the inverse of the matrix using scipy's inv function
            cov_matrix_inv = inv(cov_matrix)
            # The component weights are the dot product between cov_matrix_inv and cor_vec
            weights = np.dot(cov_matrix_inv, correlations)
            weights = np.nan_to_num(
                weights, copy=True, nan=10e-5, posinf=10e-5, neginf=10e-5
            )
            weights_dot_correlations = np.dot(weights, correlations)
            # If the dot product of the weights with the correlations is
            # larger than 1.0, we assign a weight of 0.0 to the noise (to make
            # it numerically stable)
            if weights_dot_correlations > 1.0:
                noise_weight = np.array([0])
            # Calculate the noise weight
            else:
                noise_weight = np.sqrt(1.0 - weights_dot_correlations)
            # Convert weights to a 1D array
            weights = np.array(weights).flatten()
            # Ensure noise_weight is a 1D array before concatenation
            noise_weight = np.array(noise_weight).flatten()
            # Finally, add the noise_weights to the weights variable.
            weights = np.concatenate((weights, noise_weight), axis=0)

    # Otherwise, the weight equals the correlation on that scale level and
    # the noise component weight equals 1 - this weight. This only occurs for
    # the weights calculation outside the radar domain where in the case of 1
    # NWP model or ensemble member, no blending of multiple models has to take
    # place
    else:
        noise_weight = 1.0 - correlations
        weights = np.concatenate((correlations, noise_weight), axis=0)

    # Make sure weights are always a real number
    weights = np.nan_to_num(weights, copy=True, nan=10e-5, posinf=10e-5, neginf=10e-5)

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
      dimensions, obtained by calling either
      :py:func:`pysteps.blending.steps.calculate_weights_bps` or
      :py:func:`pysteps.blending.steps.calculate_weights_spn`.

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


def _check_inputs(
    precip, precip_models, velocity, velocity_models, timesteps, ar_order
):
    if precip.ndim != 3:
        raise ValueError("precip must be a three-dimensional array")
    if precip.shape[0] < ar_order + 1:
        raise ValueError("precip.shape[0] < ar_order+1")
    if precip_models.ndim != 2 and precip_models.ndim != 4:
        raise ValueError(
            "precip_models must be either a two-dimensional array containing dictionaries with decomposed model fields or a four-dimensional array containing the original (NWP) model forecasts"
        )
    if velocity.ndim != 3:
        raise ValueError("velocity must be a three-dimensional array")
    if velocity_models.ndim != 5:
        raise ValueError("velocity_models must be a five-dimensional array")
    if velocity.shape[0] != 2 or velocity_models.shape[2] != 2:
        raise ValueError(
            "velocity and velocity_models must have an x- and y-component, check the shape"
        )
    if precip.shape[1:3] != velocity.shape[1:3]:
        raise ValueError(
            "dimension mismatch between precip and velocity: shape(precip)=%s, shape(velocity)=%s"
            % (str(precip.shape), str(velocity.shape))
        )
    if precip_models.shape[0] != velocity_models.shape[0]:
        raise ValueError(
            "precip_models and velocity_models must consist of the same number of models"
        )
    if isinstance(timesteps, list) and not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")
    if isinstance(timesteps, list):
        if precip_models.shape[1] != math.ceil(timesteps[-1]) + 1:
            raise ValueError(
                "precip_models does not contain sufficient lead times for this forecast"
            )
    else:
        if precip_models.shape[1] != timesteps + 1:
            raise ValueError(
                "precip_models does not contain sufficient lead times for this forecast"
            )


def _compute_incremental_mask(Rbin, kr, r):
    # buffer the observation mask Rbin using the kernel kr
    # add a grayscale rim r (for smooth rain/no-rain transition)

    # buffer observation mask
    Rbin = np.ndarray.astype(Rbin.copy(), "uint8")
    Rd = binary_dilation(Rbin, kr)

    # add grayscale rim
    kr1 = generate_binary_structure(2, 1)
    mask = Rd.astype(float)
    for n in range(r):
        Rd = binary_dilation(Rd, kr1)
        mask += Rd
    # normalize between 0 and 1
    return mask / mask.max()


def _transform_to_lagrangian(
    precip, velocity, ar_order, xy_coords, extrapolator, extrap_kwargs, num_workers
):
    """Advect the previous precipitation fields to the same position with the
    most recent one (i.e. transform them into the Lagrangian coordinates).
    """
    extrap_kwargs = extrap_kwargs.copy()
    extrap_kwargs["xy_coords"] = xy_coords
    res = list()

    def f(precip, i):
        return extrapolator(
            precip[i, :, :],
            velocity,
            ar_order - i,
            "min",
            allow_nonfinite_values=True,
            **extrap_kwargs,
        )[-1]

    for i in range(ar_order):
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
    return precip


def _init_noise(
    precip,
    precip_thr,
    n_cascade_levels,
    bp_filter,
    decompositor,
    fft,
    noise_method,
    noise_kwargs,
    noise_stddev_adj,
    measure_time,
    num_workers,
    seed,
):
    """Initialize the noise method."""
    if noise_method is None:
        return None, None, None

    # get methods for perturbations
    init_noise, generate_noise = noise.get_method(noise_method)

    # initialize the perturbation generator for the precipitation field
    pp = init_noise(precip, fft_method=fft, **noise_kwargs)

    if noise_stddev_adj == "auto":
        print("Computing noise adjustment coefficients... ", end="", flush=True)
        if measure_time:
            starttime = time.time()

        R_min = np.min(precip)
        noise_std_coeffs = noise.utils.compute_noise_stddev_adjs(
            precip[-1, :, :],
            precip_thr,
            R_min,
            bp_filter,
            decompositor,
            pp,
            generate_noise,
            20,
            conditional=True,
            num_workers=num_workers,
            seed=seed,
        )

        if measure_time:
            print(f"{time.time() - starttime:.2f} seconds.")
        else:
            print("done.")
    elif noise_stddev_adj == "fixed":
        f = lambda k: 1.0 / (0.75 + 0.09 * k)
        noise_std_coeffs = [f(k) for k in range(1, n_cascade_levels + 1)]
    else:
        noise_std_coeffs = np.ones(n_cascade_levels)

    if noise_stddev_adj is not None:
        print(f"noise std. dev. coeffs:   {noise_std_coeffs}")

    return pp, generate_noise, noise_std_coeffs


def _compute_cascade_decomposition_radar(
    precip,
    ar_order,
    n_cascade_levels,
    n_ens_members,
    MASK_thr,
    domain,
    bp_filter,
    decompositor,
    fft,
):
    """Compute the cascade decompositions of the input precipitation fields."""
    R_d = []
    for i in range(ar_order + 1):
        R_ = decompositor(
            precip[i, :, :],
            bp_filter,
            mask=MASK_thr,
            fft_method=fft,
            output_domain=domain,
            normalize=True,
            compute_stats=True,
            compact_output=True,
        )
        R_d.append(R_)

    # Rearrange the cascaded into a four-dimensional array of shape
    # (n_cascade_levels,ar_order+1,m,n) for the autoregressive model
    R_c = nowcast_utils.stack_cascades(R_d, n_cascade_levels)

    R_d = R_d[-1]
    mu_extrapolation = np.array(R_d["means"])
    sigma_extrapolation = np.array(R_d["stds"])
    R_d = [R_d.copy() for j in range(n_ens_members)]
    return R_c, mu_extrapolation, sigma_extrapolation


def _compute_cascade_recomposition_nwp(precip_models_cascade, recompositor):
    """If necessary, recompose (NWP) model forecasts."""
    precip_models = None

    # Recompose the (NWP) model cascades to have rainfall fields per
    # model and time step, which will be used in the probability matching steps.
    # Recomposed cascade will have shape: [n_models, n_timesteps, m, n]
    precip_models = []
    for i in range(precip_models_cascade.shape[0]):
        precip_model = []
        for time_step in range(precip_models_cascade.shape[1]):
            precip_model.append(recompositor(precip_models_cascade[i, time_step]))
        precip_models.append(precip_model)

    precip_models = np.stack(precip_models)
    precip_model = None

    return precip_models


def _estimate_ar_parameters_radar(
    R_c, ar_order, n_cascade_levels, MASK_thr, zero_precip_radar
):
    """Estimate AR parameters for the radar rainfall field."""
    # If there are values in the radar fields, compute the autocorrelations
    GAMMA = np.empty((n_cascade_levels, ar_order))
    if not zero_precip_radar:
        # compute lag-l temporal autocorrelation coefficients for each cascade level
        for i in range(n_cascade_levels):
            GAMMA[i, :] = correlation.temporal_autocorrelation(R_c[i], mask=MASK_thr)

    # Else, use standard values for the autocorrelations
    else:
        # Get the climatological lag-1 and lag-2 autocorrelation values from Table 2
        # in `BPS2004`.
        # Hard coded, change to own (climatological) values when present.
        GAMMA = np.array(
            [
                [0.99805, 0.9925, 0.9776, 0.9297, 0.796, 0.482, 0.079, 0.0006],
                [0.9933, 0.9752, 0.923, 0.750, 0.367, 0.069, 0.0018, 0.0014],
            ]
        )

        # Check whether the number of cascade_levels is correct
        if GAMMA.shape[1] > n_cascade_levels:
            GAMMA = GAMMA[:, 0:n_cascade_levels]
        elif GAMMA.shape[1] < n_cascade_levels:
            # Get the number of cascade levels that is missing
            n_extra_lev = n_cascade_levels - GAMMA.shape[1]
            # Append the array with correlation values of 10e-4
            GAMMA = np.append(
                GAMMA,
                [np.repeat(0.0006, n_extra_lev), np.repeat(0.0014, n_extra_lev)],
                axis=1,
            )

        # Finally base GAMMA.shape[0] on the AR-level
        if ar_order == 1:
            GAMMA = GAMMA[0, :]
        if ar_order > 2:
            for repeat_index in range(ar_order - 2):
                GAMMA = np.vstack((GAMMA, GAMMA[1, :]))

        # Finally, transpose GAMMA to ensure that the shape is the same as np.empty((n_cascade_levels, ar_order))
        GAMMA = GAMMA.transpose()
        assert GAMMA.shape == (n_cascade_levels, ar_order)

    # Print the GAMMA value
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
    return PHI


def _find_nwp_combination(
    precip_models,
    R_models_pm,
    velocity_models,
    mu_models,
    sigma_models,
    n_ens_members,
    ar_order,
    n_cascade_levels,
    blend_nwp_members,
):
    """Determine which (NWP) models will be combined with which nowcast ensemble members.
    With the way it is implemented at this moment: n_ens_members of the output equals
    the maximum number of (ensemble) members in the input (either the nowcasts or NWP).
    """
    # Make sure the number of model members is not larger than than or equal to
    # n_ens_members
    n_model_members = precip_models.shape[0]
    if n_model_members > n_ens_members:
        raise ValueError(
            "The number of NWP model members is larger than the given number of ensemble members. n_model_members <= n_ens_members."
        )

    # Check if NWP models/members should be used individually, or if all of
    # them are blended together per nowcast ensemble member.
    if blend_nwp_members:
        n_model_indices = None

    else:
        # Start with determining the maximum and mimimum number of members/models
        # in both input products
        n_ens_members_max = max(n_ens_members, n_model_members)
        n_ens_members_min = min(n_ens_members, n_model_members)
        # Also make a list of the model index numbers. These indices are needed
        # for indexing the right climatological skill file when pysteps calculates
        # the blended forecast in parallel.
        if n_model_members > 1:
            n_model_indices = np.arange(n_model_members)
        else:
            n_model_indices = [0]

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
                precip_models = np.repeat(precip_models, n_ens_members_max, axis=0)
                mu_models = np.repeat(mu_models, n_ens_members_max, axis=0)
                sigma_models = np.repeat(sigma_models, n_ens_members_max, axis=0)
                velocity_models = np.repeat(velocity_models, n_ens_members_max, axis=0)
                # For the prob. matching
                R_models_pm = np.repeat(R_models_pm, n_ens_members_max, axis=0)
                # Finally, for the model indices
                n_model_indices = np.repeat(n_model_indices, n_ens_members_max, axis=0)

            elif n_model_members == n_ens_members_min:
                repeats = [
                    (n_ens_members_max + i) // n_ens_members_min
                    for i in range(n_ens_members_min)
                ]
                if n_model_members == n_ens_members_min:
                    precip_models = np.repeat(precip_models, repeats, axis=0)
                    mu_models = np.repeat(mu_models, repeats, axis=0)
                    sigma_models = np.repeat(sigma_models, repeats, axis=0)
                    velocity_models = np.repeat(velocity_models, repeats, axis=0)
                    # For the prob. matching
                    R_models_pm = np.repeat(R_models_pm, repeats, axis=0)
                    # Finally, for the model indices
                    n_model_indices = np.repeat(n_model_indices, repeats, axis=0)

    return (
        precip_models,
        R_models_pm,
        velocity_models,
        mu_models,
        sigma_models,
        n_model_indices,
    )


def _init_random_generators(
    velocity,
    noise_method,
    vel_pert_method,
    vp_par,
    vp_perp,
    seed,
    n_ens_members,
    kmperpixel,
    timestep,
):
    """Initialize all the random generators."""
    if noise_method is not None:
        randgen_prec = []
        randgen_motion = []
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
            vp_ = init_vel_noise(velocity, 1.0 / kmperpixel, timestep, **kwargs)
            vps.append(vp_)
    else:
        vps, generate_vel_noise = None, None

    return randgen_prec, vps, generate_vel_noise


def _prepare_forecast_loop(
    R_c,
    noise_method,
    fft_method,
    n_cascade_levels,
    n_ens_members,
    mask_method,
    mask_kwargs,
    timestep,
    kmperpixel,
):
    """Prepare for the forecast loop."""
    # Empty arrays for the previous displacements and the forecast cascade
    D = np.stack([None for j in range(n_ens_members)])
    D_Yn = np.stack([None for j in range(n_ens_members)])
    D_pb = np.stack([None for j in range(n_ens_members)])
    R_f = [[] for j in range(n_ens_members)]

    if mask_method == "incremental":
        # get mask parameters
        mask_rim = mask_kwargs.get("mask_rim", 10)
        mask_f = mask_kwargs.get("mask_f", 1.0)
        # initialize the structuring element
        struct = generate_binary_structure(2, 1)
        # iterate it to expand it nxn
        n = mask_f * timestep / kmperpixel
        struct = iterate_structure(struct, int((n - 1) / 2.0))
    else:
        mask_rim, struct = None, None

    if noise_method is None:
        R_m = [R_c[0][i].copy() for i in range(n_cascade_levels)]
    else:
        R_m = None

    fft_objs = []
    for i in range(n_ens_members):
        fft_objs.append(utils.get_method(fft_method, shape=R_c.shape[-2:]))

    return D, D_Yn, D_pb, R_f, R_m, mask_rim, struct, fft_objs


def _compute_initial_nwp_skill(
    R_c, precip_models, domain_mask, issuetime, outdir_path_skill, clim_kwargs
):
    """Calculate the initial skill of the (NWP) model forecasts at t=0."""
    rho_nwp_models = [
        blending.skill_scores.spatial_correlation(
            obs=R_c[0, :, -1, :, :].copy(),
            mod=precip_models[n_model, :, :, :].copy(),
            domain_mask=domain_mask,
        )
        for n_model in range(precip_models.shape[0])
    ]
    rho_nwp_models = np.stack(rho_nwp_models)

    # Ensure that the model skill decreases with increasing scale level.
    for n_model in range(precip_models.shape[0]):
        for i in range(1, precip_models.shape[1]):
            if rho_nwp_models[n_model, i] > rho_nwp_models[n_model, i - 1]:
                # Set it equal to the previous scale level
                rho_nwp_models[n_model, i] = rho_nwp_models[n_model, i - 1]

    # Save this in the climatological skill file
    blending.clim.save_skill(
        current_skill=rho_nwp_models,
        validtime=issuetime,
        outdir_path=outdir_path_skill,
        **clim_kwargs,
    )
    return rho_nwp_models


def _init_noise_cascade(
    shape,
    n_ens_members,
    n_cascade_levels,
    generate_noise,
    decompositor,
    pp,
    randgen_prec,
    fft_objs,
    bp_filter,
    domain,
    noise_method,
    noise_std_coeffs,
    ar_order,
):
    """Initialize the noise cascade with identical noise for all AR(n) steps
    We also need to return the mean and standard deviations of the noise
    for the recombination of the noise before advecting it.
    """
    noise_cascade = np.zeros(shape)
    mu_noise = np.zeros((n_ens_members, n_cascade_levels))
    sigma_noise = np.zeros((n_ens_members, n_cascade_levels))
    if noise_method:
        for j in range(n_ens_members):
            EPS = generate_noise(
                pp, randstate=randgen_prec[j], fft_method=fft_objs[j], domain=domain
            )
            EPS = decompositor(
                EPS,
                bp_filter,
                fft_method=fft_objs[j],
                input_domain=domain,
                output_domain=domain,
                compute_stats=True,
                normalize=True,
                compact_output=True,
            )
            mu_noise[j] = EPS["means"]
            sigma_noise[j] = EPS["stds"]
            for i in range(n_cascade_levels):
                EPS_ = EPS["cascade_levels"][i]
                EPS_ *= noise_std_coeffs[i]
                for n in range(ar_order):
                    noise_cascade[j][i][n] = EPS_
            EPS = None
            EPS_ = None
    return noise_cascade, mu_noise, sigma_noise


def _fill_nans_infs_nwp_cascade(
    precip_models_cascade,
    precip_models,
    precip_cascade,
    precip,
    mu_models,
    sigma_models,
):
    """Ensure that the NWP cascade and fields do no contain any nans or infinite number"""
    # Fill nans and infinite numbers with the minimum value present in precip
    # (corresponding to zero rainfall in the radar observations)
    min_cascade = np.nanmin(precip_cascade)
    min_precip = np.nanmin(precip)
    precip_models_cascade[~np.isfinite(precip_models_cascade)] = min_cascade
    precip_models[~np.isfinite(precip_models)] = min_precip
    # Also set any nans or infs in the mean and sigma of the cascade to
    # respectively 0.0 and 1.0
    mu_models[~np.isfinite(mu_models)] = 0.0
    sigma_models[~np.isfinite(sigma_models)] = 0.0

    return precip_models_cascade, precip_models, mu_models, sigma_models


def _determine_max_nr_rainy_cells_nwp(precip_models, precip_thr, n_models, timesteps):
    """Initialize noise based on the NWP field time step where the fraction of rainy cells is highest"""
    if precip_thr is None:
        precip_thr = np.nanmin(precip_models)

    max_rain_pixels = -1
    max_rain_pixels_j = -1
    max_rain_pixels_t = -1
    for j in range(n_models):
        for t in timesteps:
            rain_pixels = precip_models[j][t][precip_models[j][t] > precip_thr].size
            if rain_pixels > max_rain_pixels:
                max_rain_pixels = rain_pixels
                max_rain_pixels_j = j
                max_rain_pixels_t = t
    precip_noise_input = precip_models[max_rain_pixels_j][max_rain_pixels_t]

    return precip_noise_input.astype(np.float64, copy=False)
