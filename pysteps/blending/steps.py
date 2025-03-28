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
from copy import copy, deepcopy
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
from scipy.linalg import inv
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure

from pysteps import blending, cascade, extrapolation, noise, utils
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.postprocessing import probmatching
from pysteps.timeseries import autoregression, correlation
from pysteps.utils.check_norain import check_norain

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class StepsBlendingConfig:
    """
    Parameters
    ----------

    precip_threshold: float, optional
      Specifies the threshold value for minimum observable precipitation
      intensity. Required if mask_method is not None or conditional is True.
    norain_threshold: float
      Specifies the threshold value for the fraction of rainy (see above) pixels
      in the radar rainfall field below which we consider there to be no rain.
      Depends on the amount of clutter typically present.
      Standard set to 0.0
    kmperpixel: float, optional
      Spatial resolution of the input data (kilometers/pixel). Required if
      vel_pert_method is not None or mask_method is 'incremental'.
    timestep: float
      Time step of the motion vectors (minutes). Required if vel_pert_method is
      not None or mask_method is 'incremental'.
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
    extrapolation_method: str, optional
      Name of the extrapolation method to use. See the documentation of
      :py:mod:`pysteps.extrapolation.interface`.
    decomposition_method: {'fft'}, optional
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
    velocity_perturbation_method: {'bps',None}, optional
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
    extrapolation_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the extrapolation
      method. See the documentation of :py:func:`pysteps.extrapolation.interface`.
    filter_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the filter method.
      See the documentation of :py:mod:`pysteps.cascade.bandpass_filters`.
    noise_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the initializer of
      the noise generator. See the documentation of :py:mod:`pysteps.noise.fftgenerators`.
    velocity_perturbation_kwargs: dict, optional
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

      The above parameters have been fitted by using run_vel_pert_analysis.py
      and fit_vel_pert_params.py located in the scripts directory.

      See :py:mod:`pysteps.noise.motion` for additional documentation.
    climatology_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the climatological
      skill file. Arguments can consist of: 'outdir_path', 'n_models'
      (the number of NWP models) and 'window_length' (the minimum number of
      days the clim file should have, otherwise the default is used).
    mask_kwargs: dict
      Optional dictionary containing mask keyword arguments 'mask_f',
      'mask_rim' and 'max_mask_rim', the factor defining the the mask
      increment and the (maximum) rim size, respectively.
      The mask increment is defined as mask_f*timestep/kmperpixel.
    measure_time: bool
      If set to True, measure, print and return the computation time.
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
    """

    precip_threshold: float | None
    norain_threshold: float
    kmperpixel: float
    timestep: float
    n_ens_members: int
    n_cascade_levels: int
    blend_nwp_members: bool
    extrapolation_method: str
    decomposition_method: str
    bandpass_filter_method: str
    noise_method: str | None
    noise_stddev_adj: str | None
    ar_order: int
    velocity_perturbation_method: str | None
    weights_method: str
    conditional: bool
    probmatching_method: str | None
    mask_method: str | None
    resample_distribution: bool
    smooth_radar_mask_range: int
    seed: int | None
    num_workers: int
    fft_method: str
    domain: str
    outdir_path_skill: str
    extrapolation_kwargs: dict[str, Any] = field(default_factory=dict)
    filter_kwargs: dict[str, Any] = field(default_factory=dict)
    noise_kwargs: dict[str, Any] = field(default_factory=dict)
    velocity_perturbation_kwargs: dict[str, Any] = field(default_factory=dict)
    climatology_kwargs: dict[str, Any] = field(default_factory=dict)
    mask_kwargs: dict[str, Any] = field(default_factory=dict)
    measure_time: bool = False
    callback: Any | None = None
    return_output: bool = True


@dataclass
class StepsBlendingParams:
    noise_std_coeffs: np.ndarray | None = None
    bandpass_filter: Any | None = None
    fft: Any | None = None
    perturbation_generator: Callable[..., np.ndarray] | None = None
    noise_generator: Callable[..., np.ndarray] | None = None
    PHI: np.ndarray | None = None
    extrapolation_method: Callable[..., Any] | None = None
    decomposition_method: Callable[..., dict] | None = None
    recomposition_method: Callable[..., np.ndarray] | None = None
    velocity_perturbations: Any | None = None
    generate_velocity_noise: Callable[[Any, float], np.ndarray] | None = None
    velocity_perturbations_parallel: np.ndarray | None = None
    velocity_perturbations_perpendicular: np.ndarray | None = None
    fft_objs: list[Any] = field(default_factory=list)
    mask_rim: int | None = None
    struct: np.ndarray | None = None
    time_steps_is_list: bool = False
    precip_models_provided_is_cascade: bool = False
    xy_coordinates: np.ndarray | None = None
    precip_zerovalue: float | None = None
    precip_threshold: float | None = None
    mask_threshold: np.ndarray | None = None
    zero_precip_radar: bool = False
    zero_precip_model_fields: bool = False
    original_timesteps: list | np.ndarray | None = None
    num_ensemble_workers: int | None = None
    rho_nwp_models: np.ndarray | None = None
    domain_mask: np.ndarray | None = None
    filter_kwargs: dict | None = None
    noise_kwargs: dict | None = None
    velocity_perturbation_kwargs: dict | None = None
    climatology_kwargs: dict | None = None
    mask_kwargs: dict | None = None


@dataclass
class StepsBlendingState:
    # Radar and noise states
    precip_cascades: np.ndarray | None = None
    precip_noise_input: np.ndarray | None = None
    precip_noise_cascades: np.ndarray | None = None
    precip_mean_noise: np.ndarray | None = None
    precip_std_noise: np.ndarray | None = None

    # Extrapolation states
    mean_extrapolation: np.ndarray | None = None
    std_extrapolation: np.ndarray | None = None
    rho_extrap_cascade_prev: np.ndarray | None = None
    rho_extrap_cascade: np.ndarray | None = None
    precip_cascades_prev_subtimestep: np.ndarray | None = None
    cascade_noise_prev_subtimestep: np.ndarray | None = None
    precip_extrapolated_after_decomp: np.ndarray | None = None
    noise_extrapolated_after_decomp: np.ndarray | None = None
    precip_extrapolated_probability_matching: np.ndarray | None = None

    # NWP model states
    precip_models_cascades: np.ndarray | None = None
    precip_models_cascades_timestep: np.ndarray | None = None
    precip_models_timestep: np.ndarray | None = None
    mean_models_timestep: np.ndarray | None = None
    std_models_timestep: np.ndarray | None = None
    velocity_models_timestep: np.ndarray | None = None

    # Mapping from NWP members to ensemble members
    mapping_list_NWP_member_to_ensemble_member: np.ndarray | None = None

    # Random states for precipitation, motion and probmatching
    randgen_precip: list[np.random.RandomState] | None = None
    randgen_motion: list[np.random.RandomState] | None = None
    randgen_probmatching: list[np.random.RandomState] | None = None

    # Variables for final forecast computation
    previous_displacement: list[Any] | None = None
    previous_displacement_noise_cascade: list[Any] | None = None
    previous_displacement_prob_matching: list[Any] | None = None
    rho_final_blended_forecast: np.ndarray | None = None
    final_blended_forecast_means: np.ndarray | None = None
    final_blended_forecast_stds: np.ndarray | None = None
    final_blended_forecast_means_mod_only: np.ndarray | None = None
    final_blended_forecast_stds_mod_only: np.ndarray | None = None
    final_blended_forecast_cascades: np.ndarray | None = None
    final_blended_forecast_cascades_mod_only: np.ndarray | None = None
    final_blended_forecast_recomposed: np.ndarray | None = None
    final_blended_forecast_recomposed_mod_only: np.ndarray | None = None

    # Final outputs
    final_blended_forecast: np.ndarray | None = None
    final_blended_forecast_non_perturbed: np.ndarray | None = None

    # Timing and indexing
    time_prev_timestep: list[float] | None = None
    leadtime_since_start_forecast: list[float] | None = None
    subtimesteps: list[float] | None = None
    is_nowcast_time_step: bool | None = None
    subtimestep_index: int | None = None

    # Weights used for blending
    weights: np.ndarray | None = None
    weights_model_only: np.ndarray | None = None

    # This is stores here as well because this is changed during the forecast loop and thus no longer part of the config
    extrapolation_kwargs: dict[str, Any] = field(default_factory=dict)


class StepsBlendingNowcaster:
    def __init__(
        self,
        precip,
        precip_models,
        velocity,
        velocity_models,
        time_steps,
        issue_time,
        steps_blending_config: StepsBlendingConfig,
    ):
        """Initializes the StepsBlendingNowcaster with inputs and configurations."""
        # Store inputs
        self.__precip = precip
        self.__precip_models = precip_models
        self.__velocity = velocity
        self.__velocity_models = velocity_models
        self.__timesteps = time_steps
        self.__issuetime = issue_time

        self.__config = steps_blending_config

        # Initialize Params and State
        self.__params = StepsBlendingParams()
        self.__state = StepsBlendingState()

        # Additional variables for time measurement
        self.__start_time_init = None
        self.__init_time = None
        self.__mainloop_time = None

    def compute_forecast(self):
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
        time_steps: int or list of floats
          Number of time steps to forecast or a list of time steps for which the
          forecasts are computed (relative to the input time step). The elements of
          the list are required to be in ascending order.
        issue_time: datetime
          is issued.
        config: StepsBlendingConfig
            Provides a set of configuration parameters for the nowcast ensemble generation.

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

        self.__check_inputs()
        self.__print_forecast_info()
        # Measure time for initialization
        if self.__config.measure_time:
            self.__start_time_init = time.time()

        # Slice the precipitation field to only use the last ar_order + 1 fields
        self.__precip = self.__precip[-(self.__config.ar_order + 1) :, :, :].copy()
        self.__initialize_nowcast_components()
        self.__prepare_radar_and_NWP_fields()

        # Determine if rain is present in both radar and NWP fields
        if self.__params.zero_precip_radar and self.__params.zero_precip_model_fields:
            return self.__zero_precipitation_forecast()
        else:
            # Prepare the data for the zero precipitation radar case and initialize the noise correctly
            if self.__params.zero_precip_radar:
                self.__prepare_nowcast_for_zero_radar()
            else:
                self.__state.precip_noise_input = self.__precip.copy()
            self.__initialize_noise()
            self.__estimate_ar_parameters_radar()
            self.__multiply_precip_cascade_to_match_ensemble_members()
            self.__initialize_random_generators()
            self.__prepare_forecast_loop()
            self.__initialize_noise_cascades()
            if self.__config.measure_time:
                self.__init_time = self.__measure_time(
                    "initialization", self.__start_time_init
                )

            self.__blended_nowcast_main_loop()
            # Stack and return the forecast output
            if self.__config.return_output:
                self.__state.final_blended_forecast = np.stack(
                    [
                        np.stack(self.__state.final_blended_forecast[j])
                        for j in range(self.__config.n_ens_members)
                    ]
                )
                if self.__config.measure_time:
                    return (
                        self.__state.final_blended_forecast,
                        self.__init_time,
                        self.__mainloop_time,
                    )
                else:
                    return self.__state.final_blended_forecast
            else:
                return None

    def __blended_nowcast_main_loop(self):
        """
        Main nowcast loop that iterates through the ensemble members and time steps
        to generate forecasts.
        """
        # Isolate the last time slice of observed precipitation
        self.__precip = self.__precip[-1, :, :]
        print("Starting blended nowcast computation.")

        if self.__config.measure_time:
            starttime_mainloop = time.time()
        self.__state.extrapolation_kwargs["return_displacement"] = True

        self.__state.precip_cascades_prev_subtimestep = deepcopy(
            self.__state.precip_cascades
        )
        self.__state.cascade_noise_prev_subtimestep = deepcopy(
            self.__state.precip_noise_cascades
        )

        self.__state.time_prev_timestep = [
            0.0 for j in range(self.__config.n_ens_members)
        ]
        self.__state.leadtime_since_start_forecast = [
            0.0 for j in range(self.__config.n_ens_members)
        ]

        # iterate each time step
        for t, subtimestep_idx in enumerate(self.__timesteps):
            self.__determine_subtimesteps_and_nowcast_time_step(t, subtimestep_idx)
            if self.__config.measure_time:
                starttime = time.time()
            self.__decompose_nwp_if_needed_and_fill_nans_in_nwp(t)
            self.__find_nowcast_NWP_combination(t)
            self.__determine_skill_for_current_timestep(t)
            # the nowcast iteration for each ensemble member
            final_blended_forecast_all_members_one_timestep = [
                None for _ in range(self.__config.n_ens_members)
            ]

            def worker(j):
                worker_state = copy(self.__state)
                self.__determine_NWP_skill_for_next_timestep(t, j, worker_state)
                self.__determine_weights_per_component(worker_state)
                self.__regress_extrapolation_and_noise_cascades(j, worker_state)
                self.__perturb_blend_and_advect_extrapolation_and_noise_to_current_timestep(
                    t, j, worker_state
                )
                # 8.5 Blend the cascades
                final_blended_forecast_single_member = []
                for t_sub in self.__state.subtimesteps:
                    # TODO: does it make sense to use sub time steps - check if it works?
                    if t_sub > 0:
                        self.__blend_cascades(t_sub, j, worker_state)
                        self.__recompose_cascade_to_rainfall_field(j, worker_state)
                        final_blended_forecast_single_member = (
                            self.__post_process_output(
                                j,
                                t_sub,
                                final_blended_forecast_single_member,
                                worker_state,
                            )
                        )
                    final_blended_forecast_all_members_one_timestep[j] = (
                        final_blended_forecast_single_member
                    )

            dask_worker_collection = []

            if DASK_IMPORTED and self.__config.n_ens_members > 1:
                for j in range(self.__config.n_ens_members):
                    dask_worker_collection.append(dask.delayed(worker)(j))
                dask.compute(
                    *dask_worker_collection,
                    num_workers=self.__params.num_ensemble_workers,
                )
            else:
                for j in range(self.__config.n_ens_members):
                    worker(j)

            dask_worker_collection = None

            if self.__state.is_nowcast_time_step:
                if self.__config.measure_time:
                    _ = self.__measure_time("subtimestep", starttime)
                else:
                    print("done.")

            if self.__config.callback is not None:
                precip_forecast_final = np.stack(
                    final_blended_forecast_all_members_one_timestep
                )
                if precip_forecast_final.shape[1] > 0:
                    self.__config.callback(precip_forecast_final.squeeze())

            if self.__config.return_output:
                for j in range(self.__config.n_ens_members):
                    self.__state.final_blended_forecast[j].extend(
                        final_blended_forecast_all_members_one_timestep[j]
                    )

            final_blended_forecast_all_members_one_timestep = None
        if self.__config.measure_time:
            self.__mainloop_time = time.time() - starttime_mainloop

    def __check_inputs(self):
        """
        Validates the inputs and determines if the user provided raw forecasts or decomposed forecasts.
        """
        # Check dimensions of precip
        if self.__precip.ndim != 3:
            raise ValueError(
                "precip must be a three-dimensional array of shape (ar_order + 1, m, n)"
            )
        if self.__precip.shape[0] < self.__config.ar_order + 1:
            raise ValueError(
                f"precip must have at least {self.__config.ar_order + 1} time steps in the first dimension "
                f"to match the autoregressive order (ar_order={self.__config.ar_order})"
            )

        # Check dimensions of velocity
        if self.__velocity.ndim != 3:
            raise ValueError(
                "velocity must be a three-dimensional array of shape (2, m, n)"
            )
        if self.__velocity_models.ndim != 5:
            raise ValueError(
                "velocity_models must be a five-dimensional array of shape (n_models, timestep, 2, m, n)"
            )
        if self.__velocity.shape[0] != 2 or self.__velocity_models.shape[2] != 2:
            raise ValueError(
                "velocity and velocity_models must have an x- and y-component, check the shape"
            )

        # Check that spatial dimensions match between precip and velocity
        if self.__precip.shape[1:3] != self.__velocity.shape[1:3]:
            raise ValueError(
                f"Spatial dimensions of precip and velocity do not match: "
                f"{self.__precip.shape[1:3]} vs {self.__velocity.shape[1:3]}"
            )
        # Check if the number of members in the precipitation models and velocity models match
        if self.__precip_models.shape[0] != self.__velocity_models.shape[0]:
            raise ValueError(
                "The number of members in the precipitation models and velocity models must match"
            )

        if isinstance(self.__timesteps, list):
            self.__params.time_steps_is_list = True
            if not sorted(self.__timesteps) == self.__timesteps:
                raise ValueError(
                    "timesteps is not in ascending order", self.__timesteps
                )
            if self.__precip_models.shape[1] != math.ceil(self.__timesteps[-1]) + 1:
                raise ValueError(
                    "precip_models does not contain sufficient lead times for this forecast"
                )
            self.__params.original_timesteps = [0] + list(self.__timesteps)
            self.__timesteps = nowcast_utils.binned_timesteps(
                self.__params.original_timesteps
            )
        else:
            self.__params.time_steps_is_list = False
            if self.__precip_models.shape[1] != self.__timesteps + 1:
                raise ValueError(
                    "precip_models does not contain sufficient lead times for this forecast"
                )
            self.__timesteps = list(range(self.__timesteps + 1))

        precip_nwp_dim = self.__precip_models.ndim
        if precip_nwp_dim == 2:
            if isinstance(self.__precip_models[0][0], dict):
                # It's a 2D array of dictionaries with decomposed cascades
                self.__params.precip_models_provided_is_cascade = True
            else:
                raise ValueError(
                    "When precip_models has ndim == 2, it must contain dictionaries with decomposed cascades."
                )
        elif precip_nwp_dim == 4:
            self.__params.precip_models_provided_is_cascade = False
        else:
            raise ValueError(
                "precip_models must be either a two-dimensional array containing dictionaries with decomposed model fields"
                "or a four-dimensional array containing the original (NWP) model forecasts"
            )

        if self.__config.extrapolation_kwargs is None:
            self.__state.extrapolation_kwargs = dict()
        else:
            self.__state.extrapolation_kwargs = deepcopy(
                self.__config.extrapolation_kwargs
            )

        if self.__config.filter_kwargs is None:
            self.__params.filter_kwargs = dict()
        else:
            self.__params.filter_kwargs = deepcopy(self.__config.filter_kwargs)

        if self.__config.noise_kwargs is None:
            self.__params.noise_kwargs = {"win_fun": "tukey"}
        else:
            self.__params.noise_kwargs = deepcopy(self.__config.noise_kwargs)

        if self.__config.velocity_perturbation_kwargs is None:
            self.__params.velocity_perturbation_kwargs = dict()
        else:
            self.__params.velocity_perturbation_kwargs = deepcopy(
                self.__config.velocity_perturbation_kwargs
            )

        if self.__config.climatology_kwargs is None:
            # Make sure clim_kwargs at least contains the number of models
            self.__params.climatology_kwargs = dict(
                {"n_models": self.__precip_models.shape[0]}
            )
        else:
            self.__params.climatology_kwargs = deepcopy(
                self.__config.climatology_kwargs
            )

        if self.__config.mask_kwargs is None:
            self.__params.mask_kwargs = dict()
        else:
            self.__params.mask_kwargs = deepcopy(self.__config.mask_kwargs)

        self.__params.precip_threshold = self.__config.precip_threshold

        if np.any(~np.isfinite(self.__velocity)):
            raise ValueError("velocity contains non-finite values")

        if self.__config.mask_method not in ["obs", "incremental", None]:
            raise ValueError(
                "unknown mask method %s: must be 'obs', 'incremental' or None"
                % self.__config.mask_method
            )

        if self.__config.conditional and self.__params.precip_threshold is None:
            raise ValueError("conditional=True but precip_thr is not set")

        if (
            self.__config.mask_method is not None
            and self.__params.precip_threshold is None
        ):
            raise ValueError("mask_method!=None but precip_thr=None")

        if self.__config.noise_stddev_adj not in ["auto", "fixed", None]:
            raise ValueError(
                "unknown noise_std_dev_adj method %s: must be 'auto', 'fixed', or None"
                % self.__config.noise_stddev_adj
            )

        if self.__config.kmperpixel is None:
            if self.__config.velocity_perturbation_method is not None:
                raise ValueError(
                    "velocity_perturbation_method is set but kmperpixel=None"
                )
            if self.__config.mask_method == "incremental":
                raise ValueError("mask_method='incremental' but kmperpixel=None")

        if self.__config.timestep is None:
            if self.__config.velocity_perturbation_method is not None:
                raise ValueError(
                    "velocity_perturbation_method is set but timestep=None"
                )
            if self.__config.mask_method == "incremental":
                raise ValueError("mask_method='incremental' but timestep=None")

    def __print_forecast_info(self):
        """
        Print information about the forecast setup, including inputs, methods, and parameters.
        """
        print("STEPS blending")
        print("==============")
        print("")

        print("Inputs")
        print("------")
        print(f"forecast issue time:         {self.__issuetime.isoformat()}")
        print(
            f"input dimensions:            {self.__precip.shape[1]}x{self.__precip.shape[2]}"
        )
        if self.__config.kmperpixel is not None:
            print(f"km/pixel:                    {self.__config.kmperpixel}")
        if self.__config.timestep is not None:
            print(f"time step:                   {self.__config.timestep} minutes")
        print("")

        print("NWP and blending inputs")
        print("-----------------------")
        print(f"number of (NWP) models:      {self.__precip_models.shape[0]}")
        print(f"blend (NWP) model members:   {self.__config.blend_nwp_members}")
        print(
            f"decompose (NWP) models:      {'yes' if self.__precip_models.ndim == 4 else 'no'}"
        )
        print("")

        print("Methods")
        print("-------")
        print(f"extrapolation:               {self.__config.extrapolation_method}")
        print(f"bandpass filter:             {self.__config.bandpass_filter_method}")
        print(f"decomposition:               {self.__config.decomposition_method}")
        print(f"noise generator:             {self.__config.noise_method}")
        print(
            f"noise adjustment:            {'yes' if self.__config.noise_stddev_adj else 'no'}"
        )
        print(
            f"velocity perturbator:        {self.__config.velocity_perturbation_method}"
        )
        print(f"blending weights method:     {self.__config.weights_method}")
        print(
            f"conditional statistics:      {'yes' if self.__config.conditional else 'no'}"
        )
        print(f"precip. mask method:         {self.__config.mask_method}")
        print(f"probability matching:        {self.__config.probmatching_method}")
        print(f"FFT method:                  {self.__config.fft_method}")
        print(f"domain:                      {self.__config.domain}")
        print("")

        print("Parameters")
        print("----------")
        if isinstance(self.__timesteps, int):
            print(f"number of time steps:        {self.__timesteps}")
        else:
            print(f"time steps:                  {self.__timesteps}")
        print(f"ensemble size:               {self.__config.n_ens_members}")
        print(f"parallel threads:            {self.__config.num_workers}")
        print(f"number of cascade levels:    {self.__config.n_cascade_levels}")
        print(f"order of the AR(p) model:    {self.__config.ar_order}")
        if self.__config.velocity_perturbation_method == "bps":
            self.__params.velocity_perturbations_parallel = (
                self.__params.velocity_perturbation_kwargs.get(
                    "p_par", noise.motion.get_default_params_bps_par()
                )
            )
            self.__params.velocity_perturbations_perpendicular = (
                self.__params.velocity_perturbation_kwargs.get(
                    "p_perp", noise.motion.get_default_params_bps_perp()
                )
            )
            print(
                f"vel. pert., parallel:        {self.__params.velocity_perturbations_parallel[0]},{self.__params.velocity_perturbations_parallel[1]},{self.__params.velocity_perturbations_parallel[2]}"
            )
            print(
                f"vel. pert., perpendicular:   {self.__params.velocity_perturbations_perpendicular[0]},{self.__params.velocity_perturbations_perpendicular[1]},{self.__params.velocity_perturbations_perpendicular[2]}"
            )
        else:
            (
                self.__params.velocity_perturbations_parallel,
                self.__params.velocity_perturbations_perpendicular,
            ) = (None, None)

        if self.__config.conditional or self.__config.mask_method is not None:
            print(f"precip. intensity threshold: {self.__params.precip_threshold}")
        print(f"no-rain fraction threshold for radar: {self.__config.norain_threshold}")
        print("")

    def __initialize_nowcast_components(self):
        """
        Initialize the FFT, bandpass filters, decomposition methods, and extrapolation method.
        """
        # Initialize number of ensemble workers
        self.__params.num_ensemble_workers = min(
            self.__config.n_ens_members, self.__config.num_workers
        )

        M, N = self.__precip.shape[1:]  # Extract the spatial dimensions (height, width)

        # Initialize FFT method
        self.__params.fft = utils.get_method(
            self.__config.fft_method, shape=(M, N), n_threads=self.__config.num_workers
        )

        # Initialize the band-pass filter for the cascade decomposition
        filter_method = cascade.get_method(self.__config.bandpass_filter_method)
        self.__params.bandpass_filter = filter_method(
            (M, N),
            self.__config.n_cascade_levels,
            **(self.__params.filter_kwargs or {}),
        )

        # Get the decomposition method (e.g., FFT)
        (
            self.__params.decomposition_method,
            self.__params.recomposition_method,
        ) = cascade.get_method(self.__config.decomposition_method)

        # Get the extrapolation method (e.g., semilagrangian)
        self.__params.extrapolation_method = extrapolation.get_method(
            self.__config.extrapolation_method
        )

        # Generate the mesh grid for spatial coordinates
        x_values, y_values = np.meshgrid(np.arange(N), np.arange(M))
        self.__params.xy_coordinates = np.stack([x_values, y_values])

        self.__precip = self.__precip[-(self.__config.ar_order + 1) :, :, :].copy()
        # Determine the domain mask from non-finite values in the precipitation data
        self.__params.domain_mask = np.logical_or.reduce(
            [~np.isfinite(self.__precip[i, :]) for i in range(self.__precip.shape[0])]
        )

        print("Blended nowcast components initialized successfully.")

    def __prepare_radar_and_NWP_fields(self):
        """
        Prepare radar and NWP precipitation fields for nowcasting.
        This includes generating a threshold mask, transforming fields into
        Lagrangian coordinates, cascade decomposing/recomposing, and checking
        for zero-precip areas. The results are stored in class attributes.
        """
        # determine the precipitation threshold mask
        if self.__config.conditional:
            # TODO: is this logical_and correct here? Now only those places where precip is in all images is saved?
            self.__params.mask_threshold = np.logical_and.reduce(
                [
                    self.__precip[i, :, :] >= self.__params.precip_threshold
                    for i in range(self.__precip.shape[0])
                ]
            )
        else:
            self.__params.mask_threshold = None

        # we need to know the zerovalue of precip to replace the mask when decomposing after
        # extrapolation
        self.__params.precip_zerovalue = np.nanmin(self.__precip)

        # 1. Start with the radar rainfall fields. We want the fields in a
        # Lagrangian space

        # Advect the previous precipitation fields to the same position with the
        # most recent one (i.e. transform them into the Lagrangian coordinates).

        self.__state.extrapolation_kwargs["xy_coords"] = self.__params.xy_coordinates
        res = []

        def transform_to_lagrangian(precip, i):
            return self.__params.extrapolation_method(
                precip[i, :, :],
                self.__velocity,
                self.__config.ar_order - i,
                "min",
                allow_nonfinite_values=True,
                **self.__state.extrapolation_kwargs.copy(),
            )[-1]

        if not DASK_IMPORTED:
            # Process each earlier precipitation field directly
            for i in range(self.__config.ar_order):
                self.__precip[i, :, :] = transform_to_lagrangian(self.__precip, i)
        else:
            # Use Dask delayed for parallelization if DASK_IMPORTED is True
            for i in range(self.__config.ar_order):
                res.append(dask.delayed(transform_to_lagrangian)(self.__precip, i))
            num_workers_ = (
                len(res)
                if self.__config.num_workers > len(res)
                else self.__config.num_workers
            )
            self.__precip = np.stack(
                list(dask.compute(*res, num_workers=num_workers_))
                + [self.__precip[-1, :, :]]
            )

        # Replace non-finite values with the minimum value for each field
        self.__precip = self.__precip.copy()
        for i in range(self.__precip.shape[0]):
            self.__precip[i, ~np.isfinite(self.__precip[i, :])] = np.nanmin(
                self.__precip[i, :]
            )

        # Perform the cascade decomposition for the input precip fields and,
        # if necessary, for the (NWP) model fields
        # Compute the cascade decompositions of the input precipitation fields
        precip_forecast_decomp = []
        for i in range(self.__config.ar_order + 1):
            precip_forecast = self.__params.decomposition_method(
                self.__precip[i, :, :],
                self.__params.bandpass_filter,
                mask=self.__params.mask_threshold,
                fft_method=self.__params.fft,
                output_domain=self.__config.domain,
                normalize=True,
                compute_stats=True,
                compact_output=True,
            )
            precip_forecast_decomp.append(precip_forecast)

        # Rearrange the cascaded into a four-dimensional array of shape
        # (n_cascade_levels,ar_order+1,m,n) for the autoregressive model
        self.__state.precip_cascades = nowcast_utils.stack_cascades(
            precip_forecast_decomp, self.__config.n_cascade_levels
        )

        precip_forecast_decomp = precip_forecast_decomp[-1]
        self.__state.mean_extrapolation = np.array(precip_forecast_decomp["means"])
        self.__state.std_extrapolation = np.array(precip_forecast_decomp["stds"])

        # If necessary, recompose (NWP) model forecasts
        self.__state.precip_models_cascades = None

        if self.__params.precip_models_provided_is_cascade:
            self.__state.precip_models_cascades = self.__precip_models
            # Inline logic of _compute_cascade_recomposition_nwp
            temp_precip_models = []
            for i in range(self.__precip_models.shape[0]):
                precip_model = []
                for time_step in range(self.__precip_models.shape[1]):
                    # Use the recomposition method to rebuild the rainfall fields
                    recomposed = self.__params.recomposition_method(
                        self.__precip_models[i, time_step]
                    )
                    precip_model.append(recomposed)
                temp_precip_models.append(precip_model)

            self.__precip_models = np.stack(temp_precip_models)

        # Check for zero input fields in the radar and NWP data.
        self.__params.zero_precip_radar = check_norain(
            self.__precip,
            self.__params.precip_threshold,
            self.__config.norain_threshold,
            self.__params.noise_kwargs["win_fun"],
        )
        # The norain fraction threshold used for nwp is the default value of 0.0,
        # since nwp does not suffer from clutter.
        self.__params.zero_precip_model_fields = check_norain(
            self.__precip_models,
            self.__params.precip_threshold,
            self.__config.norain_threshold,
            self.__params.noise_kwargs["win_fun"],
        )

    def __zero_precipitation_forecast(self):
        """
        Generate a zero-precipitation forecast (filled with the minimum precip value)
        when no precipitation above the threshold is detected. The forecast is
        optionally returned or passed to a callback.
        """
        print(
            "No precipitation above the threshold found in both the radar and NWP fields"
        )
        print("The resulting forecast will contain only zeros")
        # Create the output list
        precip_forecast = [[] for j in range(self.__config.n_ens_members)]

        # Save per time step to ensure the array does not become too large if
        # no return_output is requested and callback is not None.
        for t, subtimestep_idx in enumerate(self.__timesteps):
            # If the timestep is not the first one, we need to provide the zero forecast
            if t > 0:
                # Create an empty np array with shape [n_ens_members, rows, cols]
                # and fill it with the minimum value from precip (corresponding to
                # zero precipitation)
                N, M = self.__precip.shape[1:]
                precip_forecast_workers = np.full(
                    (self.__config.n_ens_members, N, M), self.__params.precip_zerovalue
                )
                if subtimestep_idx:
                    if self.__config.callback is not None:
                        if precip_forecast_workers.shape[1] > 0:
                            self.__config.callback(precip_forecast_workers.squeeze())
                    if self.__config.return_output:
                        for j in range(self.__config.n_ens_members):
                            precip_forecast[j].append(precip_forecast_workers[j])
                precip_forecast_workers = None

        if self.__config.measure_time:
            zero_precip_time = time.time() - self.__start_time_init

        if self.__config.return_output:
            precip_forecast_all_members_all_times = np.stack(
                [
                    np.stack(precip_forecast[j])
                    for j in range(self.__config.n_ens_members)
                ]
            )

            if self.__config.measure_time:
                return (
                    precip_forecast_all_members_all_times,
                    zero_precip_time,
                    zero_precip_time,
                )
            else:
                return precip_forecast_all_members_all_times
        else:
            return None

    def __prepare_nowcast_for_zero_radar(self):
        """
        Handle the case when radar fields indicate zero precipitation. This method
        updates the cascade with NWP data, uses the NWP velocity field, and
        initializes the noise model based on the time step with the most rain.
        """
        # If zero_precip_radar is True, only use the velocity field of the NWP
        # forecast. I.e., velocity (radar) equals velocity_model at the first time
        # step.
        # Use the velocity from velocity_models at time step 0
        self.__velocity = self.__velocity_models[:, 0, :, :, :].astype(
            np.float64, copy=False
        )
        # Take the average over the first axis, which corresponds to n_models
        # (hence, the model average)
        self.__velocity = np.mean(self.__velocity, axis=0)

        # Initialize the noise method.
        # If zero_precip_radar is True, initialize noise based on the NWP field time
        # step where the fraction of rainy cells is highest (because other lead times
        # might be zero as well). Else, initialize the noise with the radar
        # rainfall data
        # Initialize noise based on the NWP field time step where the fraction of rainy cells is highest
        if self.__params.precip_threshold is None:
            self.__params.precip_threshold = np.nanmin(self.__precip_models)

        max_rain_pixels = -1
        max_rain_pixels_j = -1
        max_rain_pixels_t = -1
        for j in range(self.__precip_models.shape[0]):
            for t in self.__timesteps:
                rain_pixels = self.__precip_models[j][t][
                    self.__precip_models[j][t] > self.__params.precip_threshold
                ].size
                if rain_pixels > max_rain_pixels:
                    max_rain_pixels = rain_pixels
                    max_rain_pixels_j = j
                    max_rain_pixels_t = t
        self.__state.precip_noise_input = self.__precip_models[max_rain_pixels_j][
            max_rain_pixels_t
        ]
        self.__state.precip_noise_input = self.__state.precip_noise_input.astype(
            np.float64, copy=False
        )

        # If zero_precip_radar, make sure that precip_cascade does not contain
        # only nans or infs. If so, fill it with the zero value.
        if self.__state.precip_models_cascades is not None:
            self.__state.precip_cascades[~np.isfinite(self.__state.precip_cascades)] = (
                np.nanmin(
                    self.__state.precip_models_cascades[
                        max_rain_pixels_j, max_rain_pixels_t
                    ]["cascade_levels"]
                )
            )
        else:
            precip_models_cascade_timestep = self.__params.decomposition_method(
                self.__precip_models[max_rain_pixels_j, max_rain_pixels_t, :, :],
                bp_filter=self.__params.bandpass_filter,
                fft_method=self.__params.fft,
                output_domain=self.__config.domain,
                normalize=True,
                compute_stats=True,
                compact_output=True,
            )["cascade_levels"]
            self.__state.precip_cascades[~np.isfinite(self.__state.precip_cascades)] = (
                np.nanmin(precip_models_cascade_timestep)
            )

        # Make sure precip_noise_input is three-dimensional
        if len(self.__state.precip_noise_input.shape) != 3:
            self.__state.precip_noise_input = self.__state.precip_noise_input[
                np.newaxis, :, :
            ]

    def __initialize_noise(self):
        """
        Initialize noise-based perturbations if configured, computing any required
        adjustment coefficients and setting up the perturbation generator.
        """
        if self.__config.noise_method is not None:
            # get methods for perturbations
            init_noise, self.__params.noise_generator = noise.get_method(
                self.__config.noise_method
            )

            # initialize the perturbation generator for the precipitation field
            self.__params.perturbation_generator = init_noise(
                self.__state.precip_noise_input,
                fft_method=self.__params.fft,
                **self.__params.noise_kwargs,
            )

            if self.__config.noise_stddev_adj == "auto":
                print("Computing noise adjustment coefficients... ", end="", flush=True)
                if self.__config.measure_time:
                    starttime = time.time()

                precip_forecast_min = np.min(self.__state.precip_noise_input)
                self.__params.noise_std_coeffs = noise.utils.compute_noise_stddev_adjs(
                    self.__state.precip_noise_input[-1, :, :],
                    self.__params.precip_threshold,
                    precip_forecast_min,
                    self.__params.bandpass_filter,
                    self.__params.decomposition_method,
                    self.__params.perturbation_generator,
                    self.__params.noise_generator,
                    20,
                    conditional=True,
                    num_workers=self.__config.num_workers,
                    seed=self.__config.seed,
                )

                if self.__config.measure_time:
                    _ = self.__measure_time("Initialize noise", starttime)
                else:
                    print("done.")
            elif self.__config.noise_stddev_adj == "fixed":
                f = lambda k: 1.0 / (0.75 + 0.09 * k)
                self.__params.noise_std_coeffs = [
                    f(k) for k in range(1, self.__config.n_cascade_levels + 1)
                ]
            else:
                self.__params.noise_std_coeffs = np.ones(self.__config.n_cascade_levels)

            if self.__config.noise_stddev_adj is not None:
                print(f"noise std. dev. coeffs:   {self.__params.noise_std_coeffs}")

        else:
            self.__params.perturbation_generator = None
            self.__params.noise_generator = None
            self.__params.noise_std_coeffs = None

    def __estimate_ar_parameters_radar(self):
        """
        Estimate autoregressive (AR) parameters for the radar rainfall field. If
        precipitation exists, compute temporal auto-correlations; otherwise, use
        predefined climatological values. Adjust coefficients if necessary and
        estimate AR model parameters.
        """

        # If there are values in the radar fields, compute the auto-correlations
        GAMMA = np.empty((self.__config.n_cascade_levels, self.__config.ar_order))
        if not self.__params.zero_precip_radar:
            # compute lag-l temporal auto-correlation coefficients for each cascade level
            for i in range(self.__config.n_cascade_levels):
                GAMMA[i, :] = correlation.temporal_autocorrelation(
                    self.__state.precip_cascades[i], mask=self.__params.mask_threshold
                )

        # Else, use standard values for the auto-correlations
        else:
            # Get the climatological lag-1 and lag-2 auto-correlation values from Table 2
            # in `BPS2004`.
            # Hard coded, change to own (climatological) values when present.
            # TODO: add user warning here so users can be aware of this without reading the code?
            GAMMA = np.array(
                [
                    [0.99805, 0.9925, 0.9776, 0.9297, 0.796, 0.482, 0.079, 0.0006],
                    [0.9933, 0.9752, 0.923, 0.750, 0.367, 0.069, 0.0018, 0.0014],
                ]
            )

            # Check whether the number of cascade_levels is correct
            if GAMMA.shape[1] > self.__config.n_cascade_levels:
                GAMMA = GAMMA[:, 0 : self.__config.n_cascade_levels]
            elif GAMMA.shape[1] < self.__config.n_cascade_levels:
                # Get the number of cascade levels that is missing
                n_extra_lev = self.__config.n_cascade_levels - GAMMA.shape[1]
                # Append the array with correlation values of 10e-4
                GAMMA = np.append(
                    GAMMA,
                    [np.repeat(0.0006, n_extra_lev), np.repeat(0.0014, n_extra_lev)],
                    axis=1,
                )

            # Finally base GAMMA.shape[0] on the AR-level
            if self.__config.ar_order == 1:
                GAMMA = GAMMA[0, :]
            if self.__config.ar_order > 2:
                for _ in range(self.__config.ar_order - 2):
                    GAMMA = np.vstack((GAMMA, GAMMA[1, :]))

            # Finally, transpose GAMMA to ensure that the shape is the same as np.empty((n_cascade_levels, ar_order))
            GAMMA = GAMMA.transpose()
            assert GAMMA.shape == (
                self.__config.n_cascade_levels,
                self.__config.ar_order,
            )

        # Print the GAMMA value
        nowcast_utils.print_corrcoefs(GAMMA)

        if self.__config.ar_order == 2:
            # adjust the lag-2 correlation coefficient to ensure that the AR(p)
            # process is stationary
            for i in range(self.__config.n_cascade_levels):
                GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef2(
                    GAMMA[i, 0], GAMMA[i, 1]
                )

        # estimate the parameters of the AR(p) model from the auto-correlation
        # coefficients
        self.__params.PHI = np.empty(
            (self.__config.n_cascade_levels, self.__config.ar_order + 1)
        )
        for i in range(self.__config.n_cascade_levels):
            self.__params.PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])

        nowcast_utils.print_ar_params(self.__params.PHI)

    def __multiply_precip_cascade_to_match_ensemble_members(self):
        """
        Duplicate the last p-1 precipitation cascades across all ensemble members
        for the AR(p) model, ensuring each member has the required input structure.
        """
        self.__state.precip_cascades = np.stack(
            [
                [
                    self.__state.precip_cascades[i][-self.__config.ar_order :].copy()
                    for i in range(self.__config.n_cascade_levels)
                ]
            ]
            * self.__config.n_ens_members
        )

    def __initialize_random_generators(self):
        """
        Initialize random generators for precipitation noise, probability matching,
        and velocity perturbations. Each ensemble member gets a separate generator,
        ensuring reproducibility and controlled randomness in forecasts.
        """
        seed = self.__config.seed
        if self.__config.noise_method is not None:
            self.__state.randgen_precip = []
            for j in range(self.__config.n_ens_members):
                rs = np.random.RandomState(seed)
                self.__state.randgen_precip.append(rs)
                seed = rs.randint(0, high=1e9)

        if self.__config.probmatching_method is not None:
            self.__state.randgen_probmatching = []
            for j in range(self.__config.n_ens_members):
                rs = np.random.RandomState(seed)
                self.__state.randgen_probmatching.append(rs)
                seed = rs.randint(0, high=1e9)

        if self.__config.velocity_perturbation_method is not None:
            self.__state.randgen_motion = []
            for j in range(self.__config.n_ens_members):
                rs = np.random.RandomState(seed)
                self.__state.randgen_motion.append(rs)
                seed = rs.randint(0, high=1e9)

            (
                init_velocity_noise,
                self.__params.generate_velocity_noise,
            ) = noise.get_method(self.__config.velocity_perturbation_method)

            # initialize the perturbation generators for the motion field
            self.__params.velocity_perturbations = []
            for j in range(self.__config.n_ens_members):
                kwargs = {
                    "randstate": self.__state.randgen_motion[j],
                    "p_par": self.__params.velocity_perturbations_parallel,
                    "p_perp": self.__params.velocity_perturbations_perpendicular,
                }
                vp_ = init_velocity_noise(
                    self.__velocity,
                    1.0 / self.__config.kmperpixel,
                    self.__config.timestep,
                    **kwargs,
                )
                self.__params.velocity_perturbations.append(vp_)
        else:
            (
                self.__params.velocity_perturbations,
                self.__params.generate_velocity_noise,
            ) = (None, None)

    def __prepare_forecast_loop(self):
        """
        Initialize variables and structures needed for the forecast loop, including
        displacement tracking, mask parameters, noise handling, FFT objects, and
        extrapolation scaling for nowcasting.
        """
        # Empty arrays for the previous displacements and the forecast cascade
        self.__state.previous_displacement = np.stack(
            [None for j in range(self.__config.n_ens_members)]
        )
        self.__state.previous_displacement_noise_cascade = np.stack(
            [None for j in range(self.__config.n_ens_members)]
        )
        self.__state.previous_displacement_prob_matching = np.stack(
            [None for j in range(self.__config.n_ens_members)]
        )
        self.__state.final_blended_forecast = [
            [] for j in range(self.__config.n_ens_members)
        ]

        if self.__config.mask_method == "incremental":
            # get mask parameters
            self.__params.mask_rim = self.__params.mask_kwargs.get("mask_rim", 10)
            self.__params.max_mask_rim = self.__params.mask_kwargs.get(
                "max_mask_rim", 10
            )
            mask_f = self.__params.mask_kwargs.get("mask_f", 1.0)
            # initialize the structuring element
            struct = generate_binary_structure(2, 1)
            # iterate it to expand it nxn
            n = mask_f * self.__config.timestep / self.__config.kmperpixel
            self.__params.struct = iterate_structure(struct, int((n - 1) / 2.0))
        else:
            self.__params.mask_rim, self.__params.struct = None, None

        if self.__config.noise_method is None:
            self.__state.final_blended_forecast_non_perturbed = [
                self.__state.precip_cascades[0][i].copy()
                for i in range(self.__config.n_cascade_levels)
            ]
        else:
            self.__state.final_blended_forecast_non_perturbed = None

        self.__params.fft_objs = []
        for i in range(self.__config.n_ens_members):
            self.__params.fft_objs.append(
                utils.get_method(
                    self.__config.fft_method,
                    shape=self.__state.precip_cascades.shape[-2:],
                )
            )

        # initizalize the current and previous extrapolation forecast scale for the nowcasting component
        # phi1 / (1 - phi2), see BPS2004
        self.__state.rho_extrap_cascade_prev = np.repeat(
            1.0, self.__params.PHI.shape[0]
        )
        self.__state.rho_extrap_cascade = self.__params.PHI[:, 0] / (
            1.0 - self.__params.PHI[:, 1]
        )

    def __initialize_noise_cascades(self):
        """
        Initialize the noise cascade with identical noise for all AR(n) steps
        We also need to return the mean and standard deviations of the noise
        for the recombination of the noise before advecting it.
        """
        self.__state.precip_noise_cascades = np.zeros(
            self.__state.precip_cascades.shape
        )
        self.__state.precip_mean_noise = np.zeros(
            (self.__config.n_ens_members, self.__config.n_cascade_levels)
        )
        self.__state.precip_std_noise = np.zeros(
            (self.__config.n_ens_members, self.__config.n_cascade_levels)
        )
        if self.__config.noise_method:
            for j in range(self.__config.n_ens_members):
                epsilon = self.__params.noise_generator(
                    self.__params.perturbation_generator,
                    randstate=self.__state.randgen_precip[j],
                    fft_method=self.__params.fft_objs[j],
                    domain=self.__config.domain,
                )
                epsilon_decomposed = self.__params.decomposition_method(
                    epsilon,
                    self.__params.bandpass_filter,
                    fft_method=self.__params.fft_objs[j],
                    input_domain=self.__config.domain,
                    output_domain=self.__config.domain,
                    compute_stats=True,
                    normalize=True,
                    compact_output=True,
                )
                self.__state.precip_mean_noise[j] = epsilon_decomposed["means"]
                self.__state.precip_std_noise[j] = epsilon_decomposed["stds"]
                for i in range(self.__config.n_cascade_levels):
                    epsilon_temp = epsilon_decomposed["cascade_levels"][i]
                    epsilon_temp *= self.__params.noise_std_coeffs[i]
                    for n in range(self.__config.ar_order):
                        self.__state.precip_noise_cascades[j][i][n] = epsilon_temp
                epsilon_decomposed = None
                epsilon_temp = None

    def __determine_subtimesteps_and_nowcast_time_step(self, t, subtimestep_idx):
        """
        Determine the current sub-timesteps and check if the current time step
        requires nowcasting. Updates the `is_nowcast_time_step` flag accordingly.
        """
        if self.__params.time_steps_is_list:
            self.__state.subtimesteps = [
                self.__params.original_timesteps[t_] for t_ in subtimestep_idx
            ]
        else:
            self.__state.subtimesteps = [t]

        if (self.__params.time_steps_is_list and self.__state.subtimesteps) or (
            not self.__params.time_steps_is_list and t > 0
        ):
            self.__state.is_nowcast_time_step = True
        else:
            self.__state.is_nowcast_time_step = False

        if self.__state.is_nowcast_time_step:
            print(
                f"Computing nowcast for time step {t}... ",
                end="",
                flush=True,
            )

    def __decompose_nwp_if_needed_and_fill_nans_in_nwp(self, t):
        """
        Decompose NWP model precipitation fields if needed, store cascade components,
        and replace any NaN or infinite values with appropriate minimum values.
        """
        if self.__state.precip_models_cascades is not None:
            decomp_precip_models = list(self.__state.precip_models_cascades[:, t])

        else:
            if self.__precip_models.shape[0] == 1:
                decomp_precip_models = [
                    self.__params.decomposition_method(
                        self.__precip_models[0, t, :, :],
                        bp_filter=self.__params.bandpass_filter,
                        fft_method=self.__params.fft,
                        output_domain=self.__config.domain,
                        normalize=True,
                        compute_stats=True,
                        compact_output=True,
                    )
                ]
            else:
                with ThreadPool(self.__config.num_workers) as pool:
                    decomp_precip_models = pool.map(
                        partial(
                            self.__params.decomposition_method,
                            bp_filter=self.__params.bandpass_filter,
                            fft_method=self.__params.fft,
                            output_domain=self.__config.domain,
                            normalize=True,
                            compute_stats=True,
                            compact_output=True,
                        ),
                        list(self.__precip_models[:, t, :, :]),
                    )

        self.__state.precip_models_cascades_timestep = np.array(
            [decomp["cascade_levels"] for decomp in decomp_precip_models]
        )
        self.__state.mean_models_timestep = np.array(
            [decomp["means"] for decomp in decomp_precip_models]
        )
        self.__state.std_models_timestep = np.array(
            [decomp["stds"] for decomp in decomp_precip_models]
        )

        # Check if the NWP fields contain nans or infinite numbers. If so,
        # fill these with the minimum value present in precip (corresponding to
        # zero rainfall in the radar observations)

        # Ensure that the NWP cascade and fields do no contain any nans or infinite number
        # Fill nans and infinite numbers with the minimum value present in precip
        self.__state.precip_models_timestep = self.__precip_models[:, t, :, :].astype(
            np.float64, copy=False
        )  # (corresponding to zero rainfall in the radar observations)
        min_cascade = np.nanmin(self.__state.precip_cascades)
        min_precip = np.nanmin(self.__precip)
        self.__state.precip_models_cascades_timestep[
            ~np.isfinite(self.__state.precip_models_cascades_timestep)
        ] = min_cascade
        self.__state.precip_models_timestep[
            ~np.isfinite(self.__state.precip_models_timestep)
        ] = min_precip
        # Also set any nans or infs in the mean and sigma of the cascade to
        # respectively 0.0 and 1.0
        self.__state.mean_models_timestep[
            ~np.isfinite(self.__state.mean_models_timestep)
        ] = 0.0
        self.__state.std_models_timestep[
            ~np.isfinite(self.__state.std_models_timestep)
        ] = 0.0

    def __find_nowcast_NWP_combination(self, t):
        """
        Determine which (NWP) models will be combined with which nowcast ensemble members.
        With the way it is implemented at this moment: n_ens_members of the output equals
        the maximum number of (ensemble) members in the input (either the nowcasts or NWP).
        """
        self.__state.velocity_models_timestep = self.__velocity_models[
            :, t, :, :, :
        ].astype(np.float64, copy=False)
        # Make sure the number of model members is not larger than or equal to n_ens_members
        n_model_members = self.__state.precip_models_cascades_timestep.shape[0]
        if n_model_members > self.__config.n_ens_members:
            raise ValueError(
                "The number of NWP model members is larger than the given number of ensemble members. n_model_members <= n_ens_members."
            )

        # Check if NWP models/members should be used individually, or if all of
        # them are blended together per nowcast ensemble member.
        if self.__config.blend_nwp_members:
            self.__state.mapping_list_NWP_member_to_ensemble_member = None

        else:
            # Start with determining the maximum and mimimum number of members/models
            # in both input products
            n_ens_members_max = max(self.__config.n_ens_members, n_model_members)
            n_ens_members_min = min(self.__config.n_ens_members, n_model_members)
            # Also make a list of the model index numbers. These indices are needed
            # for indexing the right climatological skill file when pysteps calculates
            # the blended forecast in parallel.
            if n_model_members > 1:
                self.__state.mapping_list_NWP_member_to_ensemble_member = np.arange(
                    n_model_members
                )
            else:
                self.__state.mapping_list_NWP_member_to_ensemble_member = [0]

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
                    self.__state.precip_models_cascades_timestep = np.repeat(
                        self.__state.precip_models_cascades_timestep,
                        n_ens_members_max,
                        axis=0,
                    )
                    self.__state.mean_models_timestep = np.repeat(
                        self.__state.mean_models_timestep, n_ens_members_max, axis=0
                    )
                    self.__state.std_models_timestep = np.repeat(
                        self.__state.std_models_timestep, n_ens_members_max, axis=0
                    )
                    self.__state.velocity_models_timestep = np.repeat(
                        self.__state.velocity_models_timestep, n_ens_members_max, axis=0
                    )
                    # For the prob. matching
                    self.__state.precip_models_timestep = np.repeat(
                        self.__state.precip_models_timestep, n_ens_members_max, axis=0
                    )
                    # Finally, for the model indices
                    self.__state.mapping_list_NWP_member_to_ensemble_member = np.repeat(
                        self.__state.mapping_list_NWP_member_to_ensemble_member,
                        n_ens_members_max,
                        axis=0,
                    )

                elif n_model_members == n_ens_members_min:
                    repeats = [
                        (n_ens_members_max + i) // n_ens_members_min
                        for i in range(n_ens_members_min)
                    ]
                    self.__state.precip_models_cascades_timestep = np.repeat(
                        self.__state.precip_models_cascades_timestep,
                        repeats,
                        axis=0,
                    )
                    self.__state.mean_models_timestep = np.repeat(
                        self.__state.mean_models_timestep, repeats, axis=0
                    )
                    self.__state.std_models_timestep = np.repeat(
                        self.__state.std_models_timestep, repeats, axis=0
                    )
                    self.__state.velocity_models_timestep = np.repeat(
                        self.__state.velocity_models_timestep, repeats, axis=0
                    )
                    # For the prob. matching
                    self.__state.precip_models_timestep = np.repeat(
                        self.__state.precip_models_timestep, repeats, axis=0
                    )
                    # Finally, for the model indices
                    self.__state.mapping_list_NWP_member_to_ensemble_member = np.repeat(
                        self.__state.mapping_list_NWP_member_to_ensemble_member,
                        repeats,
                        axis=0,
                    )

    def __determine_skill_for_current_timestep(self, t):
        """
        Compute the skill of NWP model forecasts at t=0 using spatial correlation,
        ensuring skill decreases with increasing scale level. For t>0, update
        extrapolation skill based on lead time.
        """
        if t == 0:
            # Calculate the initial skill of the (NWP) model forecasts at t=0.
            self.__params.rho_nwp_models = []
            for model_index in range(
                self.__state.precip_models_cascades_timestep.shape[0]
            ):
                rho_value = blending.skill_scores.spatial_correlation(
                    obs=self.__state.precip_cascades[0, :, -1, :, :].copy(),
                    mod=self.__state.precip_models_cascades_timestep[
                        model_index, :, :, :
                    ].copy(),
                    domain_mask=self.__params.domain_mask,
                )
                self.__params.rho_nwp_models.append(rho_value)
            self.__params.rho_nwp_models = np.stack(self.__params.rho_nwp_models)

            # Ensure that the model skill decreases with increasing scale level.
            for model_index in range(
                self.__state.precip_models_cascades_timestep.shape[0]
            ):
                for i in range(
                    1, self.__state.precip_models_cascades_timestep.shape[1]
                ):
                    if (
                        self.__params.rho_nwp_models[model_index, i]
                        > self.__params.rho_nwp_models[model_index, i - 1]
                    ):
                        # Set it equal to the previous scale level
                        self.__params.rho_nwp_models[model_index, i] = (
                            self.__params.rho_nwp_models[model_index, i - 1]
                        )

            # Save this in the climatological skill file
            blending.clim.save_skill(
                current_skill=self.__params.rho_nwp_models,
                validtime=self.__issuetime,
                outdir_path=self.__config.outdir_path_skill,
                **self.__params.climatology_kwargs,
            )
        if t > 0:
            # Determine the skill of the components for lead time (t0 + t)
            # First for the extrapolation component. Only calculate it when t > 0.
            (
                self.__state.rho_extrap_cascade,
                self.__state.rho_extrap_cascade_prev,
            ) = blending.skill_scores.lt_dependent_cor_extrapolation(
                PHI=self.__params.PHI,
                correlations=self.__state.rho_extrap_cascade,
                correlations_prev=self.__state.rho_extrap_cascade_prev,
            )

    def __determine_NWP_skill_for_next_timestep(self, t, j, worker_state):
        """
        Compute the skill of NWP model components for the next lead time (t0 + t),
        blending with extrapolation skill if configured. Updates the worker state
        with the final blended skill forecast.
        """
        if self.__config.blend_nwp_members:
            rho_nwp_forecast = []
            for model_index in range(self.__params.rho_nwp_models.shape[0]):
                rho_value = blending.skill_scores.lt_dependent_cor_nwp(
                    lt=(t * int(self.__config.timestep)),
                    correlations=self.__params.rho_nwp_models[model_index],
                    outdir_path=self.__config.outdir_path_skill,
                    n_model=model_index,
                    skill_kwargs=self.__params.climatology_kwargs,
                )
                rho_nwp_forecast.append(rho_value)
            rho_nwp_forecast = np.stack(rho_nwp_forecast)
            # Concatenate rho_extrap_cascade and rho_nwp
            worker_state.rho_final_blended_forecast = np.concatenate(
                (worker_state.rho_extrap_cascade[None, :], rho_nwp_forecast), axis=0
            )
        else:
            # TODO: check if j is the best accessor for this variable
            rho_nwp_forecast = blending.skill_scores.lt_dependent_cor_nwp(
                lt=(t * int(self.__config.timestep)),
                correlations=self.__params.rho_nwp_models[j],
                outdir_path=self.__config.outdir_path_skill,
                n_model=worker_state.mapping_list_NWP_member_to_ensemble_member[j],
                skill_kwargs=self.__params.climatology_kwargs,
            )
            # Concatenate rho_extrap_cascade and rho_nwp
            worker_state.rho_final_blended_forecast = np.concatenate(
                (worker_state.rho_extrap_cascade[None, :], rho_nwp_forecast[None, :]),
                axis=0,
            )

    def __determine_weights_per_component(self, worker_state):
        """
        Compute blending weights for each component based on the selected method
        ('bps' or 'spn'). Weights are determined for both full blending and
        model-only scenarios, accounting for correlations and covariance.
        """
        # Weights following the bps method. These are needed for the velocity
        # weights prior to the advection step. If weights method spn is
        # selected, weights will be overwritten with those weights prior to
        # blending step.
        # weight = [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
        worker_state.weights = calculate_weights_bps(
            worker_state.rho_final_blended_forecast
        )

        # The model only weights
        if self.__config.weights_method == "bps":
            # Determine the weights of the components without the extrapolation
            # cascade, in case this is no data or outside the mask.
            worker_state.weights_model_only = calculate_weights_bps(
                worker_state.rho_final_blended_forecast[1:, :]
            )
        elif self.__config.weights_method == "spn":
            # Only the weights of the components without the extrapolation
            # cascade will be determined here. The full set of weights are
            # determined after the extrapolation step in this method.
            if (
                self.__config.blend_nwp_members
                and worker_state.precip_models_cascades_timestep.shape[0] > 1
            ):
                worker_state.weights_model_only = np.zeros(
                    (
                        worker_state.precip_models_cascades_timestep.shape[0] + 1,
                        self.__config.n_cascade_levels,
                    )
                )
                for i in range(self.__config.n_cascade_levels):
                    # Determine the normalized covariance matrix (containing)
                    # the cross-correlations between the models
                    covariance_nwp_models = np.corrcoef(
                        np.stack(
                            [
                                worker_state.precip_models_cascades_timestep[
                                    n_model, i, :, :
                                ].flatten()
                                for n_model in range(
                                    worker_state.precip_models_cascades_timestep.shape[
                                        0
                                    ]
                                )
                            ]
                        )
                    )
                    # Determine the weights for this cascade level
                    worker_state.weights_model_only[:, i] = calculate_weights_spn(
                        correlations=worker_state.rho_final_blended_forecast[1:, i],
                        covariance=covariance_nwp_models,
                    )
            else:
                # Same as correlation and noise is 1 - correlation
                worker_state.weights_model_only = calculate_weights_bps(
                    worker_state.rho_final_blended_forecast[1:, :]
                )
        else:
            raise ValueError(
                "Unknown weights method %s: must be 'bps' or 'spn'"
                % self.__config.weights_method
            )

    def __regress_extrapolation_and_noise_cascades(self, j, worker_state):
        """
        Apply autoregressive (AR) updates to the extrapolation and noise cascades
        for the next time step. If noise is enabled, generate and decompose a
        spatially correlated noise field before applying the AR process.
        """
        # Determine the epsilon, a cascade of temporally independent
        # but spatially correlated noise
        if self.__config.noise_method is not None:
            # generate noise field
            epsilon = self.__params.noise_generator(
                self.__params.perturbation_generator,
                randstate=worker_state.randgen_precip[j],
                fft_method=self.__params.fft_objs[j],
                domain=self.__config.domain,
            )

            # decompose the noise field into a cascade
            epsilon_decomposed = self.__params.decomposition_method(
                epsilon,
                self.__params.bandpass_filter,
                fft_method=self.__params.fft_objs[j],
                input_domain=self.__config.domain,
                output_domain=self.__config.domain,
                compute_stats=True,
                normalize=True,
                compact_output=True,
            )
        else:
            epsilon_decomposed = None

        # Regress the extrapolation component to the subsequent time
        # step
        # iterate the AR(p) model for each cascade level
        for i in range(self.__config.n_cascade_levels):
            # apply AR(p) process to extrapolation cascade level
            if (
                epsilon_decomposed is not None
                or self.__config.velocity_perturbation_method is not None
            ):
                worker_state.precip_cascades[j][i] = autoregression.iterate_ar_model(
                    worker_state.precip_cascades[j][i], self.__params.PHI[i, :]
                )
                # Renormalize the cascade
                worker_state.precip_cascades[j][i][1] /= np.std(
                    worker_state.precip_cascades[j][i][1]
                )
            else:
                # use the deterministic AR(p) model computed above if
                # perturbations are disabled
                worker_state.precip_cascades[j][i] = (
                    worker_state.final_blended_forecast_non_perturbed[i]
                )

        # Regress the noise component to the subsequent time step
        # iterate the AR(p) model for each cascade level
        for i in range(self.__config.n_cascade_levels):
            # normalize the noise cascade
            if epsilon_decomposed is not None:
                epsilon_temp = epsilon_decomposed["cascade_levels"][i]
                epsilon_temp *= self.__params.noise_std_coeffs[i]
            else:
                epsilon_temp = None
            # apply AR(p) process to noise cascade level
            # (Returns zero noise if epsilon_decomposed is None)
            worker_state.precip_noise_cascades[j][i] = autoregression.iterate_ar_model(
                worker_state.precip_noise_cascades[j][i],
                self.__params.PHI[i, :],
                eps=epsilon_temp,
            )

        epsilon_decomposed = None
        epsilon_temp = None

    def __perturb_blend_and_advect_extrapolation_and_noise_to_current_timestep(
        self, t, j, worker_state
    ):
        """
        Apply perturbations, blend motion fields, and advect extrapolated and noise
        cascades to the current time step (or sub-timesteps). This step ensures
        realistic motion updates in nowcasting.
        """
        # Settings and initialize the output
        extrap_kwargs_ = worker_state.extrapolation_kwargs.copy()
        extrap_kwargs_noise = worker_state.extrapolation_kwargs.copy()
        extrap_kwargs_pb = worker_state.extrapolation_kwargs.copy()
        velocity_perturbations_extrapolation = self.__velocity
        # The following should be accessible after this function
        worker_state.precip_extrapolated_decomp = []
        worker_state.noise_extrapolated_decomp = []
        worker_state.precip_extrapolated_probability_matching = []

        # Extrapolate per sub time step
        for t_sub in worker_state.subtimesteps:
            if t_sub > 0:
                t_diff_prev_subtimestep_int = t_sub - int(t_sub)
                if t_diff_prev_subtimestep_int > 0.0:
                    precip_forecast_cascade_subtimestep = [
                        (1.0 - t_diff_prev_subtimestep_int)
                        * worker_state.precip_cascades_prev_subtimestep[j][i][-1, :]
                        + t_diff_prev_subtimestep_int
                        * worker_state.precip_cascades[j][i][-1, :]
                        for i in range(self.__config.n_cascade_levels)
                    ]
                    noise_cascade_subtimestep = [
                        (1.0 - t_diff_prev_subtimestep_int)
                        * worker_state.cascade_noise_prev_subtimestep[j][i][-1, :]
                        + t_diff_prev_subtimestep_int
                        * worker_state.precip_noise_cascades[j][i][-1, :]
                        for i in range(self.__config.n_cascade_levels)
                    ]

                else:
                    precip_forecast_cascade_subtimestep = [
                        worker_state.precip_cascades_prev_subtimestep[j][i][-1, :]
                        for i in range(self.__config.n_cascade_levels)
                    ]
                    noise_cascade_subtimestep = [
                        worker_state.cascade_noise_prev_subtimestep[j][i][-1, :]
                        for i in range(self.__config.n_cascade_levels)
                    ]

                precip_forecast_cascade_subtimestep = np.stack(
                    precip_forecast_cascade_subtimestep
                )
                noise_cascade_subtimestep = np.stack(noise_cascade_subtimestep)

                t_diff_prev_subtimestep = t_sub - worker_state.time_prev_timestep[j]
                worker_state.leadtime_since_start_forecast[j] += t_diff_prev_subtimestep

                # compute the perturbed motion field - include the NWP
                # velocities and the weights. Note that we only perturb
                # the extrapolation velocity field, as the NWP velocity
                # field is present per time step
                if self.__config.velocity_perturbation_method is not None:
                    velocity_perturbations_extrapolation = (
                        self.__velocity
                        + self.__params.generate_velocity_noise(
                            self.__params.velocity_perturbations[j],
                            worker_state.leadtime_since_start_forecast[j]
                            * self.__config.timestep,
                        )
                    )

                # Stack the perturbed extrapolation and the NWP velocities
                if self.__config.blend_nwp_members:
                    velocity_stack_all = np.concatenate(
                        (
                            velocity_perturbations_extrapolation[None, :, :, :],
                            worker_state.velocity_models_timestep,
                        ),
                        axis=0,
                    )
                else:
                    velocity_models = worker_state.velocity_models_timestep[j]
                    velocity_stack_all = np.concatenate(
                        (
                            velocity_perturbations_extrapolation[None, :, :, :],
                            velocity_models[None, :, :, :],
                        ),
                        axis=0,
                    )
                    velocity_models = None

                # Obtain a blended optical flow, using the weights of the
                # second cascade following eq. 24 in BPS2006
                velocity_blended = blending.utils.blend_optical_flows(
                    flows=velocity_stack_all,
                    weights=worker_state.weights[
                        :-1, 1
                    ],  # [(extr_field, n_model_fields), cascade_level=2]
                )

                # Extrapolate both cascades to the next time step
                # First recompose the cascade, advect it and decompose it again
                # This is needed to remove the interpolation artifacts.
                # In addition, the number of extrapolations is greatly reduced
                # A. Radar Rain
                precip_forecast_recomp_subtimestep = blending.utils.recompose_cascade(
                    combined_cascade=precip_forecast_cascade_subtimestep,
                    combined_mean=worker_state.mean_extrapolation,
                    combined_sigma=worker_state.std_extrapolation,
                )
                # Make sure we have values outside the mask
                if self.__params.zero_precip_radar:
                    precip_forecast_recomp_subtimestep = np.nan_to_num(
                        precip_forecast_recomp_subtimestep,
                        copy=True,
                        nan=self.__params.precip_zerovalue,
                        posinf=self.__params.precip_zerovalue,
                        neginf=self.__params.precip_zerovalue,
                    )
                # Put back the mask
                precip_forecast_recomp_subtimestep[self.__params.domain_mask] = np.nan
                worker_state.extrapolation_kwargs["displacement_prev"] = (
                    worker_state.previous_displacement[j]
                )
                (
                    precip_forecast_extrapolated_recomp_subtimestep_temp,
                    worker_state.previous_displacement[j],
                ) = self.__params.extrapolation_method(
                    precip_forecast_recomp_subtimestep,
                    velocity_blended,
                    [t_diff_prev_subtimestep],
                    allow_nonfinite_values=True,
                    **worker_state.extrapolation_kwargs,
                )
                precip_extrapolated_recomp_subtimestep = (
                    precip_forecast_extrapolated_recomp_subtimestep_temp[0].copy()
                )
                temp_mask = ~np.isfinite(precip_extrapolated_recomp_subtimestep)
                # TODO: WHERE DO CAN I FIND THIS -15.0
                precip_extrapolated_recomp_subtimestep[
                    ~np.isfinite(precip_extrapolated_recomp_subtimestep)
                ] = self.__params.precip_zerovalue
                precip_extrapolated_decomp = self.__params.decomposition_method(
                    precip_extrapolated_recomp_subtimestep,
                    self.__params.bandpass_filter,
                    mask=self.__params.mask_threshold,
                    fft_method=self.__params.fft,
                    output_domain=self.__config.domain,
                    normalize=True,
                    compute_stats=True,
                    compact_output=True,
                )["cascade_levels"]
                # Make sure we have values outside the mask
                if self.__params.zero_precip_radar:
                    precip_extrapolated_decomp = np.nan_to_num(
                        precip_extrapolated_decomp,
                        copy=True,
                        nan=np.nanmin(precip_forecast_cascade_subtimestep),
                        posinf=np.nanmin(precip_forecast_cascade_subtimestep),
                        neginf=np.nanmin(precip_forecast_cascade_subtimestep),
                    )
                for i in range(self.__config.n_cascade_levels):
                    precip_extrapolated_decomp[i][temp_mask] = np.nan
                # B. Noise
                noise_cascade_subtimestep_recomp = blending.utils.recompose_cascade(
                    combined_cascade=noise_cascade_subtimestep,
                    combined_mean=worker_state.precip_mean_noise[j],
                    combined_sigma=worker_state.precip_std_noise[j],
                )
                extrap_kwargs_noise["displacement_prev"] = (
                    worker_state.previous_displacement_noise_cascade[j]
                )
                extrap_kwargs_noise["map_coordinates_mode"] = "wrap"
                (
                    noise_extrapolated_recomp_temp,
                    worker_state.previous_displacement_noise_cascade[j],
                ) = self.__params.extrapolation_method(
                    noise_cascade_subtimestep_recomp,
                    velocity_blended,
                    [t_diff_prev_subtimestep],
                    allow_nonfinite_values=True,
                    **extrap_kwargs_noise,
                )
                noise_extrapolated_recomp = noise_extrapolated_recomp_temp[0].copy()
                noise_extrapolated_decomp = self.__params.decomposition_method(
                    noise_extrapolated_recomp,
                    self.__params.bandpass_filter,
                    mask=self.__params.mask_threshold,
                    fft_method=self.__params.fft,
                    output_domain=self.__config.domain,
                    normalize=True,
                    compute_stats=True,
                    compact_output=True,
                )["cascade_levels"]
                for i in range(self.__config.n_cascade_levels):
                    noise_extrapolated_decomp[i] *= self.__params.noise_std_coeffs[i]

                # Append the results to the output lists
                worker_state.precip_extrapolated_decomp.append(
                    precip_extrapolated_decomp.copy()
                )
                worker_state.noise_extrapolated_decomp.append(
                    noise_extrapolated_decomp.copy()
                )
                precip_forecast_cascade_subtimestep = None
                precip_forecast_recomp_subtimestep = None
                precip_forecast_extrapolated_recomp_subtimestep_temp = None
                precip_extrapolated_recomp_subtimestep = None
                precip_extrapolated_decomp = None
                noise_cascade_subtimestep = None
                noise_cascade_subtimestep_recomp = None
                noise_extrapolated_recomp_temp = None
                noise_extrapolated_recomp = None
                noise_extrapolated_decomp = None

                # Finally, also extrapolate the initial radar rainfall
                # field. This will be blended with the rainfall field(s)
                # of the (NWP) model(s) for Lagrangian blended prob. matching
                # min_R = np.min(precip)
                extrap_kwargs_pb["displacement_prev"] = (
                    worker_state.previous_displacement_prob_matching[j]
                )
                # Apply the domain mask to the extrapolation component
                precip_forecast_temp_for_probability_matching = self.__precip.copy()
                precip_forecast_temp_for_probability_matching[
                    self.__params.domain_mask
                ] = np.nan

                (
                    precip_forecast_extrapolated_probability_matching_temp,
                    worker_state.previous_displacement_prob_matching[j],
                ) = self.__params.extrapolation_method(
                    precip_forecast_temp_for_probability_matching,
                    velocity_blended,
                    [t_diff_prev_subtimestep],
                    allow_nonfinite_values=True,
                    **extrap_kwargs_pb,
                )

                worker_state.precip_extrapolated_probability_matching.append(
                    precip_forecast_extrapolated_probability_matching_temp[0]
                )

                worker_state.time_prev_timestep[j] = t_sub

        if len(worker_state.precip_extrapolated_decomp) > 0:
            worker_state.precip_extrapolated_decomp = np.stack(
                worker_state.precip_extrapolated_decomp
            )
            worker_state.noise_extrapolated_decomp = np.stack(
                worker_state.noise_extrapolated_decomp
            )
            worker_state.precip_extrapolated_probability_matching = np.stack(
                worker_state.precip_extrapolated_probability_matching
            )

        # advect the forecast field by one time step if no subtimesteps in the
        # current interval were found
        if not worker_state.subtimesteps:
            t_diff_prev_subtimestep = t + 1 - worker_state.time_prev_timestep[j]
            worker_state.leadtime_since_start_forecast[j] += t_diff_prev_subtimestep

            # compute the perturbed motion field - include the NWP
            # velocities and the weights
            if self.__config.velocity_perturbation_method is not None:
                velocity_perturbations_extrapolation = (
                    self.__velocity
                    + self.__params.generate_velocity_noise(
                        self.__params.velocity_perturbations[j],
                        worker_state.leadtime_since_start_forecast[j]
                        * self.__config.timestep,
                    )
                )

            # Stack the perturbed extrapolation and the NWP velocities
            if self.__config.blend_nwp_members:
                velocity_stack_all = np.concatenate(
                    (
                        velocity_perturbations_extrapolation[None, :, :, :],
                        worker_state.velocity_models_timestep,
                    ),
                    axis=0,
                )
            else:
                velocity_models = worker_state.velocity_models_timestep[j]
                velocity_stack_all = np.concatenate(
                    (
                        velocity_perturbations_extrapolation[None, :, :, :],
                        velocity_models[None, :, :, :],
                    ),
                    axis=0,
                )
                velocity_models = None

            # Obtain a blended optical flow, using the weights of the
            # second cascade following eq. 24 in BPS2006
            velocity_blended = blending.utils.blend_optical_flows(
                flows=velocity_stack_all,
                weights=worker_state.weights[
                    :-1, 1
                ],  # [(extr_field, n_model_fields), cascade_level=2]
            )

            # Extrapolate the extrapolation and noise cascade

            extrap_kwargs_["displacement_prev"] = worker_state.previous_displacement[j]
            extrap_kwargs_noise["displacement_prev"] = (
                worker_state.previous_displacement_noise_cascade[j]
            )
            extrap_kwargs_noise["map_coordinates_mode"] = "wrap"

            (
                _,
                worker_state.previous_displacement[j],
            ) = self.__params.extrapolation_method(
                None,
                velocity_blended,
                [t_diff_prev_subtimestep],
                allow_nonfinite_values=True,
                **extrap_kwargs_,
            )

            (
                _,
                worker_state.previous_displacement_noise_cascade[j],
            ) = self.__params.extrapolation_method(
                None,
                velocity_blended,
                [t_diff_prev_subtimestep],
                allow_nonfinite_values=True,
                **extrap_kwargs_noise,
            )

            # Also extrapolate the radar observation, used for the probability
            # matching and post-processing steps
            extrap_kwargs_pb["displacement_prev"] = (
                worker_state.previous_displacement_prob_matching[j]
            )
            (
                _,
                worker_state.previous_displacement_prob_matching[j],
            ) = self.__params.extrapolation_method(
                None,
                velocity_blended,
                [t_diff_prev_subtimestep],
                allow_nonfinite_values=True,
                **extrap_kwargs_pb,
            )

            worker_state.time_prev_timestep[j] = t + 1

        worker_state.precip_cascades_prev_subtimestep[j] = worker_state.precip_cascades[
            j
        ]
        worker_state.cascade_noise_prev_subtimestep[j] = (
            worker_state.precip_noise_cascades[j]
        )

    def __blend_cascades(self, t_sub, j, worker_state):
        """
        Blend extrapolated, NWP model, and noise cascades using predefined weights.
        Computes both full and model-only blends while also blending means and
        standard deviations across scales.
        """
        worker_state.subtimestep_index = np.where(
            np.array(worker_state.subtimesteps) == t_sub
        )[0][0]
        # First concatenate the cascades and the means and sigmas
        # precip_models = [n_models,timesteps,n_cascade_levels,m,n]
        if self.__config.blend_nwp_members:
            cascade_stack_all_components = np.concatenate(
                (
                    worker_state.precip_extrapolated_decomp[
                        None, worker_state.subtimestep_index
                    ],
                    worker_state.precip_models_cascades_timestep,
                    worker_state.noise_extrapolated_decomp[
                        None, worker_state.subtimestep_index
                    ],
                ),
                axis=0,
            )  # [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
            means_stacked = np.concatenate(
                (
                    worker_state.mean_extrapolation[None, :],
                    worker_state.mean_models_timestep,
                ),
                axis=0,
            )
            sigmas_stacked = np.concatenate(
                (
                    worker_state.std_extrapolation[None, :],
                    worker_state.std_models_timestep,
                ),
                axis=0,
            )
        else:
            cascade_stack_all_components = np.concatenate(
                (
                    worker_state.precip_extrapolated_decomp[
                        None, worker_state.subtimestep_index
                    ],
                    worker_state.precip_models_cascades_timestep[None, j],
                    worker_state.noise_extrapolated_decomp[
                        None, worker_state.subtimestep_index
                    ],
                ),
                axis=0,
            )  # [(extr_field, n_model_fields, noise), n_cascade_levels, ...]
            means_stacked = np.concatenate(
                (
                    worker_state.mean_extrapolation[None, :],
                    worker_state.mean_models_timestep[None, j],
                ),
                axis=0,
            )
            sigmas_stacked = np.concatenate(
                (
                    worker_state.std_extrapolation[None, :],
                    worker_state.std_models_timestep[None, j],
                ),
                axis=0,
            )

        # First determine the blending weights if method is spn. The
        # weights for method bps have already been determined.

        if self.__config.weights_method == "spn":
            worker_state.weights = np.zeros(
                (
                    cascade_stack_all_components.shape[0],
                    self.__config.n_cascade_levels,
                )
            )
            for i in range(self.__config.n_cascade_levels):
                # Determine the normalized covariance matrix (containing)
                # the cross-correlations between the models
                cascade_stack_all_components_temp = np.stack(
                    [
                        cascade_stack_all_components[n_model, i, :, :].flatten()
                        for n_model in range(cascade_stack_all_components.shape[0] - 1)
                    ]
                )  # -1 to exclude the noise component
                covariance_nwp_models = np.ma.corrcoef(
                    np.ma.masked_invalid(cascade_stack_all_components_temp)
                )
                # Determine the weights for this cascade level
                worker_state.weights[:, i] = calculate_weights_spn(
                    correlations=worker_state.rho_final_blended_forecast[:, i],
                    covariance=covariance_nwp_models,
                )

        # Blend the extrapolation, (NWP) model(s) and noise cascades
        worker_state.final_blended_forecast_cascades = blending.utils.blend_cascades(
            cascades_norm=cascade_stack_all_components, weights=worker_state.weights
        )

        # Also blend the cascade without the extrapolation component
        worker_state.final_blended_forecast_cascades_mod_only = (
            blending.utils.blend_cascades(
                cascades_norm=cascade_stack_all_components[1:, :],
                weights=worker_state.weights_model_only,
            )
        )

        # Blend the means and standard deviations
        # Input is array of shape [number_components, scale_level, ...]
        (
            worker_state.final_blended_forecast_means,
            worker_state.final_blended_forecast_stds,
        ) = blend_means_sigmas(
            means=means_stacked, sigmas=sigmas_stacked, weights=worker_state.weights
        )
        # Also blend the means and sigmas for the cascade without extrapolation

        (
            worker_state.final_blended_forecast_means_mod_only,
            worker_state.final_blended_forecast_stds_mod_only,
        ) = blend_means_sigmas(
            means=means_stacked[1:, :],
            sigmas=sigmas_stacked[1:, :],
            weights=worker_state.weights_model_only,
        )

    def __recompose_cascade_to_rainfall_field(self, j, worker_state):
        """
        Recompose the blended cascade into a precipitation field using the blended
        means and standard deviations. If using the spectral domain, apply inverse
        FFT for reconstruction.
        """
        worker_state.final_blended_forecast_recomposed = (
            blending.utils.recompose_cascade(
                combined_cascade=worker_state.final_blended_forecast_cascades,
                combined_mean=worker_state.final_blended_forecast_means,
                combined_sigma=worker_state.final_blended_forecast_stds,
            )
        )
        # The recomposed cascade without the extrapolation (for NaN filling
        # outside the radar domain)
        worker_state.final_blended_forecast_recomposed_mod_only = (
            blending.utils.recompose_cascade(
                combined_cascade=worker_state.final_blended_forecast_cascades_mod_only,
                combined_mean=worker_state.final_blended_forecast_means_mod_only,
                combined_sigma=worker_state.final_blended_forecast_stds_mod_only,
            )
        )
        if self.__config.domain == "spectral":
            # TODO: Check this! (Only tested with domain == 'spatial')
            worker_state.final_blended_forecast_recomposed = self.__params.fft_objs[
                j
            ].irfft2(worker_state.final_blended_forecast_recomposed)
            worker_state.final_blended_forecast_recomposed_mod_only = (
                self.__params.fft_objs[j].irfft2(
                    worker_state.final_blended_forecast_recomposed_mod_only
                )
            )

    def __post_process_output(
        self, j, t_sub, final_blended_forecast_single_member, worker_state
    ):
        """
        Apply post-processing steps to refine the final blended forecast. This
        involves masking, filling missing data with the blended NWP forecast,
        and applying probability matching to ensure consistency.

        **Steps:**

        1. **Use Mask and Fill Missing Data:**
           - Areas without reliable radar extrapolation are filled using the
             blended NWP forecast to maintain spatial coherence.

        2. **Lagrangian Blended Probability Matching:**
           - Uses the latest extrapolated radar rainfall field blended with
             the NWP model(s) forecast as a reference.
           - Ensures that the statistical distribution of the final forecast
             remains consistent with the benchmark dataset.

        3. **Blend the Extrapolated Rainfall Field with NWP Forecasts:**
           - The extrapolated rainfall field is used only for post-processing.
           - The forecast is blended using predefined weights at scale level 2.
           - This ensures that both extrapolated and modeled precipitation
             contribute appropriately to the final output.

        4. **Apply Probability Matching:**
           - Adjusts the final precipitation distribution using either empirical
             cumulative distribution functions (CDF) or mean adjustments to
             match the reference dataset.

        The final processed forecast is stored in `final_blended_forecast_single_member`.
        """

        weights_probability_matching = worker_state.weights[
            :-1, 1
        ]  # Weights without noise, level 2
        weights_probability_matching_normalized = weights_probability_matching / np.sum(
            weights_probability_matching
        )
        # And the weights for outside the radar domain
        weights_probability_matching_mod_only = worker_state.weights_model_only[
            :-1, 1
        ]  # Weights without noise, level 2
        weights_probability_matching_normalized_mod_only = (
            weights_probability_matching_mod_only
            / np.sum(weights_probability_matching_mod_only)
        )
        # Stack the fields
        if self.__config.blend_nwp_members:
            precip_forecast_probability_matching_final = np.concatenate(
                (
                    worker_state.precip_extrapolated_probability_matching[
                        None, worker_state.subtimestep_index
                    ],
                    worker_state.precip_models_timestep,
                ),
                axis=0,
            )
        else:
            precip_forecast_probability_matching_final = np.concatenate(
                (
                    worker_state.precip_extrapolated_probability_matching[
                        None, worker_state.subtimestep_index
                    ],
                    worker_state.precip_models_timestep[None, j],
                ),
                axis=0,
            )
        # Blend it
        precip_forecast_probability_matching_blended = np.sum(
            weights_probability_matching_normalized.reshape(
                weights_probability_matching_normalized.shape[0], 1, 1
            )
            * precip_forecast_probability_matching_final,
            axis=0,
        )
        if self.__config.blend_nwp_members:
            precip_forecast_probability_matching_blended_mod_only = np.sum(
                weights_probability_matching_normalized_mod_only.reshape(
                    weights_probability_matching_normalized_mod_only.shape[0],
                    1,
                    1,
                )
                * worker_state.precip_models_timestep,
                axis=0,
            )
        else:
            precip_forecast_probability_matching_blended_mod_only = (
                worker_state.precip_models_timestep[j]
            )

        # The extrapolation components are NaN outside the advected
        # radar domain. This results in NaN values in the blended
        # forecast outside the radar domain. Therefore, fill these
        # areas with the "..._mod_only" blended forecasts, consisting
        # of the NWP and noise components.

        nan_indices = np.isnan(worker_state.final_blended_forecast_recomposed)
        if self.__config.smooth_radar_mask_range != 0:
            # Compute the smooth dilated mask
            new_mask = blending.utils.compute_smooth_dilated_mask(
                nan_indices,
                max_padding_size_in_px=self.__config.smooth_radar_mask_range,
            )

            # Ensure mask values are between 0 and 1
            mask_model = np.clip(new_mask, 0, 1)
            mask_radar = np.clip(1 - new_mask, 0, 1)

            # Handle NaNs in precip_forecast_new and precip_forecast_new_mod_only by setting NaNs to 0 in the blending step
            precip_forecast_recomposed_mod_only_no_nan = np.nan_to_num(
                worker_state.final_blended_forecast_recomposed_mod_only, nan=0
            )
            precip_forecast_recomposed_no_nan = np.nan_to_num(
                worker_state.final_blended_forecast_recomposed, nan=0
            )

            # Perform the blending of radar and model inside the radar domain using a weighted combination
            worker_state.final_blended_forecast_recomposed = np.nansum(
                [
                    mask_model * precip_forecast_recomposed_mod_only_no_nan,
                    mask_radar * precip_forecast_recomposed_no_nan,
                ],
                axis=0,
            )

            precip_forecast_probability_matching_blended = np.nansum(
                [
                    precip_forecast_probability_matching_blended * mask_radar,
                    precip_forecast_probability_matching_blended_mod_only * mask_model,
                ],
                axis=0,
            )
        else:
            worker_state.final_blended_forecast_recomposed[nan_indices] = (
                worker_state.final_blended_forecast_recomposed_mod_only[nan_indices]
            )
            nan_indices = np.isnan(precip_forecast_probability_matching_blended)
            precip_forecast_probability_matching_blended[nan_indices] = (
                precip_forecast_probability_matching_blended_mod_only[nan_indices]
            )

        # Finally, fill the remaining nan values, if present, with
        # the minimum value in the forecast
        nan_indices = np.isnan(worker_state.final_blended_forecast_recomposed)
        worker_state.final_blended_forecast_recomposed[nan_indices] = np.nanmin(
            worker_state.final_blended_forecast_recomposed
        )
        nan_indices = np.isnan(precip_forecast_probability_matching_blended)
        precip_forecast_probability_matching_blended[nan_indices] = np.nanmin(
            precip_forecast_probability_matching_blended
        )

        # Apply the masking and prob. matching
        precip_field_mask_temp = None
        if self.__config.mask_method is not None:
            # apply the precipitation mask to prevent generation of new
            # precipitation into areas where it was not originally
            # observed
            precip_forecast_min_value = (
                worker_state.final_blended_forecast_recomposed.min()
            )
            if self.__config.mask_method == "incremental":
                # The incremental mask is slightly different from the implementation in
                # nowcasts.steps.py, as it is not computed in the Lagrangian space. Instead,
                # we use precip_forecast_probability_matched and let the mask_rim increase with
                # the time step until mask_rim_max. This ensures that for the first t time
                # steps, the buffer mask keeps increasing.
                precip_field_mask = (
                    precip_forecast_probability_matching_blended
                    >= self.__params.precip_threshold
                )

                # Buffer the mask
                # Convert the precipitation field mask into an 8-bit unsigned integer mask
                obs_mask_uint8 = precip_field_mask.astype("uint8")

                # Perform an initial binary dilation using the provided structuring element
                dilated_mask = binary_dilation(obs_mask_uint8, self.__params.struct)

                # Create a binary structure element for incremental dilations
                struct_element = generate_binary_structure(2, 1)

                # Initialize a floating-point mask to accumulate dilations for a smooth transition
                accumulated_mask = dilated_mask.astype(float)

                # Iteratively dilate the mask and accumulate the results to create a grayscale rim
                mask_rim_temp = min(
                    self.__params.mask_rim + t_sub - 1, self.__params.max_mask_rim
                )
                for _ in range(mask_rim_temp):
                    dilated_mask = binary_dilation(dilated_mask, struct_element)
                    accumulated_mask += dilated_mask

                # Normalize the accumulated mask values between 0 and 1
                precip_field_mask = accumulated_mask / np.max(accumulated_mask)
                # Get the final mask
                worker_state.final_blended_forecast_recomposed = (
                    precip_forecast_min_value
                    + (
                        worker_state.final_blended_forecast_recomposed
                        - precip_forecast_min_value
                    )
                    * precip_field_mask
                )
                precip_field_mask_temp = (
                    worker_state.final_blended_forecast_recomposed
                    > precip_forecast_min_value
                )
            elif self.__config.mask_method == "obs":
                # The mask equals the most recent benchmark
                # rainfall field
                precip_field_mask_temp = (
                    precip_forecast_probability_matching_blended
                    >= self.__params.precip_threshold
                )

            # Set to min value outside of mask
            worker_state.final_blended_forecast_recomposed[~precip_field_mask_temp] = (
                precip_forecast_min_value
            )

        # If probmatching_method is not None, resample the distribution from
        # both the extrapolation cascade and the model (NWP) cascade and use
        # that for the probability matching.
        if (
            self.__config.probmatching_method is not None
            and self.__config.resample_distribution
        ):
            arr1 = worker_state.precip_extrapolated_probability_matching[
                worker_state.subtimestep_index
            ]
            arr2 = worker_state.precip_models_timestep[j]
            # resample weights based on cascade level 2.
            # Areas where one of the fields is nan are not included.
            precip_forecast_probability_matching_resampled = (
                probmatching.resample_distributions(
                    first_array=arr1,
                    second_array=arr2,
                    probability_first_array=weights_probability_matching_normalized[0],
                    randgen=worker_state.randgen_probmatching[j],
                )
            )
        else:
            precip_forecast_probability_matching_resampled = (
                precip_forecast_probability_matching_blended.copy()
            )

        if self.__config.probmatching_method == "cdf":
            # nan indices in the extrapolation nowcast
            nan_indices = np.isnan(
                worker_state.precip_extrapolated_probability_matching[
                    worker_state.subtimestep_index
                ]
            )
            # Adjust the CDF of the forecast to match the resampled distribution combined from
            # extrapolation and model fields.
            # Rainfall outside the pure extrapolation domain is not taken into account.
            if np.any(np.isfinite(worker_state.final_blended_forecast_recomposed)):
                worker_state.final_blended_forecast_recomposed = (
                    probmatching.nonparam_match_empirical_cdf(
                        worker_state.final_blended_forecast_recomposed,
                        precip_forecast_probability_matching_resampled,
                        nan_indices,
                    )
                )
                precip_forecast_probability_matching_resampled = None
        elif self.__config.probmatching_method == "mean":
            # Use R_pm_blended as benchmark field and
            mean_probabiltity_matching_forecast = np.mean(
                precip_forecast_probability_matching_resampled[
                    precip_forecast_probability_matching_resampled
                    >= self.__params.precip_threshold
                ]
            )
            no_rain_mask = (
                worker_state.final_blended_forecast_recomposed
                >= self.__params.precip_threshold
            )
            mean_precip_forecast = np.mean(
                worker_state.final_blended_forecast_recomposed[no_rain_mask]
            )
            worker_state.final_blended_forecast_recomposed[no_rain_mask] = (
                worker_state.final_blended_forecast_recomposed[no_rain_mask]
                - mean_precip_forecast
                + mean_probabiltity_matching_forecast
            )
            precip_forecast_probability_matching_resampled = None

        final_blended_forecast_single_member.append(
            worker_state.final_blended_forecast_recomposed
        )
        return final_blended_forecast_single_member

    def __measure_time(self, label, start_time):
        """
        Measure and print the time taken for a specific part of the process.

        Parameters:
        - label: A description of the part of the process being measured.
        - start_time: The timestamp when the process started (from time.time()).
        """
        if self.__config.measure_time:
            elapsed_time = time.time() - start_time
            print(f"{label} took {elapsed_time:.2f} seconds.")
            return elapsed_time
        return None


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
    norain_thr: float
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

      The above parameters have been fitted by using run_vel_pert_analysis.py
      and fit_vel_pert_params.py located in the scripts directory.

      See :py:mod:`pysteps.noise.motion` for additional documentation.
    clim_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the climatological
      skill file. Arguments can consist of: 'outdir_path', 'n_models'
      (the number of NWP models) and 'window_length' (the minimum number of
      days the clim file should have, otherwise the default is used).
    mask_kwargs: dict
      Optional dictionary containing mask keyword arguments 'mask_f',
      'mask_rim' and 'max_mask_rim', the factor defining the the mask
      increment and the (maximum) rim size, respectively.
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

    blending_config = StepsBlendingConfig(
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        blend_nwp_members=blend_nwp_members,
        precip_threshold=precip_thr,
        norain_threshold=norain_thr,
        kmperpixel=kmperpixel,
        timestep=timestep,
        extrapolation_method=extrap_method,
        decomposition_method=decomp_method,
        bandpass_filter_method=bandpass_filter_method,
        noise_method=noise_method,
        noise_stddev_adj=noise_stddev_adj,
        ar_order=ar_order,
        velocity_perturbation_method=vel_pert_method,
        weights_method=weights_method,
        conditional=conditional,
        probmatching_method=probmatching_method,
        mask_method=mask_method,
        resample_distribution=resample_distribution,
        smooth_radar_mask_range=smooth_radar_mask_range,
        seed=seed,
        num_workers=num_workers,
        fft_method=fft_method,
        domain=domain,
        outdir_path_skill=outdir_path_skill,
        extrapolation_kwargs=extrap_kwargs,
        filter_kwargs=filter_kwargs,
        noise_kwargs=noise_kwargs,
        velocity_perturbation_kwargs=vel_pert_kwargs,
        climatology_kwargs=clim_kwargs,
        mask_kwargs=mask_kwargs,
        measure_time=measure_time,
        callback=callback,
        return_output=return_output,
    )

    """
    With the new refactoring, the blending nowcaster is a class that can be used in multiple ways.
    This method is here to ensure that the class can be used in a similar way as the old function.
    The new refactoring provides more possibilities, eg. when doing multiple forecasts in a row, 
    the config does not need to be provided each time
    """
    # Create an instance of the new class with all the provided arguments
    blended_nowcaster = StepsBlendingNowcaster(
        precip,
        precip_models,
        velocity,
        velocity_models,
        timesteps,
        issuetime,
        blending_config,
    )

    forecast_steps_nowcast = blended_nowcaster.compute_forecast()
    return forecast_steps_nowcast


# TODO: Where does this piece of code best fit: in utils or inside the class?
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


# TODO: Where does this piece of code best fit: in utils or inside the class?
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


# TODO: Where does this piece of code best fit: in utils or inside the class?
def calculate_weights_spn(correlations, covariance):
    """Calculate SPN blending weights for STEPS blending from correlation.

    Parameters
    ----------
    correlations : array-like
      Array of shape [n_components]
      containing correlation (skills) for each component (NWP models and nowcast).
    covariance : array-like
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

    if correlations.shape[0] > 1 and len(covariance) > 1:
        if isinstance(covariance, type(None)):
            raise ValueError("cov must contain a covariance matrix")
        else:
            # Make a numpy array out of cov and get the inverse
            covariance = np.where(covariance == 0.0, 10e-5, covariance)
            # Make sure the determinant of the matrix is not zero, otherwise
            # subtract 10e-5 from the cross-correlations between the models
            if np.linalg.det(covariance) == 0.0:
                covariance = covariance - 10e-5
            # Ensure the correlation of the model with itself is always 1.0
            for i, _ in enumerate(covariance):
                covariance[i][i] = 1.0
            # Use a numpy array instead of a matrix
            cov_matrix = np.array(covariance)
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


# TODO: Where does this piece of code best fit: in utils or inside the class?
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

    return combined_means, combined_sigmas
