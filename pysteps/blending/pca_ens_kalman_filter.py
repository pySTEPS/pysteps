# -*- coding: utf-8 -*-
"""
pysteps.blending.pca_ens_kalman_filter
======================================

Implementation of the reduced-space ensemble Kalman filter method described in
:cite:`Nerini2019MWR`. The nowcast is iteratively corrected by NWP data using
an ensemble Kalman filter in principal component (PC) space. The reduced-space
ensemble Kalman filter method consists of the following main steps:

Initialization step
-------------------
1. Set the radar rainfall fields in a Lagrangian space.
2. Perform the cascade decomposition for the input radar rainfall fields.
3. Estimate AR parameters for the extrapolation nowcast and noise cascade.
4. Initialize the noise method and precompute a set of noise fields.
5. Initialize forecast models equal to the number of ensemble members.
6. Initialize the ensemble Kalman filter method.
7. Start the forecasting loop.

Forecast step
-------------
1. Decompose the rainfall forecast field of the previous timestep.
2. Update the common precipitation mask of nowcast and NWP fields for noise imprint.
3. Iterate the AR model.
4. Recompose the rainfall forecast field.
5. (Optional) Apply probability matching.
6. Extrapolate the recomposed rainfall field to the current timestep.

Correction step
---------------
1. Identify grid boxes where rainfall is forecast.
2. Reduce nowcast and NWP ensembles onto these grid boxes and apply principal
   component analysis to further reduce the dimensionality.
3. Apply the update step of the ensemble Kalman filter.

Finalization
------------
1. Set no-data values in the final forecast fields.
2. The original approach iterates between forecast and correction steps.
   However, to reduce smoothing effects in this implementation, a pure
   forecast step is computed at the first forecast timestep, and afterwards
   the method alternates between correction and forecast steps. The smoothing
   effects arise due to the NWP effective horizontal resolution and due to
   the spatial decomposition at each forecast timestep.

.. autosummary::
    :toctree: ../generated/

    forecast
"""
import time
import datetime
from copy import deepcopy

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
)

from pysteps import blending, cascade, extrapolation, noise, utils
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.timeseries import autoregression, correlation
from pysteps.blending.ens_kalman_filter_methods import MaskedEnKF
from pysteps.postprocessing import probmatching
from pysteps.utils.check_norain import check_norain

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class EnKFCombinationConfig:
    """
    Parameters
    ----------

    n_ens_members: int
        The number of ensemble members to generate. This number should always be
        equal to the number of NWP ensemble members / number of NWP models.
    n_cascade_levels: int
        The number of cascade levels to use. Defaults to 6,
        see issue #385 on GitHub.
    precip_threshold: float
        Specifies the threshold value for minimum observable precipitation
        intensity.
    norain_threshold: float
        Specifies the threshold value for the fraction of rainy (see above) pixels
        in the radar rainfall field below which we consider there to be no rain.
        Depends on the amount of clutter typically present.
    precip_mask_dilation: int
        Number of grid boxes by which the precipitation mask should be extended per
        timestep.
    extrapolation_method: str
        Name of the extrapolation method to use. See the documentation of
        :py:mod:`pysteps.extrapolation.interface`.
    decomposition_method: str, {'fft'}
        Name of the cascade decomposition method to use. See the documentation
        of :py:mod:`pysteps.cascade.interface`.
    bandpass_filter_method: str, {'gaussian', 'uniform'}
        Name of the bandpass filter method to use with the cascade decomposition.
        See the documentation of :py:mod:`pysteps.cascade.interface`.
    noise_method: str, {'parametric','nonparametric','ssft','nested',None}
        Name of the noise generator to use for perturbating the precipitation
        field. See the documentation of :py:mod:`pysteps.noise.interface`. If set to
        None, no noise is generated.
    enkf_method: str, {'masked_enkf'}
        Name of the ensemble Kalman filter method to use for the correction step.
        Currently, only 'masked_enkf' is implemented. This method corresponds to the
        reduced-space ensemble Kalman filter method described by Nerini et al., 2019.
    enable_combination: bool
        Flag to specify whether the correction step or only the forecast steps should
        be processed.
    noise_stddev_adj: str, {'auto','fixed',None}
        Optional adjustment for the standard deviations of the noise fields added
        to each cascade level. This is done to compensate incorrect std. dev.
        estimates of casace levels due to presence of no-rain areas. 'auto'=use
        the method implemented in :py:func:`pysteps.noise.utils.
        compute_noise_stddev_adjs`.
        'fixed'= use the formula given in :cite:`BPS2006` (eq. 6), None=disable
        noise std. dev adjustment.
    ar_order: int
        The order of the autoregressive model to use. Currently, only AR(1) is
        implemented.
    seed: int
        Optional seed number for the random generators.
    num_workers: int
        The number of workers to use for parallel computation. Applicable if dask
        is enabled or pyFFTW is used for computing the FFT. When num_workers>1, it
        is advisable to disable OpenMP by setting the environment variable
        OMP_NUM_THREADS to 1. This avoids slowdown caused by too many simultaneous
        threads.
    fft_method: str
        A string defining the FFT method to use (see FFT methods in
        :py:func:`pysteps.utils.interface.get_method`).
        Defaults to 'numpy' for compatibility reasons. If pyFFTW is installed,
        the recommended method is 'pyfftw'.
    domain: str, {"spatial", "spectral"}
        If "spatial", all computations are done in the spatial domain (the
        classical STEPS model). If "spectral", the AR(2) models and stochastic
        perturbations are applied directly in the spectral domain to reduce
        memory footprint and improve performance :cite:`PCH2019b`.
    extrapolation_kwargs: dict
        Optional dictionary containing keyword arguments for the extrapolation
        method. See the documentation of :py:func:`pysteps.extrapolation.interface`.
    filter_kwargs: dict
        Optional dictionary containing keyword arguments for the filter method.
        See the documentation of :py:mod:`pysteps.cascade.bandpass_filters`.
    noise_kwargs: dict
        Optional dictionary containing keyword arguments for the initializer of
        the noise generator. See the documentation of :py:mod:`pysteps.noise.
        fftgenerators`.
    combination_kwargs: dict
        Optional dictionary containing keyword arguments for the initializer of the
        correction step. Options are: {nwp_hres_eff: float, the effective horizontal
        resolution of the utilized NWP model; prob_matching: str, specifies the
        probability matching method that should be applied}. See the documentation of
        :py:mod:`pysteps.blending.ens_kalman_filter_methods`.
    measure_time: bool
        If set to True, measure, print and return the computation time.
    callback: function, optional
      Optional function that is called after computation of each time step of
      the nowcast. The function takes one argument: a three-dimensional array
      of shape (n_ens_members,h,w), where h and w are the height and width
      of the input field precip, respectively. This can be used, for instance,
      writing the outputs into files.
    return_output: bool
        Set to False to disable returning the outputs as numpy arrays. This can
        save memory if the intermediate results are written to output files using
        the callback function. (Call back function is currently not implemented.)
    n_noise_fields: int
        Number of precomputed noise fields. A number of 30 is adequate to generate
        sufficient spread in the Nowcast.
    """

    n_ens_members: int
    n_cascade_levels: int
    precip_threshold: float | None
    norain_threshold: float
    precip_mask_dilation: int
    extrapolation_method: str
    decomposition_method: str
    bandpass_filter_method: str
    noise_method: str | None
    enkf_method: str | None
    enable_combination: bool
    noise_stddev_adj: str | None
    ar_order: int
    seed: int | None
    num_workers: int
    fft_method: str
    domain: str
    extrapolation_kwargs: dict[str, Any] = field(default_factory=dict)
    filter_kwargs: dict[str, Any] = field(default_factory=dict)
    noise_kwargs: dict[str, Any] = field(default_factory=dict)
    combination_kwargs: dict[str, Any] = field(default_factory=dict)
    measure_time: bool = False
    callback: Any | None = None
    return_output: bool = True
    n_noise_fields: int = 30


@dataclass
class EnKFCombinationParams:
    noise_std_coeffs: np.ndarray | None = None
    bandpass_filter: Any | None = None
    fft: Any | None = None
    perturbation_generator: Callable[..., np.ndarray] | None = None
    noise_generator: Callable[..., np.ndarray] | None = None
    PHI: np.ndarray | None = None
    extrapolation_method: Callable[..., Any] | None = None
    decomposition_method: Callable[..., dict] | None = None
    recomposition_method: Callable[..., np.ndarray] | None = None
    fft_objs: list[Any] = field(default_factory=list)
    xy_coordinates: np.ndarray | None = None
    precip_threshold: float | None = None
    mask_threshold: np.ndarray | None = None
    num_ensemble_workers: int | None = None
    domain_mask: np.ndarray | None = None
    extrapolation_kwargs: dict | None = None
    filter_kwargs: dict | None = None
    noise_kwargs: dict | None = None
    combination_kwargs: dict | None = None
    len_y: int | None = None
    len_x: int | None = None
    no_rain_case: str | None = None


class ForecastInitialization:
    """
    Class to bundle the steps necessary for the forecast initialization.
    These steps are:

    #. Set the radar rainfall fields in a Lagrangian space.
    #. Perform the cascade decomposition for the input radar rainfall fields.
    #. Estimate AR parameters for the extrapolation nowcast and noise cascade.
    #. Initialize the noise method and precompute a set of noise fields.
    """

    def __init__(
        self,
        enkf_combination_config: EnKFCombinationConfig,
        enkf_combination_params: EnKFCombinationParams,
        obs_precip: np.ndarray,
        obs_velocity: np.ndarray,
    ):
        self.__config = enkf_combination_config
        self.__params = enkf_combination_params

        self.__obs_precip = obs_precip
        self.__obs_velocity = obs_velocity

        # Measure time for initialization.
        if self.__config.measure_time:
            self.__start_time_init = time.time()

        self.__initialize_nowcast_components()

        self.__prepare_radar_data_and_ar_parameters()

        self.__initialize_noise()

        self.__initialize_noise_field_pool()

        if self.__config.measure_time:
            print(
                f"Elapsed time for initialization:    {time.time() - self.__start_time_init}"
            )

    # Initialize FFT, bandpass filters, decomposition methods, and extrapolation
    # method.
    def __initialize_nowcast_components(self):

        # Initialize number of ensemble workers
        self.__params.num_ensemble_workers = min(
            self.__config.n_ens_members,
            self.__config.num_workers,
        )

        # Extract the spatial dimensions of the observed precipitation (x, y)
        self.__params.len_y, self.__params.len_x = self.__obs_precip.shape[1:]

        # Generate the mesh grid for spatial coordinates
        x_values, y_values = np.meshgrid(
            np.arange(self.__params.len_x),
            np.arange(self.__params.len_y),
        )
        self.__params.xy_coordinates = np.stack([x_values, y_values])

        # Initialize FFT method
        self.__params.fft = utils.get_method(
            self.__config.fft_method,
            shape=(
                self.__params.len_y,
                self.__params.len_x,
            ),
            n_threads=self.__config.num_workers,
        )

        # Initialize the band-pass filter for the cascade decomposition
        filter_method = cascade.get_method(self.__config.bandpass_filter_method)
        self.__params.bandpass_filter = filter_method(
            (self.__params.len_y, self.__params.len_x),
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

        # Determine the domain mask from non-finite values in the precipitation data
        self.__params.domain_mask = np.logical_or.reduce(
            [
                ~np.isfinite(self.__obs_precip[i, :])
                for i in range(self.__obs_precip.shape[0])
            ]
        )

        print("Nowcast components initialized successfully.")

    # Prepare radar precipitation fields for nowcasting and estimate the AR
    # parameters.
    def __prepare_radar_data_and_ar_parameters(self):
        """
        Prepare radar and NWP precipitation fields for nowcasting.
        This includes generating a threshold mask, transforming fields into
        Lagrangian coordinates, cascade decomposing/recomposing, and checking
        for zero-precip areas. The results are stored in class attributes.

        Estimate autoregressive (AR) parameters for the radar rainfall field. If
        precipitation exists, compute temporal auto-correlations; otherwise, use
        predefined climatological values. Adjust coefficients if necessary and
        estimate AR model parameters.
        """

        # Start with the radar rainfall fields. We want the fields in a Lagrangian
        # space. Advect the previous precipitation fields to the same position with
        # the most recent one (i.e. transform them into the Lagrangian coordinates).
        self.__params.extrapolation_kwargs["xy_coords"] = self.__params.xy_coordinates
        self.__params.extrapolation_kwargs["outval"] = (
            self.__config.precip_threshold - 2.0
        )
        res = []

        def transform_to_lagrangian(precip, i):
            return self.__params.extrapolation_method(
                precip[i, :, :],
                self.__obs_velocity,
                self.__config.ar_order - i,
                allow_nonfinite_values=True,
                **self.__params.extrapolation_kwargs.copy(),
            )[-1]

        if not DASK_IMPORTED:
            # Process each earlier precipitation field directly
            for i in range(self.__config.ar_order):
                self.__obs_precip[i, :, :] = transform_to_lagrangian(
                    self.__obs_precip, i
                )
        else:
            # Use Dask delayed for parallelization if DASK_IMPORTED is True
            for i in range(self.__config.ar_order):
                res.append(dask.delayed(transform_to_lagrangian)(self.__obs_precip, i))
            num_workers_ = (
                len(res)
                if self.__config.num_workers > len(res)
                else self.__config.num_workers
            )
            self.__obs_precip = np.stack(
                list(dask.compute(*res, num_workers=num_workers_))
                + [self.__obs_precip[-1, :, :]]
            )

        # Mask the observations
        obs_mask = np.logical_or(
            ~np.isfinite(self.__obs_precip),
            self.__obs_precip < self.__config.precip_threshold,
        )
        self.__obs_precip[obs_mask] = self.__config.precip_threshold - 2.0

        # Compute the cascade decompositions of the input precipitation fields
        precip_forecast_decomp = []
        for i in range(self.__config.ar_order + 1):
            precip_forecast = self.__params.decomposition_method(
                self.__obs_precip[i, :, :],
                self.__params.bandpass_filter,
                mask=self.__params.mask_threshold,
                fft_method=self.__params.fft,
                output_domain=self.__config.domain,
                normalize=True,
                compute_stats=True,
                compact_output=False,
            )
            precip_forecast_decomp.append(precip_forecast)

        # Rearrange the cascaded into a four-dimensional array of shape
        # (n_cascade_levels,ar_order+1,m,n) for the autoregressive model
        self.precip_cascades = nowcast_utils.stack_cascades(
            precip_forecast_decomp, self.__config.n_cascade_levels
        )

        # Set the mean and standard deviations based on the most recent field.
        precip_forecast_decomp = precip_forecast_decomp[-1]
        self.mean_extrapolation = np.array(precip_forecast_decomp["means"])
        self.std_extrapolation = np.array(precip_forecast_decomp["stds"])

        if self.__params.no_rain_case == "obs":

            GAMMA = np.ones((self.__config.n_cascade_levels, self.__config.ar_order))

        else:

            # If there are values in the radar fields, compute the auto-correlations
            GAMMA = np.empty((self.__config.n_cascade_levels, self.__config.ar_order))

            # compute lag-l temporal auto-correlation coefficients for each cascade level
            for i in range(self.__config.n_cascade_levels):
                GAMMA[i, :] = correlation.temporal_autocorrelation(
                    self.precip_cascades[i], mask=self.__params.mask_threshold
                )

        # Print the GAMMA value
        nowcast_utils.print_corrcoefs(GAMMA)

        if self.__config.ar_order == 2:
            # Adjust the lag-2 correlation coefficient to ensure that the AR(p)
            # process is stationary
            for i in range(self.__config.n_cascade_levels):
                GAMMA[i, 1] = autoregression.adjust_lag2_corrcoef2(
                    GAMMA[i, 0], GAMMA[i, 1]
                )

        # Estimate the parameters of the AR(p) model from the auto-correlation
        # coefficients
        self.__params.PHI = np.empty(
            (self.__config.n_cascade_levels, self.__config.ar_order + 1)
        )
        for i in range(self.__config.n_cascade_levels):
            self.__params.PHI[i, :] = autoregression.estimate_ar_params_yw(GAMMA[i, :])

        nowcast_utils.print_ar_params(self.__params.PHI)

    # Initialize the noise generation and get n_noise_fields.
    def __initialize_noise(self):
        """
        Initialize noise-based perturbations if configured, computing any required
        adjustment coefficients and setting up the perturbation generator.
        """
        if (
            self.__config.noise_method is not None
            and self.__params.no_rain_case != "obs"
        ):

            # get methods for perturbations
            init_noise, self.__params.noise_generator = noise.get_method(
                self.__config.noise_method
            )

            self.__precip_noise_input = self.__obs_precip.copy()

            # initialize the perturbation generator for the precipitation field
            self.__params.perturbation_generator = init_noise(
                self.__precip_noise_input,
                fft_method=self.__params.fft,
                **self.__params.noise_kwargs,
            )

            if self.__config.noise_stddev_adj == "auto":
                print("Computing noise adjustment coefficients... ", end="", flush=True)
                precip_forecast_min = np.min(self.__precip_noise_input)
                self.__params.noise_std_coeffs = noise.utils.compute_noise_stddev_adjs(
                    self.__precip_noise_input[-1, :, :],
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

    # Create a pool of n noise fields.
    def __initialize_noise_field_pool(self):
        """
        Initialize a pool of noise fields avoiding the separate generation of noise fields for each
        time step and ensemble member. A pool of 30 fields is sufficient to generate adequate spread
        in the nowcast for combination.
        """
        self.noise_field_pool = np.zeros(
            (
                self.__config.n_noise_fields,
                self.__config.n_cascade_levels,
                self.__params.len_y,
                self.__params.len_x,
            )
        )

        # Get a seed value for each ensemble member
        seed = self.__config.seed
        if self.__config.noise_method is not None:
            self.__randgen_precip = []
            # for j in range(self.__config.n_ens_members):
            for j in range(self.__config.n_noise_fields):
                rs = np.random.RandomState(seed)
                self.__randgen_precip.append(rs)
                seed = rs.randint(0, high=1e9)

        # Get the decomposition method
        self.__params.fft_objs = []
        for _ in range(self.__config.n_noise_fields):
            self.__params.fft_objs.append(
                utils.get_method(
                    self.__config.fft_method,
                    shape=self.precip_cascades.shape[-2:],
                )
            )

        if self.__params.noise_generator is not None:

            # Determine the noise field for each ensemble member
            for j in range(self.__config.n_noise_fields):
                epsilon = self.__params.noise_generator(
                    self.__params.perturbation_generator,
                    randstate=self.__randgen_precip[j],
                    fft_method=self.__params.fft_objs[j],
                    domain=self.__config.domain,
                )
                # Decompose the noise field into a cascade
                self.noise_field_pool[j] = self.__params.decomposition_method(
                    epsilon,
                    self.__params.bandpass_filter,
                    fft_method=self.__params.fft_objs[j],
                    input_domain=self.__config.domain,
                    output_domain=self.__config.domain,
                    compute_stats=False,
                    normalize=True,
                    compact_output=True,
                )["cascade_levels"]


class ForecastState:
    """
    Common memory of ForecastModel instances.
    """

    def __init__(
        self,
        enkf_combination_config: EnKFCombinationConfig,
        enkf_combination_params: EnKFCombinationParams,
        noise_field_pool: np.ndarray,
        latest_obs: np.ndarray,
        precip_mask: np.ndarray,
    ):

        self.config = enkf_combination_config
        self.params = enkf_combination_params
        self.noise_field_pool = noise_field_pool
        self.precip_mask = np.repeat(
            precip_mask[None, :], self.config.n_ens_members, axis=0
        )

        latest_obs[~np.isfinite(latest_obs)] = self.config.precip_threshold - 2.0
        self.nwc_prediction = np.repeat(
            latest_obs[None, :, :], self.config.n_ens_members, axis=0
        )
        self.fc_resampled = np.repeat(
            latest_obs[None, :, :], self.config.n_ens_members, axis=0
        )
        self.nwc_prediction_btf = self.nwc_prediction.copy()

        self.final_combined_forecast = []

        return


class ForecastModel:
    """
    Class to manage the forecast step of each ensemble member.
    """

    def __init__(
        self,
        forecast_state: ForecastState,
        precip_cascades: np.ndarray,
        velocity: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        ens_member: int,
    ):

        # Initialize instance variables
        self.__forecast_state = forecast_state
        self.__precip_cascades = precip_cascades
        self.__velocity = velocity

        self.__mu = mu
        self.__sigma = sigma

        self.__previous_displacement = np.zeros(
            (2, self.__forecast_state.params.len_y, self.__forecast_state.params.len_x)
        )

        # Get NWP effective horizontal resolution and type of probability matching from
        # combination kwargs.
        # It's not the best practice to mix parameters. Maybe the cascade mask as well
        # as the probability matching should be implemented at another location.
        self.__nwp_hres_eff = self.__forecast_state.params.combination_kwargs.get(
            "nwp_hres_eff", 0.0
        )
        self.__prob_matching = self.__forecast_state.params.combination_kwargs.get(
            "prob_matching", "iterative"
        )

        # Get spatial scales whose central wavelengths are above the effective
        # horizontal resolution of the NWP model.
        # Factor 3 on the effective resolution is similar to that factor of the
        # localization of AR parameters and scaling parameters.
        self.__resolution_mask = (
            self.__forecast_state.params.len_y
            / self.__forecast_state.params.bandpass_filter["central_wavenumbers"]
            >= self.__nwp_hres_eff * 3.0
        )

        self.__ens_member = ens_member

    # Bundle single steps of the forecast.
    def run_forecast_step(self, nwp, is_correction_timestep=False):

        # Decompose precipitation field.
        self.__decompose(is_correction_timestep)

        # Update precipitation mask.
        self.__update_precip_mask(nwp=nwp)

        # Iterate through the AR process.
        self.__iterate()

        # Recompose the precipitation field for the correction step.
        self.__forecast_state.nwc_prediction[self.__ens_member] = (
            blending.utils.recompose_cascade(
                combined_cascade=self.__precip_cascades[:, -1],
                combined_mean=self.__mu,
                combined_sigma=self.__sigma,
            )
        )

        # Apply probability matching
        if self.__prob_matching == "iterative":
            self.__probability_matching()

        # Extrapolate the precipitation field onto the position of the current timestep.
        self.__advect()

    # Create the resulting precipitation field and set no data area. In future, when
    # transformation between linear and logarithmic scale will be necessary, it will be
    # implemented in this function.
    def backtransform(self):

        # Set the resulting field as shallow copy of the field that is used
        # continuously for forecast computation.
        self.__forecast_state.nwc_prediction_btf[self.__ens_member] = (
            self.__forecast_state.nwc_prediction[self.__ens_member]
        )

        # Set no data area
        self.__set_no_data()

    # Call spatial decomposition function and compute an adjusted standard deviation of
    # each spatial scale at timesteps where NWP information is incorporated.
    def __decompose(self, is_correction_timestep):

        # Call spatial decomposition method.
        precip_extrap_decomp = self.__forecast_state.params.decomposition_method(
            self.__forecast_state.nwc_prediction[self.__ens_member],
            self.__forecast_state.params.bandpass_filter,
            fft_method=self.__forecast_state.params.fft_objs[self.__ens_member],
            input_domain=self.__forecast_state.config.domain,
            output_domain=self.__forecast_state.config.domain,
            compute_stats=False,
            normalize=True,
            compact_output=False,
        )

        # Set decomposed field onto the latest precipitation cascade.
        self.__precip_cascades[:, -1] = precip_extrap_decomp["cascade_levels"]

        # If NWP information is incorporated, use the current mean of the decomposed
        # field and adjust standard deviation on spatial scales that have a central
        # wavelength below the effective horizontal resolution of the NWP model.
        if is_correction_timestep:
            # Set the mean of the spatial scales onto the mean values of the currently
            # decomposed field.
            self.__mu = np.array(precip_extrap_decomp["means"])
            # Compute the standard deviation evolved by an AR(1)-process.
            self.__sigma = np.sqrt(
                self.__forecast_state.params.PHI[:, 0] ** 2.0 * self.__sigma**2.0
                + self.__forecast_state.params.PHI[:, 1] ** 2.0
                * self.__forecast_state.params.noise_std_coeffs**2.0
            )

            # Use the standard deviations of the currently decomposed field for spatial
            # scales above the effective horizontal resolution of the NWP model.
            self.__sigma[self.__resolution_mask] = np.array(
                precip_extrap_decomp["stds"]
            )[self.__resolution_mask]
        # Else, keep mean and standard deviation constant for pure nowcasting forecast steps.
        # It's not necessary but describes better the handling of the scaling
        # parameters.
        else:
            self.__mu = self.__mu
            self.__sigma = self.__sigma

    # Call extrapolation function to extrapolate the precipitation field onto the
    # position of the current timestep.
    def __advect(self):

        # Since previous displacement is the sum of displacement over all previous
        # timesteps, we have to compute the differences between the displacements to
        # get the motion vector field for one time step.
        displacement_tmp = self.__previous_displacement.copy()

        # Call the extrapolation method
        (
            self.__forecast_state.nwc_prediction[self.__ens_member],
            self.__previous_displacement,
        ) = self.__forecast_state.params.extrapolation_method(
            self.__forecast_state.nwc_prediction[self.__ens_member],
            self.__velocity,
            [1],
            allow_nonfinite_values=True,
            displacement_previous=self.__previous_displacement,
            **self.__forecast_state.params.extrapolation_kwargs,
        )

        # Get the difference of the previous displacement field.
        self.__previous_displacement -= displacement_tmp

    # Get a noise field out of the respective pool and iterate through the AR(1)
    # process.
    def __iterate(self):

        # Get a noise field out of the noise field pool and multiply it with
        # precipitation mask and the standard deviation coefficients.
        epsilon = (
            self.__forecast_state.noise_field_pool[
                np.random.randint(self.__forecast_state.config.n_noise_fields)
            ]
            * self.__forecast_state.precip_mask[self.__ens_member][None, :, :]
            * self.__forecast_state.params.noise_std_coeffs[:, None, None]
        )

        # Iterate through the AR(1) process for each cascade level.
        for i in range(self.__forecast_state.config.n_cascade_levels):

            self.__precip_cascades[i] = autoregression.iterate_ar_model(
                self.__precip_cascades[i],
                self.__forecast_state.params.PHI[i],
                epsilon[i],
            )

    # Update the precipitation mask for the forecast step by incorporating areas
    # where the NWP model forecast precipitation.
    def __update_precip_mask(self, nwp):

        # Get the area where the NWP ensemble member forecast precipitation above
        # precipitation threshold and dilate it by a configurable range.
        precip_mask = (
            binary_dilation(
                nwp > self.__forecast_state.config.precip_threshold,
                structure=np.ones(
                    (
                        self.__forecast_state.config.precip_mask_dilation,
                        self.__forecast_state.config.precip_mask_dilation,
                    ),
                    dtype=int,
                ),
            )
            * 1.0
        )
        # Get the area where the combined member forecast precipitation above the
        # precipitation threshold and dilate it by a configurable range.
        precip_mask += (
            binary_dilation(
                self.__forecast_state.nwc_prediction[self.__ens_member]
                > self.__forecast_state.config.precip_threshold,
                structure=np.ones(
                    (
                        self.__forecast_state.config.precip_mask_dilation,
                        self.__forecast_state.config.precip_mask_dilation,
                    ),
                    dtype=int,
                ),
            )
            * 1.0
        )
        # Set values above 1 to 1 for conversion into bool.
        precip_mask[precip_mask >= 1.0] = 1.0
        # Some additional dilation of the precipitation mask.
        precip_mask = gaussian_filter(precip_mask, (1, 1))
        # Set the mask outside the radar domain to 0.
        precip_mask[self.__forecast_state.params.domain_mask] = 0.0
        # Convert mask into bool.
        self.__forecast_state.precip_mask[self.__ens_member] = np.array(
            precip_mask, dtype=bool
        )

    # Apply probability matching
    def __probability_matching(self):

        # Apply probability matching
        self.__forecast_state.nwc_prediction[self.__ens_member] = (
            probmatching.nonparam_match_empirical_cdf(
                self.__forecast_state.nwc_prediction[self.__ens_member],
                self.__forecast_state.fc_resampled[self.__ens_member],
            )
        )

    # Set no data area in the resulting precipitation field.
    def __set_no_data(self):

        self.__forecast_state.nwc_prediction_btf[self.__ens_member][
            self.__forecast_state.params.domain_mask
        ] = np.nan


class EnKFCombinationNowcaster:
    def __init__(
        self,
        obs_precip: np.ndarray,
        obs_timestamps: np.ndarray,
        nwp_precip: np.ndarray,
        nwp_timestamps: np.ndarray,
        obs_velocity: np.ndarray,
        fc_period: int,
        fc_init: datetime.datetime,
        enkf_combination_config: EnKFCombinationConfig,
    ):
        """
        Initialize EnKFCombinationNowcaster with inputs and configurations.
        """
        # Store inputs
        self.__obs_precip = obs_precip
        self.__nwp_precip = nwp_precip
        self.__obs_velocity = obs_velocity
        self.__fc_period = fc_period
        self.__fc_init = fc_init

        # Store config
        self.__config = enkf_combination_config

        # Initialize Params
        self.__params = EnKFCombinationParams()

        # Store input timestamps
        self.__obs_timestamps = obs_timestamps
        self.__nwp_timestamps = nwp_timestamps

    def compute_forecast(self):
        """
        Generate a combined nowcast ensemble by using the reduced-space ensemble Kalman
        filter method.

        Parameters
        ----------
        obs_precip: np.ndarray
            Array of shape (ar_order+1,m,n) containing the observed input precipitation
            fields ordered by timestamp from oldest to newst. The time steps between
            the inputs are assumed to be regular.
        obs_timestamps: np.ndarray
            Array of shape (ar_order+1) containing the corresponding time stamps of
            observed input precipitation fields as datetime objects.
        nwp_precip: np.ndarray
            Array of shape (n_ens,n_times,m,n) containing the (NWP) ensemble model
            forecast.
        nwp_timestamps: np.ndarray
            Array of shape (n_times) containing the corresponding time stamps of the
            (NWP) ensemble model forecast as datetime objects.
        obs_velocity: np.ndarray
            Array of shape (2,m,n) containing the x- and y-components of the advection
            field. The velocities are based on the observed input precipitation fields
            and are assumed to represent one time step between the inputs. All values
            are required to be finite.
        fc_period: int
            Forecast range in minutes.
        fc_init: datetime object
            Issuetime of the combined forecast to compute.
        enkf_combination_config: EnKFCombinationConfig
            Provides a set of configuration parameters for the nowcast ensemble
            generation.

        Returns
        -------
        out: np.ndarray
          If return_output is True, a four-dimensional array of shape
          (n_ens_members,num_timesteps,m,n) containing a time series of forecast
          precipitation fields for each ensemble member. Otherwise, a None value
          is returned. The time series starts from t0. The timestep is taken from the
          input precipitation fields precip.

        See also
        --------
        :py:mod:`pysteps.extrapolation.interface`, :py:mod:`pysteps.cascade.interface`,
        :py:mod:`pysteps.noise.interface`, :py:func:`pysteps.noise.utils.
        compute_noise_stddev_adjs`

        References
        ----------
        :cite:`Nerini2019MWR`

        Notes
        -----
        1. The combination method currently supports only an AR(1) process for the
        forecast step.
        """

        # Check for the inputs.
        self.__check_inputs()

        # Check timestamps of radar and nwp input and determine forecast and correction
        # timesteps as well as the temporal resolution
        self.__check_input_timestamps()

        # Check wehther there is no precipitation in observation, but in NWP or the other way around
        self.__check_no_rain_case()

        # Print forecast information.
        self.__print_forecast_info()

        # Initialize and compute the forecast initialization.
        self.FI = ForecastInitialization(
            self.__config, self.__params, self.__obs_precip, self.__obs_velocity
        )

        # NWP: Set values below precip thr and nonfinite values to norain thr.
        nwp_mask = np.logical_or(
            ~np.isfinite(self.__nwp_precip),
            self.__nwp_precip < self.__config.precip_threshold,
        )
        self.__nwp_precip[nwp_mask] = self.__config.precip_threshold - 2.0

        # Set an initial precipitation mask for the NWC models.
        precip_mask = binary_dilation(
            self.__obs_precip[-1] > self.__config.precip_threshold,
            structure=np.ones(
                (self.__config.precip_mask_dilation, self.__config.precip_mask_dilation)
            ),
        )

        # Initialize an instance of NWC forecast model class for each ensemble member.
        self.FS = ForecastState(
            enkf_combination_config=self.__config,
            enkf_combination_params=self.__params,
            noise_field_pool=self.FI.noise_field_pool,
            latest_obs=self.__obs_precip[-1, :, :],
            precip_mask=precip_mask.copy(),
        )

        self.FC_Models = {}
        for j in range(self.__config.n_ens_members):
            FC = ForecastModel(
                forecast_state=self.FS,
                precip_cascades=deepcopy(self.FI.precip_cascades),
                velocity=self.__obs_velocity,
                mu=deepcopy(self.FI.mean_extrapolation),
                sigma=deepcopy(self.FI.std_extrapolation),
                ens_member=j,
            )
            self.FC_Models[j] = FC

        # Initialize the combination model.
        if self.__config.enkf_method == "masked_enkf":
            kalman_filter_model = MaskedEnKF
        else:
            raise ValueError(
                "Currently, only 'masked_enkf' is implemented as ensemble"
                "Kalman filter method!"
            )
        self.KalmanFilterModel = kalman_filter_model(self.__config, self.__params)

        # Start the main forecast loop.
        self.__integrated_nowcast_main_loop()

        # Stack and return the forecast output.
        if self.__config.return_output:
            self.FS.final_combined_forecast = np.array(
                self.FS.final_combined_forecast
            ).swapaxes(0, 1)

            if self.__config.measure_time:
                return (
                    self.FS.final_combined_forecast,
                    self.__fc_init,
                    self.__mainloop_time,
                )
            return self.FS.final_combined_forecast

        # Else, return None
        return None

    def __check_inputs(self):
        """
        Validates user's input.
        """

        # Check dimensions of obs precip
        if self.__obs_precip.ndim != 3:
            raise ValueError(
                "Precipitation observation must be a three-dimensional "
                "array of shape (ar_order + 1, m, n)"
            )
        if self.__obs_precip.shape[0] < self.__config.ar_order + 1:
            raise ValueError(
                f"Precipitation observation must have at least "
                f"{self.__config.ar_order + 1} time steps in the first"
                f"dimension to match the autoregressive order "
                f"(ar_order={self.__config.ar_order})"
            )

        # If it is necessary, slice the precipitation field to only use the last
        # ar_order +1 time steps.
        if self.__obs_precip.shape[0] > self.__config.ar_order + 1:
            self.__obs_precip = np.delete(
                self.__obs_precip,
                np.arange(
                    0, self.__obs_precip.shape[0] - (self.__config.ar_order + 1), 1
                ),
                axis=0,
            )

        # Check NWP data dimensions
        NWP_shape = self.__nwp_precip.shape
        NWP_timestamps_len = len(self.__nwp_timestamps)
        if not NWP_timestamps_len in NWP_shape:
            raise ValueError(
                f"nwp_timestamps has not the same length as NWP data!"
                f"nwp_timestamps length: {NWP_timestamps_len}"
                f"nwp_precip shape:      {NWP_shape}"
            )

        # Ensure that model has shape: [n_ens_members, t, y, x]
        # n_ens_members and t can sometimes be swapped when using grib datasets.
        # Check for temporal resolution of NWP data
        if NWP_shape[0] == NWP_timestamps_len:
            self.__nwp_precip = self.__nwp_precip.swapaxes(0, 1)

        # Check dimensions of obs velocity
        if self.__obs_velocity.ndim != 3:
            raise ValueError(
                "The velocity field must be a three-dimensional array of shape (2, m, n)"
            )

        # Check whether the spatial dimensions match between obs precip and
        # obs velocity
        if self.__obs_precip.shape[1:3] != self.__obs_velocity.shape[1:3]:
            raise ValueError(
                f"Spatial dimension of Precipitation observation and the"
                "velocity field do not match: "
                f"{self.__obs_precip.shape[1:3]} vs. {self.__obs_velocity.shape[1:3]}"
            )

        # Check velocity field for non-finite values
        if np.any(~np.isfinite(self.__obs_velocity)):
            raise ValueError("Velocity contains non-finite values")

        # Check whether there are extrapolation kwargs
        if self.__config.extrapolation_kwargs is None:
            self.__params.extrapolation_kwargs = dict()
        else:
            self.__params.extrapolation_kwargs = deepcopy(
                self.__config.extrapolation_kwargs
            )

        # Check whether there are filter kwargs
        if self.__config.filter_kwargs is None:
            self.__params.filter_kwargs = dict()
        else:
            self.__params.filter_kwargs = deepcopy(self.__config.filter_kwargs)

        # Check for noise kwargs
        if self.__config.noise_kwargs is None:
            self.__params.noise_kwargs = {"win_fun": "tukey"}
        else:
            self.__params.noise_kwargs = deepcopy(self.__config.noise_kwargs)

        # Check for combination kwargs
        if self.__config.combination_kwargs is None:
            self.__params.combination_kwargs = dict()
        else:
            self.__params.combination_kwargs = deepcopy(
                self.__config.combination_kwargs
            )

        # Set the precipitation threshold also in params
        self.__params.precip_threshold = self.__config.precip_threshold

        # Check for the standard deviation adjustment of the noise fields
        if self.__config.noise_stddev_adj not in ["auto", "fixed", None]:
            raise ValueError(
                f"Unknown noise_std_dev_adj method {self.__config.noise_stddev_adj}. "
                "Must be 'auto', 'fixed', or None"
            )

    def __check_input_timestamps(self):
        """
        Check for timestamps of radar data and NWP data, determine forecasts and
        correction timesteps as well as the temporal resolution of the combined forecast
        """

        # Check for temporal resolution of radar data
        obs_time_diff = np.unique(np.diff(self.__obs_timestamps))
        if obs_time_diff.size > 1:
            raise ValueError(
                "Observation data has a different temporal resolution or "
                "observations are missing!"
            )
        self.__temporal_res = int(obs_time_diff[0].total_seconds() / 60)

        # Check for temporal resolution of NWP data
        nwp_time_diff = np.unique(np.diff(self.__nwp_timestamps))
        if nwp_time_diff.size > 1:
            raise ValueError(
                "NWP data has a different temporal resolution or some time steps are missing!"
            )
        nwp_temporal_res = int(nwp_time_diff[0].total_seconds() / 60)

        # Check whether all necessary timesteps are included in the observation
        if self.__obs_timestamps[-1] != self.__fc_init:
            raise ValueError(
                "The last observation timestamp differs from forecast issue time!"
            )
        if self.__obs_timestamps.size < self.__config.ar_order + 1:
            raise ValueError(
                f"Precipitation observation must have at least "
                f"{self.__config.ar_order + 1} time steps in the first"
                f"dimension to match the autoregressive order "
                f"(ar_order={self.__config.ar_order})"
            )

        # Check whether the NWP forecasts includes the combined forecast range
        if np.logical_or(
            self.__fc_init < self.__nwp_timestamps[0],
            self.__fc_init > self.__nwp_timestamps[-1],
        ):
            raise ValueError("Forecast issue time is not included in the NWP forecast!")

        max_nwp_fc_period = (
            self.__nwp_timestamps.size
            - np.where(self.__nwp_timestamps == self.__fc_init)[0][0]
            - 1
        ) * nwp_temporal_res
        if max_nwp_fc_period < self.__fc_period - nwp_temporal_res:
            raise ValueError(
                "The remaining NWP forecast is not sufficient for the combined forecast period"
            )

        # Truncate the NWP dataset if there sufficient remaining timesteps are available
        self.__nwp_precip = np.delete(
            self.__nwp_precip,
            np.logical_or(
                self.__nwp_timestamps <= self.__fc_init,
                self.__nwp_timestamps
                > self.__fc_init + datetime.timedelta(minutes=self.__fc_period),
            ),
            axis=1,
        )

        # Define forecast and correction timesteps assuming that temporal resolution of
        # the combined forecast is equal to that of the radar data
        self.__forecast_leadtimes = np.arange(
            0, self.__fc_period + 1, self.__temporal_res
        )
        trunc_nwp_timestamps = (
            self.__nwp_timestamps[
                np.logical_and(
                    self.__nwp_timestamps > self.__fc_init,
                    self.__nwp_timestamps
                    <= self.__fc_init + datetime.timedelta(minutes=self.__fc_period),
                )
            ]
            - self.__fc_init
        )
        self.__correction_leadtimes = np.array(
            [int(timestamp.total_seconds() / 60) for timestamp in trunc_nwp_timestamps]
        )

    def __check_no_rain_case(self):

        print("Test for no rain cases")
        print("======================")
        print("")

        # Check for zero input fields in the radar and NWP data.
        zero_precip_radar = check_norain(
            self.__obs_precip,
            self.__config.precip_threshold,
            self.__config.norain_threshold,
            self.__params.noise_kwargs["win_fun"],
        )
        # The norain fraction threshold used for nwp is the default value of 0.0,
        # since nwp does not suffer from clutter.
        zero_precip_nwp_forecast = check_norain(
            self.__nwp_precip,
            self.__config.precip_threshold,
            self.__config.norain_threshold,
            self.__params.noise_kwargs["win_fun"],
        )

        # If there is no precipitation in the observation, set no_rain_case to "obs"
        # and use only the NWP ensemble forecast
        if zero_precip_radar:
            self.__params.no_rain_case = "obs"
        # If there is no precipitation at the first usable NWP forecast timestep, but
        # in the observation, compute an extrapolation forecast
        elif zero_precip_nwp_forecast:
            self.__params.no_rain_case = "nwp"
        # Otherwise, set no_rain_case to 'none' and compute combined forecast as usual
        else:
            self.__params.no_rain_case = "none"

        return

    def __print_forecast_info(self):
        """
        Print information about the forecast configuration, including inputs, methods,
        and parameters.
        """
        print("Reduced-space ensemble Kalman filter")
        print("====================================")
        print("")

        print("Inputs")
        print("------")
        print(f"Forecast issue time:                {self.__fc_init.isoformat()}")
        print(
            f"Input dimensions:                   {self.__obs_precip.shape[1]}x{self.__obs_precip.shape[2]}"
        )
        print(f"Temporal resolution:                {self.__temporal_res} minutes")
        print("")

        print("NWP and blending inputs")
        print("-----------------------")
        print(f"Number of (NWP) models:             {self.__nwp_precip.shape[0]}")
        print("")

        print("Methods")
        print("-------")
        print(
            f"Extrapolation:                      {self.__config.extrapolation_method}"
        )
        print(
            f"Bandpass filter:                    {self.__config.bandpass_filter_method}"
        )
        print(
            f"Decomposition:                      {self.__config.decomposition_method}"
        )
        print(f"Noise generator:                    {self.__config.noise_method}")
        print(
            f"Noise adjustment:                   {'yes' if self.__config.noise_stddev_adj else 'no'}"
        )

        print(f"EnKF implementation:                {self.__config.enkf_method}")

        print(f"FFT method:                         {self.__config.fft_method}")
        print(f"Domain:                             {self.__config.domain}")
        print("")

        print("Parameters")
        print("----------")
        print(f"Forecast length in min:             {self.__fc_period}")
        print(f"Ensemble size:                      {self.__config.n_ens_members}")
        print(f"Parallel threads:                   {self.__config.num_workers}")
        print(f"Number of cascade levels:           {self.__config.n_cascade_levels}")
        print(f"Order of the AR(p) model:           {self.__config.ar_order}")
        print("")

        print(f"No rain forecast:                   {self.__params.no_rain_case}")

    def __integrated_nowcast_main_loop(self):

        if self.__config.measure_time:
            starttime_mainloop = time.time()

        self.__params.extrapolation_kwargs["return_displacement"] = True
        is_correction_timestep = False

        for t, fc_leadtime in enumerate(self.__forecast_leadtimes):
            if self.__config.measure_time:
                starttime = time.time()

            # Check whether forecast time step is also a correction time step.
            is_correction_timestep = (
                self.__forecast_leadtimes[t - 1] in self.__correction_leadtimes
                and t > 1
                and np.logical_and(
                    self.__config.enable_combination,
                    self.__params.no_rain_case != "nwp",
                )
            )

            # Check whether forecast time step is a nowcasting time step.
            is_nowcasting_timestep = t > 0

            # Check whether full NWP weight is reached.
            is_full_nwp_weight = (
                self.KalmanFilterModel.get_inflation_factor_obs() <= 0.02
                or self.__params.no_rain_case == "obs"
            )

            # If full NWP weight is reached, set pure NWP ensemble forecast in combined
            # forecast output
            if is_full_nwp_weight:

                # Set t_corr to the first available NWP data timestep and that is 0
                try:
                    t_corr
                except NameError:
                    t_corr = 0

                print(f"Full NWP weight is reached for lead time + {fc_leadtime} min")
                if is_correction_timestep:
                    t_corr = np.where(
                        self.__correction_leadtimes == self.__forecast_leadtimes[t - 1]
                    )[0][0]
                self.FS.nwc_prediction = self.__nwp_precip[:, t_corr]

            # Otherwise compute the combined forecast.
            else:
                print(f"Computing combination for lead time + {fc_leadtime} min")
                self.__forecast_loop(t, is_correction_timestep, is_nowcasting_timestep)

            # Apply back transformation
            for FC_Model in self.FC_Models.values():
                FC_Model.backtransform()

            self.__write_output()

            if self.__config.measure_time:
                _ = self.__measure_time("timestep", starttime)
            else:
                print("...done.")

        if self.__config.measure_time:
            self.__mainloop_time = time.time() - starttime_mainloop
            print(
                f"Elapsed time for computing forecast: {(self.__mainloop_time / 60.0):.4} min"
            )

        return

    def __forecast_loop(self, t, is_correction_timestep, is_nowcasting_timestep):

        # If the temporal resolution of the NWP data is equal to those of the
        # observation, the correction step can be applied after the forecast
        # step for the current forecast leadtime.
        # However, if the temporal resolution is different, the correction step
        # has to be applied before the forecast step to avoid smoothing effects
        # in the resulting precipitation fields.
        if is_correction_timestep:
            t_corr = np.where(
                self.__correction_leadtimes == self.__forecast_leadtimes[t - 1]
            )[0][0]

            self.FS.nwc_prediction, self.FS.fc_resampled = (
                self.KalmanFilterModel.correct_step(
                    self.FS.nwc_prediction,
                    self.__nwp_precip[:, t_corr],
                    self.FS.fc_resampled,
                )
            )

        # Run nowcasting time step
        if is_nowcasting_timestep:

            # Set t_corr to the first available NWP data timestep and that is 0
            try:
                t_corr
            except NameError:
                t_corr = 0

            def worker(j):

                self.FC_Models[j].run_forecast_step(
                    nwp=self.__nwp_precip[j, t_corr],
                    is_correction_timestep=is_correction_timestep,
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

    def __write_output(self):

        if (
            self.__config.callback is not None
            and self.FS.nwc_prediction_btf.shape[1] > 0
        ):
            self.__config.callback(self.FS.nwc_prediction_btf)

        if self.__config.return_output:

            self.FS.final_combined_forecast.append(self.FS.nwc_prediction_btf.copy())

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
    obs_precip,
    obs_timestamps,
    nwp_precip,
    nwp_timestamps,
    velocity,
    forecast_horizon,
    issuetime,
    n_ens_members,
    precip_mask_dilation=1,
    n_cascade_levels=6,
    precip_thr=-10.0,
    norain_thr=0.01,
    extrap_method="semilagrangian",
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    noise_method="nonparametric",
    enkf_method="masked_enkf",
    enable_combination=True,
    noise_stddev_adj=None,
    ar_order=1,
    callback=None,
    return_output=True,
    seed=None,
    num_workers=1,
    fft_method="numpy",
    domain="spatial",
    extrap_kwargs=None,
    filter_kwargs=None,
    noise_kwargs=None,
    combination_kwargs=None,
    measure_time=False,
):
    """
    Generate a combined nowcast ensemble by using the reduced-space ensemble Kalman
    filter method described in Nerini et al. 2019.

    Parameters
    ----------
    obs_precip: np.ndarray
        Array of shape (ar_order+1,m,n) containing the observed input precipitation
        fields ordered by timestamp from oldest to newst. The time steps between
        the inputs are assumed to be regular.
    obs_timestamps: np.ndarray
        Array of shape (ar_order+1) containing the corresponding time stamps of
        observed input precipitation fields as datetime objects.
    nwp_precip: np.ndarray
        Array of shape (n_ens,n_times,m,n) containing the (NWP) ensemble model
        forecast.
    nwp_timestamps: np.ndarray
        Array of shape (n_times) containing the corresponding time stamps of the
        (NWP) ensemble model forecast as datetime objects.
    velocity: np.ndarray
        Array of shape (2,m,n) containing the x- and y-components of the advection
        field. The velocities are based on the observed input precipitation fields
        and are assumed to represent one time step between the inputs. All values
        are required to be finite.
    forecast_horizon: int
        The length of the forecast horizon (the length of the forecast) in minutes.
    issuetime: datetime object
        Issuetime of the combined forecast to compute.
    n_ens_members: int
        The number of ensemble members to generate. This number should always be
        equal to or larger than the number of NWP ensemble members / number of
        NWP models.
    precip_mask_dilation: int
        Range by which the precipitation mask within the forecast step should be
        extended per time step. Defaults to 1.
    n_cascade_levels: int, optional
        The number of cascade levels to use. Defaults to 6, see issue #385 on GitHub.
    precip_thr: float, optional
        pecifies the threshold value for minimum observable precipitation
        intensity. Required if mask_method is not None or conditional is True.
        Defaults to -10.0.
    norain_thr: float
        Specifies the threshold value for the fraction of rainy (see above) pixels
        in the radar rainfall field below which we consider there to be no rain.
        Depends on the amount of clutter typically present. Defaults to -15.0.
    extrap_method: str, optional
        Name of the extrapolation method to use. See the documentation of
        :py:mod:`pysteps.extrapolation.interface`. Defaults to 'semilagrangian'.
    decomp_method: {'fft'}, optional
        Name of the cascade decomposition method to use. See the documentation
        of :py:mod:`pysteps.cascade.interface`. Defaults to 'fft'.
    bandpass_filter_method: {'gaussian', 'uniform'}, optional
        Name of the bandpass filter method to use with the cascade decomposition.
        See the documentation of :py:mod:`pysteps.cascade.interface`. Defaults to
        'guassian'.
    noise_method: {'parametric','nonparametric','ssft','nested',None}, optional
        Name of the noise generator to use for perturbating the precipitation
        field. See the documentation of :py:mod:`pysteps.noise.interface`. If set to
        None, no noise is generated. Defaults to 'nonparametric'.
    enkf_method: {'masked_enkf}, optional
        Name of the ensemble Kalman filter method to use for the correction step.
        Currently, only 'masked_enkf' method is implemented that corresponds to the
        reduced-space ensemble Kalman filter technique described in Nerini et al. 2019.
        Defaults to 'masked_enkf'.
    enable_combination: bool, optional
        Flag to specify whether the correction step should be applied or a pure
        nowcasting ensemble should be computed. Defaults to True.
    noise_stddev_adj: {'auto','fixed',None}, optional
        Optional adjustment for the standard deviations of the noise fields added
        to each cascade level. This is done to compensate incorrect std. dev.
        estimates of casace levels due to presence of no-rain areas. 'auto'=use
        the method implemented in :py:func:`pysteps.noise.utils.
        compute_noise_stddev_adjs`.
        'fixed'= use the formula given in :cite:`BPS2006` (eq. 6), None=disable
        noise std. dev adjustment.
    ar_order: int, optional
        The order of the autoregressive model to use. Must be 1, since only this order
        is currently implemented.
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
    extrap_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the extrapolation
        method. See the documentation of :py:func:`pysteps.extrapolation.interface`.
    filter_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the filter method.
        See the documentation of :py:mod:`pysteps.cascade.bandpass_filters`.
    noise_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the initializer of
        the noise generator. See the documentation of :py:mod:`pysteps.noise.
        fftgenerators`.
    combination_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the initializer of the
        ensemble Kalman filter method. See the documentation of
        :py:mod:`pysteps.blending.ens_kalman_filter_methods`.
    measure_time: bool
      If set to True, measure, print and return the computation time.

    Returns
    -------
    out: np.ndarray
        If return_output is True, a four-dimensional array of shape
        (n_ens_members,num_timesteps,m,n) containing a time series of forecast
        precipitation fields for each ensemble member. Otherwise, a None value
        is returned. The time series starts from t0. The timestep is taken from the
        input precipitation fields precip.

    See also
    --------
    :py:mod:`pysteps.extrapolation.interface`, :py:mod:`pysteps.cascade.interface`,
    :py:mod:`pysteps.noise.interface`, :py:func:`pysteps.noise.utils.
    compute_noise_stddev_adjs`

    References
    ----------
    :cite:`Nerini2019MWR`

    Notes
    -----
    1. The combination method currently supports only an AR(1) process for the
    forecast step.
    """

    combination_config = EnKFCombinationConfig(
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        precip_threshold=precip_thr,
        norain_threshold=norain_thr,
        precip_mask_dilation=precip_mask_dilation,
        extrapolation_method=extrap_method,
        decomposition_method=decomp_method,
        bandpass_filter_method=bandpass_filter_method,
        noise_method=noise_method,
        enkf_method=enkf_method,
        enable_combination=enable_combination,
        noise_stddev_adj=noise_stddev_adj,
        ar_order=ar_order,
        seed=seed,
        num_workers=num_workers,
        fft_method=fft_method,
        domain=domain,
        extrapolation_kwargs=extrap_kwargs,
        filter_kwargs=filter_kwargs,
        noise_kwargs=noise_kwargs,
        combination_kwargs=combination_kwargs,
        measure_time=measure_time,
        callback=callback,
        return_output=return_output,
        n_noise_fields=30,
    )

    combination_nowcaster = EnKFCombinationNowcaster(
        obs_precip=obs_precip,
        obs_timestamps=obs_timestamps,
        nwp_precip=nwp_precip,
        nwp_timestamps=nwp_timestamps,
        obs_velocity=velocity,
        fc_period=forecast_horizon,
        fc_init=issuetime,
        enkf_combination_config=combination_config,
    )

    forecast_enkf_combination = combination_nowcaster.compute_forecast()

    return forecast_enkf_combination
