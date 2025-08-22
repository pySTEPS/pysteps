# -*- coding: utf-8 -*-
"""
pysteps.blending.pca_enkf
=========================

Implementation of the reduced-space ensemble Kalman filter method described in
:cite:`Nerini2019`. The nowcast is iteratively corrected by NWP data utilizing
an ensemble Kalman filter in PCA space.
"""
import math
import time
import datetime
from copy import deepcopy

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    generate_binary_structure,
    iterate_structure,
    gaussian_filter,
)

from pysteps import blending, cascade, combination, extrapolation, noise, utils
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.timeseries import autoregression, correlation

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

    """

    n_ens_members: int
    n_cascade_levels: int
    precip_threshold: float | None
    norain_threshold: float
    kmperpixel: float
    precip_mask_dilation: int
    extrapolation_method: str
    decomposition_method: str
    bandpass_filter_method: str
    noise_method: str | None
    enkf_method: str | None
    noise_stddev_adj: str | None
    ar_order: int
    velocity_perturbation_method: str | None
    conditional: bool
    probmatching_method: str | None
    iter_probability_matching: bool
    post_probability_matching: bool
    non_precip_mask: bool
    seed: int | None
    num_workers: int
    fft_method: str
    domain: str
    n_tapering: int
    n_ens_prec: int
    lien_criterion: bool
    n_lien: int
    extrapolation_kwargs: dict[str, Any] = field(default_factory=dict)
    filter_kwargs: dict[str, Any] = field(default_factory=dict)
    noise_kwargs: dict[str, Any] = field(default_factory=dict)
    velocity_perturbation_kwargs: dict[str, Any] = field(default_factory=dict)
    mask_kwargs: dict[str, Any] = field(default_factory=dict)
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
    velocity_perturbations: Any | None = None
    generate_velocity_noise: Callable[[Any, float], np.ndarray] | None = None
    velocity_perturbations_parallel: np.ndarray | None = None
    velocity_perturbations_perpendicular: np.ndarray | None = None
    fft_objs: list[Any] = field(default_factory=list)
    time_steps_is_list: bool = False
    xy_coordinates: np.ndarray | None = None
    precip_threshold: float | None = None
    mask_threshold: np.ndarray | None = None
    original_timesteps: list | np.ndarray | None = None
    num_ensemble_workers: int | None = None
    rho_nwp_models: np.ndarray | None = None
    domain_mask: np.ndarray | None = None
    extrapolation_kwargs: dict | None = None
    filter_kwargs: dict | None = None
    noise_kwargs: dict | None = None
    velocity_perturbation_kwargs: dict | None = None
    mask_kwargs: dict | None = None
    len_y: int | None = None
    len_x: int | None = None
    inflation_factor_bg: float | None = None
    inflation_factor_obs: float | None = None
    offset_bg: float | None = None
    offset_obs: float | None = None

    subtimesteps: list[float] | None = None
    is_nowcast_time_step: bool | None = None


class ForecastInitialization:

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

        self.__initialize_random_generators()

        self.__prepare_forecast_loop()

        self.__initialize_noise_field_pool()

        if self.__config.measure_time:
            print(
                f"Elapsed time for initialization:    {time.time() - self.__start_time_init}"
            )

        return

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

        return

    # Prepare radar precipitation fields for nowcasting and estimate the AR
    # parameters
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

        # 1. Start with the radar rainfall fields. We want the fields in a
        # Lagrangian space

        # Advect the previous precipitation fields to the same position with the
        # most recent one (i.e. transform them into the Lagrangian coordinates).

        self.__params.extrapolation_kwargs["xy_coords"] = self.__params.xy_coordinates
        self.__params.extrapolation_kwargs["outval"] = self.__config.norain_threshold
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

        obs_mask = np.logical_or(
            ~np.isfinite(self.__obs_precip),
            self.__obs_precip < self.__config.precip_threshold,
        )
        self.__obs_precip[obs_mask] = self.__config.norain_threshold

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

        precip_forecast_decomp = precip_forecast_decomp[-1]
        self.mean_extrapolation = np.array(precip_forecast_decomp["means"])
        self.std_extrapolation = np.array(precip_forecast_decomp["stds"])

        # If there are values in the radar fields, compute the auto-correlations
        GAMMA = np.empty((self.__config.n_cascade_levels, self.__config.ar_order))
        # if not self.__params.zero_precip_radar:
        # compute lag-l temporal auto-correlation coefficients for each cascade level
        for i in range(self.__config.n_cascade_levels):
            GAMMA[i, :] = correlation.temporal_autocorrelation(
                self.precip_cascades[i], mask=self.__params.mask_threshold
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

        return

    # Initialize the noise generation and get n_noise_fields
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

            self.__precip_noise_input = self.__obs_precip.copy()

            # initialize the perturbation generator for the precipitation field
            self.__params.perturbation_generator = init_noise(
                self.__precip_noise_input,
                fft_method=self.__params.fft,
                **self.__params.noise_kwargs,
            )

            if self.__config.noise_stddev_adj == "auto":
                print("Computing noise adjustment coefficients... ", end="", flush=True)
                if self.__config.measure_time:
                    starttime = time.time()

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

        return

    # Initialize the random generators
    def __initialize_random_generators(self):
        """
        Initialize random generators for precipitation noise, probability matching,
        and velocity perturbations. Each ensemble member gets a separate generator,
        ensuring reproducibility and controlled randomness in forecasts.
        """
        seed = self.__config.seed
        if self.__config.noise_method is not None:
            self.__randgen_precip = []
            # for j in range(self.__config.n_ens_members):
            for j in range(self.__config.n_noise_fields):
                rs = np.random.RandomState(seed)
                self.__randgen_precip.append(rs)
                seed = rs.randint(0, high=1e9)

        if self.__config.probmatching_method is not None:
            self.__randgen_probmatching = []
            for j in range(self.__config.n_ens_members):
                rs = np.random.RandomState(seed)
                self.__randgen_probmatching.append(rs)
                seed = rs.randint(0, high=1e9)

        if self.__config.velocity_perturbation_method is not None:
            self.__randgen_motion = []
            for j in range(self.__config.n_ens_members):
                rs = np.random.RandomState(seed)
                self.__randgen_motion.append(rs)
                seed = rs.randint(0, high=1e9)

            (
                init_velocity_noise,
                self.__params.generate_velocity_noise,
            ) = noise.get_method(self.__config.velocity_perturbation_method)

            # initialize the perturbation generators for the motion field
            self.__params.velocity_perturbations = []
            for j in range(self.__config.n_ens_members):
                kwargs = {
                    "randstate": self.__randgen_motion[j],
                    "p_par": self.__params.velocity_perturbations_parallel,
                    "p_perp": self.__params.velocity_perturbations_perpendicular,
                }
                vp_ = init_velocity_noise(
                    self.__obs_velocity,
                    1.0 / self.__config.kmperpixel,
                    self.__config.temporal_resolution,
                    **kwargs,
                )
                self.__params.velocity_perturbations.append(vp_)
        else:
            (
                self.__params.velocity_perturbations,
                self.__params.generate_velocity_noise,
            ) = (None, None)

        return

    # Prepare all necessary values and objects for the main forecast loop
    def __prepare_forecast_loop(self):
        """
        Initialize variables and structures needed for the forecast loop, including
        displacement tracking, mask parameters, noise handling, FFT objects, and
        extrapolation scaling for nowcasting.
        """

        self.__params.fft_objs = []
        for _ in range(self.__config.n_noise_fields):
            self.__params.fft_objs.append(
                utils.get_method(
                    self.__config.fft_method,
                    shape=self.precip_cascades.shape[-2:],
                )
            )

        # TODO: Implement adpative inflation factor functions
        # For the moment, set inflation factors and offsets here
        self.__params.inflation_factor_bg = 1.0
        self.__params.inflation_factor_obs = 1.0
        self.__params.offset_bg = 0.0
        self.__params.offset_obs = 0.0

    # Create n noise fields
    def __initialize_noise_field_pool(self):
        """
        Initialize a pool of noise fields avoiding the separate generation of noise fields for each time step and ensemble member. A pool of 30 fields is sufficient to generate adequate spread in the nowcast for combination.
        """
        self.noise_field_pool = np.zeros(
            (
                self.__config.n_noise_fields,
                self.__config.n_cascade_levels,
                self.__params.len_y,
                self.__params.len_x,
            )
        )

        for j in range(self.__config.n_noise_fields):
            epsilon = self.__params.noise_generator(
                self.__params.perturbation_generator,
                randstate=self.__randgen_precip[j],
                fft_method=self.__params.fft_objs[j],
                domain=self.__config.domain,
            )
            # decompose the noise field into a cascade
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

        return


class ForecastModel:

    __config: EnKFCombinationConfig | None = None
    __params: EnKFCombinationParams | None = None
    __noise_field_pool: np.ndarray | None = None
    __precip_mask: np.ndarray | None = None
    __mu: np.ndarray | None = None
    __sigma: np.ndarray | None = None

    nwc_prediction: np.ndarray | None = None
    nwc_prediction_btf: np.ndarray | None = None

    final_blended_forecast: list | None = None

    def __init__(
        self,
        enkf_combination_config: EnKFCombinationConfig,
        enkf_combination_params: EnKFCombinationParams,
        precip_cascades: np.ndarray,
        velocity: np.ndarray,
        noise_field_pool: np.ndarray,
        latest_obs: np.ndarray,
        precip_mask: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        ens_member: int,
    ):

        if ForecastModel.__config is None:

            ForecastModel.__config = enkf_combination_config
            ForecastModel.__params = enkf_combination_params
            ForecastModel.__noise_field_pool = noise_field_pool
            ForecastModel.__precip_mask = np.repeat(
                precip_mask[None, :], self.__config.n_ens_members, axis=0
            )
            ForecastModel.__mu = mu
            ForecastModel.__sigma = sigma

            latest_obs[~np.isfinite(latest_obs)] = self.__config.norain_threshold
            ForecastModel.nwc_prediction = np.repeat(
                latest_obs[None, :, :], self.__config.n_ens_members, axis=0
            )
            ForecastModel.nwc_prediction_btf = ForecastModel.nwc_prediction.copy()

            ForecastModel.final_blended_forecast = []

        self.__precip_cascades = precip_cascades
        self.__velocity = velocity

        self.__previous_displacement = np.zeros(
            (2, ForecastModel.__params.len_y, ForecastModel.__params.len_x)
        )

        self.__ens_member = ens_member

        return

    def run_forecast_step(self, nwp):

        # decompose preicpitation fields
        self.__decompose()

        # advect oldest cascade if ar_order = 2
        # self.__advect_cascade(time_step=timestep)

        # update precipitation mask
        self.__update_precip_mask(nwp=nwp)

        # compute forecast step
        self.__iterate()

        ForecastModel.nwc_prediction[self.__ens_member] = (
            blending.utils.recompose_cascade(
                combined_cascade=self.__precip_cascades[:, -1],
                combined_mean=ForecastModel.__mu,
                combined_sigma=ForecastModel.__sigma,
            )
        )

        self.__advect()

    def backtransform(self):

        ForecastModel.nwc_prediction_btf[self.__ens_member] = (
            ForecastModel.nwc_prediction[self.__ens_member]
        )

        self.__set_no_data()

        return

    def __decompose(self):

        self.__precip_cascades[:, -1] = self.__params.decomposition_method(
            ForecastModel.nwc_prediction[self.__ens_member],
            self.__params.bandpass_filter,
            fft_method=self.__params.fft_objs[self.__ens_member],
            input_domain=self.__config.domain,
            output_domain=self.__config.domain,
            compute_stats=False,
            normalize=False,
            compact_output=False,
        )["cascade_levels"]

        self.__renormalize_latest_cascade()

        return

    def __renormalize_latest_cascade(self):

        self.__precip_cascades[:, -1] = (
            self.__precip_cascades[:, -1] - ForecastModel.__mu[:, None, None]
        ) / ForecastModel.__sigma[:, None, None]
        return

    def __advect_cascade(self, time_step):

        # TODO: Implement for ar_order=2

        return

    def __advect(self):

        displacement_tmp = self.__previous_displacement.copy()

        (
            ForecastModel.nwc_prediction[self.__ens_member],
            self.__previous_displacement,
        ) = ForecastModel.__params.extrapolation_method(
            ForecastModel.nwc_prediction[self.__ens_member],
            self.__velocity,
            [1],
            allow_nonfinite_values=True,
            displacement_previous=self.__previous_displacement,
            **self.__params.extrapolation_kwargs,
        )

        self.__previous_displacement -= displacement_tmp

        return

    def __iterate(self):

        eps = (
            ForecastModel.__noise_field_pool[
                np.random.randint(ForecastModel.__config.n_noise_fields)
            ]
            * ForecastModel.__precip_mask[self.__ens_member][None, :, :]
        )

        for i in range(self.__config.n_cascade_levels):

            self.__precip_cascades[i] = autoregression.iterate_ar_model(
                self.__precip_cascades[i], self.__params.PHI[i], eps[i]
            )

        return

    def __update_precip_mask(self, nwp):

        precip_mask = (
            binary_dilation(
                nwp > ForecastModel.__config.precip_threshold,
                structure=np.ones(
                    (
                        ForecastModel.__config.precip_mask_dilation,
                        ForecastModel.__config.precip_mask_dilation,
                    ),
                    dtype=int,
                ),
            )
            * 1.0
        )
        precip_mask += (
            binary_dilation(
                ForecastModel.nwc_prediction[self.__ens_member]
                > ForecastModel.__config.precip_threshold,
                structure=np.ones(
                    (
                        ForecastModel.__config.precip_mask_dilation,
                        ForecastModel.__config.precip_mask_dilation,
                    ),
                    dtype=int,
                ),
            )
            * 1.0
        )
        precip_mask[precip_mask >= 1.0] = 1.0
        precip_mask = gaussian_filter(precip_mask, (1, 1))

        # precip_mask[ForecastModel.__params.domain_mask] = 0.0

        ForecastModel.__precip_mask[self.__ens_member] = np.array(
            precip_mask, dtype=bool
        )

        return

    def __set_no_data(self):

        ForecastModel.nwc_prediction_btf[self.__ens_member][
            ForecastModel.__params.domain_mask
        ] = np.nan

        return


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

        # Stor config
        self.__config = enkf_combination_config

        # Initialize Params
        self.__params = EnKFCombinationParams()

        # Store inpute timestamps
        self.__obs_timestamps = obs_timestamps
        self.__nwp_timestamps = nwp_timestamps

        return

    def compute_forecast(self):

        # Check timestamps of radar and nwp input and determine forecast and correction
        # timesteps as well as the temporal resolution
        self.__check_input_timestamps()

        # Check for the inputs.
        self.__check_inputs()

        # Print forecast information.
        self.__print_forecast_info()

        self.FI = ForecastInitialization(
            self.__config, self.__params, self.__obs_precip, self.__obs_velocity
        )

        # NWP: Set values below precip thr and nonfinite values to norain thr
        nwp_mask = np.logical_or(
            ~np.isfinite(self.__nwp_precip),
            self.__nwp_precip < self.__config.precip_threshold,
        )
        self.__nwp_precip[nwp_mask] = self.__config.norain_threshold

        # Set an initial precipitation mask for the NWC models
        precip_mask = binary_dilation(
            self.__obs_precip[-1] > self.__config.precip_threshold,
            structure=np.ones(
                (self.__config.precip_mask_dilation, self.__config.precip_mask_dilation)
            ),
        )

        # Initialize an instance of NWC forecast model class for each ensemble member
        self.FC_Models = {}
        for j in range(self.__config.n_ens_members):
            FC = ForecastModel(
                enkf_combination_config=self.__config,
                enkf_combination_params=self.__params,
                precip_cascades=self.FI.precip_cascades.copy(),
                velocity=self.__obs_velocity,
                noise_field_pool=self.FI.noise_field_pool,
                latest_obs=self.__obs_precip[-1, :, :],
                precip_mask=precip_mask.copy(),
                mu=self.FI.mean_extrapolation,
                sigma=self.FI.std_extrapolation,
                ens_member=j,
            )
            self.FC_Models[j] = FC

        # Initialize the combination model
        kalman_filter_model = combination.get_method("masked_enkf")
        self.KalmanFilterModel = kalman_filter_model(self.__config, self.__params)

        # Start the main forecast loop
        self.__integrated_nowcast_main_loop()

        # Stack and return the forecast output
        if self.__config.return_output:
            ForecastModel.final_blended_forecast = np.array(
                ForecastModel.final_blended_forecast
            ).swapaxes(0, 1)

            # ForecastModel.final_blended_forecast = np.stack(
            #    [
            #        np.stack(ForecastModel.final_blended_forecast[j])
            #        for j in range(self.__config.n_ens_members)
            #    ]
            # )
            if self.__config.measure_time:
                return (
                    ForecastModel.final_blended_forecast,
                    self.__fc_init,
                    self.__mainloop_time,
                )
            else:
                return ForecastModel.final_blended_forecast
        else:
            return None

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
                "NWP data has a different temporal resolution or "
                "some timesteps are missing!"
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
                "The remaining NWP forecast is not sufficient for the combined "
                "forecast period"
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

        return

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
            # self.__obs_precip = self.__obs_precip[
            #    -(self.__config.ar_order + 1) :, :, :
            # ].copy()

        # Check dimensions of obs velocity
        if self.__obs_velocity.ndim != 3:
            raise ValueError(
                "The velocity field must be a three-dimensional array"
                "of shape (2, m, n)"
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
            raise ValueError("velocity contains non-finite values")

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

        # Check for velocity perturbation kwargs
        if self.__config.velocity_perturbation_kwargs is None:
            self.__params.velocity_perturbation_kwargs = dict()
        else:
            self.__params.velocity_perturbation_kwargs = deepcopy(
                self.__config.velocity_perturbation_kwargs
            )

        # Check whether there are mask kwargs
        if self.__config.mask_kwargs is None:
            self.__params.mask_kwargs = dict()
        else:
            self.__params.mask_kwargs = deepcopy(self.__config.mask_kwargs)

        if self.__config.conditional and self.__params.precip_threshold is None:
            raise ValueError("conditional=True but precip_thr is not set")

        # Set the precipitation threshold also in params
        self.__params.precip_threshold = self.__config.precip_threshold

        # Check for the standard deviation adjustment of the noise fields
        if self.__config.noise_stddev_adj not in ["auto", "fixed", None]:
            raise ValueError(
                "unknown noise_std_dev_adj method %s: must be 'auto', 'fixed', or None"
                % self.__config.noise_stddev_adj
            )

        # Check whether the horizontal resolution is given
        if self.__config.kmperpixel is None:
            if self.__config.velocity_perturbation_method is not None:
                raise ValueError(
                    "velocity_perturbation_method is set but kmperpixel=None"
                )

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
        print(f"forecast issue time:                {self.__fc_init.isoformat()}")
        print(
            f"input dimensions:                   {self.__obs_precip.shape[1]}x{self.__obs_precip.shape[2]}"
        )
        if self.__config.kmperpixel is not None:
            print(f"km/pixel:                           {self.__config.kmperpixel}")
        print(f"temporal resolution:                {self.__temporal_res} minutes")
        print("")

        print("NWP and blending inputs")
        print("-----------------------")
        print(f"number of (NWP) models:             {self.__nwp_precip.shape[0]}")
        print("")

        print("Methods")
        print("-------")
        print(
            f"extrapolation:                      {self.__config.extrapolation_method}"
        )
        print(
            f"bandpass filter:                    {self.__config.bandpass_filter_method}"
        )
        print(
            f"decomposition:                      {self.__config.decomposition_method}"
        )
        print(f"noise generator:                    {self.__config.noise_method}")
        print(
            f"noise adjustment:                   {'yes' if self.__config.noise_stddev_adj else 'no'}"
        )
        print(
            f"velocity perturbator:               {self.__config.velocity_perturbation_method}"
        )

        print(f"EnKF implementation:                {self.__config.enkf_method}")
        print(
            f"probability matching:               {self.__config.probmatching_method}"
        )
        print(
            f"iterative probability matching:     {self.__config.iter_probability_matching}"
        )
        print(
            f"post forecast probability matching: {self.__config.post_probability_matching}"
        )
        print(f"FFT method:                         {self.__config.fft_method}")
        print(f"domain:                             {self.__config.domain}")
        print("")

        print("Parameters")
        print("----------")
        print(f"Forecast length in min:             {self.__fc_period}")
        print(f"ensemble size:                      {self.__config.n_ens_members}")
        print(f"parallel threads:                   {self.__config.num_workers}")
        print(f"number of cascade levels:           {self.__config.n_cascade_levels}")
        print(f"order of the AR(p) model:           {self.__config.ar_order}")
        print("")

        return

    def __integrated_nowcast_main_loop(self):

        if self.__config.measure_time:
            starttime_mainloop = time.time()

        self.__params.extrapolation_kwargs["return_displacement"] = True

        for t, fc_leadtime in enumerate(self.__forecast_leadtimes):

            if self.__config.measure_time:
                starttime = time.time()

            if t > 0:

                print(
                    f"Computing combination for lead time +{fc_leadtime}min... ",
                    end="",
                    flush=True,
                )
                # Set t_corr to -1 to compute the precip mask with the first NWP fields
                # Afterwards, the NWP fields closest in the future are used
                t_corr = -1

                def worker(j):

                    self.FC_Models[j].run_forecast_step(
                        nwp=self.__nwp_precip[j, t_corr + 1]
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

                if fc_leadtime in self.__correction_leadtimes:
                    t_corr = np.where(self.__correction_leadtimes == fc_leadtime)[0][0]

                    ForecastModel.nwc_prediction = self.KalmanFilterModel.correct_step(
                        ForecastModel.nwc_prediction, self.__nwp_precip[:, t_corr]
                    )

                    [FC_Model.backtransform() for FC_Model in self.FC_Models.values()]

                if self.__config.measure_time:
                    _ = self.__measure_time("timestep", starttime)
                else:
                    print("done.")

            if self.__config.return_output:

                ForecastModel.final_blended_forecast.append(
                    ForecastModel.nwc_prediction_btf.copy()
                )

        if self.__config.measure_time:
            self.__mainloop_time = time.time() - starttime_mainloop
            print(f"Elapsed time for computing forecast:{self.__mainloop_time}")

        return

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
    forecast_period,
    issuetime,
    n_ens_members,
    precip_mask_dilation=1,
    n_cascade_levels=12,
    precip_thr=-10.0,
    norain_thr=-15.0,
    kmperpixel=1.0,
    extrap_method="semilagrangian",
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    noise_method="nonparametric",
    enkf_method="masked_enkf",
    noise_stddev_adj=None,
    ar_order=1,
    vel_pert_method=None,
    conditional=False,
    probmatching_method="cdf",
    iter_probability_matching=True,
    post_probability_matching=False,
    non_precip_mask=True,
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
    mask_kwargs=None,
    measure_time=False,
    n_tapering=0,
    n_ens_prec=1,
    lien_criterion=True,
    n_lien=10,
):

    # TODO: Add descriptions and docstrings

    combination_config = EnKFCombinationConfig(
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        precip_threshold=precip_thr,
        norain_threshold=norain_thr,
        kmperpixel=kmperpixel,
        precip_mask_dilation=precip_mask_dilation,
        extrapolation_method=extrap_method,
        decomposition_method=decomp_method,
        bandpass_filter_method=bandpass_filter_method,
        noise_method=noise_method,
        enkf_method=enkf_method,
        noise_stddev_adj=noise_stddev_adj,
        ar_order=ar_order,
        velocity_perturbation_method=vel_pert_method,
        conditional=conditional,
        probmatching_method=probmatching_method,
        iter_probability_matching=iter_probability_matching,
        post_probability_matching=post_probability_matching,
        non_precip_mask=non_precip_mask,
        seed=seed,
        num_workers=num_workers,
        fft_method=fft_method,
        domain=domain,
        extrapolation_kwargs=extrap_kwargs,
        filter_kwargs=filter_kwargs,
        noise_kwargs=noise_kwargs,
        velocity_perturbation_kwargs=vel_pert_kwargs,
        mask_kwargs=mask_kwargs,
        measure_time=measure_time,
        callback=callback,
        return_output=return_output,
        n_noise_fields=30,
        n_tapering=n_tapering,
        n_ens_prec=n_ens_prec,
        lien_criterion=lien_criterion,
        n_lien=n_lien,
    )

    combination_nowcaster = EnKFCombinationNowcaster(
        obs_precip=obs_precip,
        obs_timestamps=obs_timestamps,
        nwp_precip=nwp_precip,
        nwp_timestamps=nwp_timestamps,
        obs_velocity=velocity,
        fc_period=forecast_period,
        fc_init=issuetime,
        enkf_combination_config=combination_config,
    )

    forecast_enkf_combination = combination_nowcaster.compute_forecast()
    return forecast_enkf_combination
