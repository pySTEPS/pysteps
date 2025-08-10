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
    timestep: float
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
    mask_rim: int | None = None
    struct: np.ndarray | None = None
    time_steps_is_list: bool = False
    xy_coordinates: np.ndarray | None = None
    precip_zerovalue: float | None = None
    precip_threshold: float | None = None
    mask_threshold: np.ndarray | None = None
    original_timesteps: list | np.ndarray | None = None
    num_ensemble_workers: int | None = None
    rho_nwp_models: np.ndarray | None = None
    domain_mask: np.ndarray | None = None
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


@dataclass
class EnKFCombinationState:
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

    noise_field_pool: np.ndarray | None = None


class ForecastModel:

    __config: EnKFCombinationConfig | None = None
    __params: EnKFCombinationParams | None = None
    __noise_field_pool: np.ndarray | None = None
    __precip_mask: np.ndarray | None = None
    __no_data_mask: np.ndarray | None = None
    __mu: np.ndarray | None = None
    __sigma: np.ndarray | None = None

    nwc_prediction: np.ndarray | None = None
    nwc_prediction_btf: np.ndarray | None = None

    def __init__(
        self,
        enkf_combination_config: EnKFCombinationConfig,
        enkf_combination_params: EnKFCombinationParams,
        velocity: np.ndarray,
        noise_field_pool: np.ndarray,
        latest_obs: np.ndarray,
        precip_mask: np.ndarray,
        no_data_mask: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        extrapolation_kwargs: dict[str, Any],
        ens_member: int,
    ):

        if ForecastModel.__config is None:

            ForecastModel.__config = enkf_combination_config
            ForecastModel.__params = enkf_combination_params
            ForecastModel.__noise_field_pool = noise_field_pool
            ForecastModel.__precip_mask = np.repeat(
                precip_mask[None, :], self.__config.n_ens_members, axis=0
            )
            ForecastModel.__no_data_mask = no_data_mask
            ForecastModel.__mu = mu
            ForecastModel.__sigma = sigma

            ForecastModel.nwc_prediction = np.repeat(
                latest_obs[None, :, :], self.__config.n_ens_members, axis=0
            )

        self.__velocity = velocity

        self.__previous_displacement = np.zeros(
            (2, ForecastModel.__params.len_y, ForecastModel.__params.len_x)
        )

        self.__extrapolation_kwargs = extrapolation_kwargs
        self.__ens_member = ens_member

        return

    def set_latest_cascade(self, precip_cascade):

        self.__precip_cascade[:, -1] = precip_cascade
        return

    def get_latest_cascade(self):

        return self.__precip_cascade[:, -1]

    def set_precip_cascade(self, precip_cascade):

        self.__precip_cascade = precip_cascade.copy()
        return

    def get_precip_cascade(self):

        return self.__precip_cascade

    def __renormalize_latest_cascade(self):

        self.__precip_cascade[:, -1] = (
            self.__precip_cascade[:, -1] - ForecastModel.__mu[:, None, None]
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
            **self.__extrapolation_kwargs,
        )

        self.__previous_displacement -= displacement_tmp

        return

    def __iterate(self):

        if ForecastModel.__noise_field_pool is not None:

            for i in range(self.__config.n_cascade_levels):

                self.__precip_cascade[i] = autoregression.iterate_ar_model(
                    self.__precip_cascade[i],
                    self.__params.PHI[i],
                    ForecastModel.__noise_field_pool[
                        np.random.randint(ForecastModel.__config.n_noise_fields)
                    ][i],
                )

        return

    def __predict(self):

        self.__iterate()

        ForecastModel.nwc_prediction[self.__ens_member] = (
            blending.utils.recompose_cascade(
                combined_cascade=self.__precip_cascade[:, -1],
                combined_mean=ForecastModel.__mu,
                combined_sigma=ForecastModel.__sigma,
            )
        )

        self.__advect()

        return

    def __update_precip_mask(self, nwp):

        precip_mask = binary_dilation(
            nwp > ForecastModel.__config.precip_threshold,
            structure=np.ones(
                (
                    ForecastModel.__config.precip_mask_dilation,
                    ForecastModel.__config.precip_mask_dilation,
                )
            ),
        )
        precip_mask += binary_dilation(
            ForecastModel.nwc_prediction[self.__ens_member]
            > ForecastModel.__config.precip_threshold,
            structure=np.ones(
                (
                    ForecastModel.__config.precip_mask_dilation,
                    ForecastModel.__config.precip_mask_dilation,
                )
            ),
        )
        precip_mask[precip_mask >= 1.0] = 1.0
        precip_mask = gaussian_filter(precip_mask, (1, 1))

        precip_mask[ForecastModel.__no_data_mask] = 0.0

        ForecastModel.__precip_mask[self.__ens_member] = np.array(
            precip_mask, dtype=bool
        )

        return

    def decompose(self):

        ForecastModel.nwc_prediction[self.__ens_member][
            ~np.isfinite(ForecastModel.nwc_prediction[self.__ens_member])
        ] = self.__config.norain_threshold

        self.__precip_cascade[:, -1] = self.__params.decomposition_method(
            ForecastModel.nwc_prediction[self.__ens_member],
            self.__params.bandpass_filter,
            fft_method=self.__params.fft_objs[self.__ens_member],
            input_domain=self.__config.domain,
            output_domain=self.__config.domain,
            compute_stats=True,
            normalize=False,
            compact_output=False,
        )["cascade_levels"]

        self.__renormalize_latest_cascade()

        return

    def set_no_data(self):

        ForecastModel.nwc_prediction[self.__ens_member][
            ForecastModel.__no_data_mask
        ] = np.nan

        return

    def run_forecast_step(self, nwp):

        # advect oldest cascade if ar_order = 2
        # self.__advect_cascade(time_step=timestep)

        # update precipitation mask
        self.__update_precip_mask(nwp=nwp)

        # compute forecast step
        self.__predict()

    def backtransform(self):

        return


class EnKFCombinationNowcaster:
    def __init__(
        self,
        obs_precip,
        nwp_precip,
        obs_velocity,
        fc_period,
        fc_init,
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

        self.__config = enkf_combination_config

        # Initialize Params and State
        self.__params = EnKFCombinationParams()
        self.__state = EnKFCombinationState()

        self.__no_data_mask = np.isnan(self.__obs_precip[-1])

        return

    def compute_forecast(self):

        # Check for the inputs.
        self.__check_inputs()

        # Print forecast information.
        self.__print_forecast_info()

        # Measure time for initialization.
        if self.__config.measure_time:
            self.__start_time_init = time.time()

        # If it is necessary, slice the precipitation field to only use the last
        # ar_order +1 time steps.
        if self.__obs_precip.shape[0] > self.__config.ar_order + 1:
            self.__obs_precip = self.__obs_precip[
                -(self.__config.ar_order + 1) :, :, :
            ].copy()

        # Initialize FFT, bandpass filters, decomposition methods, and extrapolation
        # method.
        self.__initialize_nowcast_components()

        # Prepare radar and NWP precipitation fields for nowcasting.
        self.__prepare_radar_and_NWP_fields()

        # Initialize the noise generation and get n_noise_fields
        self.__state.precip_noise_input = self.__obs_precip.copy()
        self.__initialize_noise()

        # Estimate the AR parameters
        self.__estimate_ar_parameters_radar()

        # Initialize the random generators
        self.__initialize_random_generators()

        # Prepare all necessary values and objects for the main forecast loop
        self.__prepare_forecast_loop()

        # Create n noise fields
        self.__initialize_noise_field_pool()

        # Set an initial precipitation mask for the NWC models
        precip_mask = binary_dilation(
            self.__obs_precip[-1] > self.__config.precip_threshold,
            structure=np.ones(
                (self.__config.precip_mask_dilation, self.__config.precip_mask_dilation)
            ),
        )

        # Initialize n_ens NWC models
        self.FC_Models = {}
        for j in range(self.__config.n_ens_members):
            FC = ForecastModel(
                enkf_combination_config=self.__config,
                enkf_combination_params=self.__params,
                velocity=self.__obs_velocity,
                noise_field_pool=self.__state.noise_field_pool,
                latest_obs=self.__obs_precip[-1, :, :],
                precip_mask=precip_mask,
                no_data_mask=self.__no_data_mask,
                mu=self.__state.mean_extrapolation,
                sigma=self.__state.std_extrapolation,
                extrapolation_kwargs=self.__state.extrapolation_kwargs,
                ens_member=j,
            )
            FC.set_precip_cascade(self.__state.precip_cascades)
            self.FC_Models[j] = FC

        # Initialize the combination model
        kalman_filter_model = combination.get_method("masked_enkf")
        self.KalmanFilterModel = kalman_filter_model(self.__config, self.__params)

        # Start the main forecast loop
        self.__integrated_nowcast_main_loop()

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

        # Check the temporal dimension of the NWP data
        if isinstance(self.__fc_period, list):
            self.__params.time_steps_is_list = True
            if not sorted(self.__fc_period) == self.__fc_period:
                raise ValueError(
                    "Timesteps are not in ascending order", self.__fc_period
                )
            if self.__nwp_precip.shape[1] < math.ceil(self.__fc_period[-1]) + 1:
                raise ValueError(
                    "NWP precipitation data does not contain sufficient"
                    "lead times for this forecast!"
                )
            self.__params.original_timesteps = [0] + list(self.__fc_period)
            self.__fc_period = nowcast_utils.binned_timesteps(
                self.__params.original_timesteps
            )
        else:
            self.__params.time_steps_is_list = False
            if self.__nwp_precip.shape[1] < self.__fc_period + 1:
                raise ValueError(
                    "NWP precipitation data does not contain sufficient"
                    "lead times for this forecast!"
                )
            self.__fc_period = list(range(self.__fc_period + 1))

        # Check whether there are extrapolation kwargs
        if self.__config.extrapolation_kwargs is None:
            self.__state.extrapolation_kwargs = dict()
        else:
            self.__state.extrapolation_kwargs = deepcopy(
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

        # Check whether the temporal resolution is given
        if self.__config.timestep is None:
            if self.__config.velocity_perturbation_method is not None:
                raise ValueError(
                    "velocity_perturbation_method is set but timestep=None"
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
        print(f"forecast issue time:         {self.__fc_init.isoformat()}")
        print(
            f"input dimensions:            {self.__obs_precip.shape[1]}x{self.__obs_precip.shape[2]}"
        )
        if self.__config.kmperpixel is not None:
            print(f"km/pixel:                    {self.__config.kmperpixel}")
        if self.__config.timestep is not None:
            print(f"time step:                   {self.__config.timestep} minutes")
        print("")

        print("NWP and blending inputs")
        print("-----------------------")
        print(f"number of (NWP) models:      {self.__nwp_precip.shape[0]}")
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

        print(f"EnKF implementation:         {self.__config.enkf_method}")
        print(f"probability matching:        {self.__config.probmatching_method}")
        print(
            f"iterative probability matching:   {self.__config.iter_probability_matching}"
        )
        print(
            f"post forecast probability matching:   {self.__config.post_probability_matching}"
        )
        print(f"FFT method:                  {self.__config.fft_method}")
        print(f"domain:                      {self.__config.domain}")
        print("")

        print("Parameters")
        print("----------")
        if isinstance(self.__fc_period, int):
            print(f"number of time steps:        {self.__fc_period}")
        else:
            print(f"time steps:                  {self.__fc_period}")
        print(f"ensemble size:               {self.__config.n_ens_members}")
        print(f"parallel threads:            {self.__config.num_workers}")
        print(f"number of cascade levels:    {self.__config.n_cascade_levels}")
        print(f"order of the AR(p) model:    {self.__config.ar_order}")
        print("")

        return

    def __initialize_nowcast_components(self):
        """
        Initialize the FFT, bandpass filters, decomposition methods, and extrapolation method.
        """

        # Initialize number of ensemble workers
        self.__params.num_ensemble_workers = min(
            self.__config.n_ens_members, self.__config.num_workers
        )

        # Extract the spatial dimensions of the observed precipitation (x, y)
        self.__params.len_y, self.__params.len_x = self.__obs_precip.shape[1:]

        # Initialize FFT method
        self.__params.fft = utils.get_method(
            self.__config.fft_method,
            shape=(self.__params.len_y, self.__params.len_x),
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

        # Generate the mesh grid for spatial coordinates
        x_values, y_values = np.meshgrid(
            np.arange(self.__params.len_x), np.arange(self.__params.len_y)
        )
        self.__params.xy_coordinates = np.stack([x_values, y_values])

        self.__obs_precip = self.__obs_precip[
            -(self.__config.ar_order + 1) :, :, :
        ].copy()
        # Determine the domain mask from non-finite values in the precipitation data
        self.__params.domain_mask = np.logical_or.reduce(
            [
                ~np.isfinite(self.__obs_precip[i, :])
                for i in range(self.__obs_precip.shape[0])
            ]
        )

        print("Nowcast components initialized successfully.")
        return

    def __prepare_radar_and_NWP_fields(self):
        """
        Prepare radar and NWP precipitation fields for nowcasting.
        This includes generating a threshold mask, transforming fields into
        Lagrangian coordinates, cascade decomposing/recomposing, and checking
        for zero-precip areas. The results are stored in class attributes.
        """

        # We need to know the zerovalue of precip to replace the mask when decomposing after
        # extrapolation.
        self.__params.precip_zerovalue = np.nanmin(self.__obs_precip)

        # 1. Start with the radar rainfall fields. We want the fields in a
        # Lagrangian space

        # Advect the previous precipitation fields to the same position with the
        # most recent one (i.e. transform them into the Lagrangian coordinates).

        self.__state.extrapolation_kwargs["xy_coords"] = self.__params.xy_coordinates
        res = []

        def transform_to_lagrangian(precip, i):
            return self.__params.extrapolation_method(
                precip[i, :, :],
                self.__obs_velocity,
                self.__config.ar_order - i,
                "min",
                allow_nonfinite_values=True,
                **self.__state.extrapolation_kwargs.copy(),
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

        nwp_mask = np.logical_or(
            ~np.isfinite(self.__nwp_precip),
            self.__nwp_precip < self.__config.precip_threshold,
        )
        self.__nwp_precip[nwp_mask] = self.__config.norain_threshold

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
        self.__state.precip_cascades = nowcast_utils.stack_cascades(
            precip_forecast_decomp, self.__config.n_cascade_levels
        )

        precip_forecast_decomp = precip_forecast_decomp[-1]
        self.__state.mean_extrapolation = np.array(precip_forecast_decomp["means"])
        self.__state.std_extrapolation = np.array(precip_forecast_decomp["stds"])

        return

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

        return

    def __integrated_nowcast_main_loop(self):

        if self.__config.measure_time:
            starttime_mainloop = time.time()

        self.__state.extrapolation_kwargs["return_displacement"] = True

        for t, subtimesteps_idx in enumerate(self.__fc_period):

            self.__determine_subtimesteps_and_nowcast_time_step(t, subtimesteps_idx)
            if self.__config.measure_time:
                starttime = time.time()

            if t == 0:

                if self.__config.return_output:

                    obs = self.__obs_precip[-1]
                    obs[self.__no_data_mask] = np.nan

                    for j in range(self.__config.n_ens_members):
                        self.__state.final_blended_forecast[j].append(obs)

            else:

                def worker(j):

                    self.FC_Models[j].decompose()
                    self.FC_Models[j].run_forecast_step(
                        nwp=self.__nwp_precip[j, subtimesteps_idx]
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

                ForecastModel.nwc_prediction = self.KalmanFilterModel.correct_step(
                    ForecastModel.nwc_prediction, self.__nwp_precip[:, subtimesteps_idx]
                )

                [FC_Model.set_no_data() for FC_Model in self.FC_Models.values()]

                if self.__state.is_nowcast_time_step:
                    if self.__config.measure_time:
                        _ = self.__measure_time("subtimestep", starttime)
                    else:
                        print("done.")

                final_blended_forecast_all_members_one_timestep = (
                    ForecastModel.nwc_prediction.copy()
                )

                # if self.__config.callback is not None:
                #   precip_forecast_final = np.stack(
                #       final_blended_forecast_all_members_one_timestep
                #   )
                #    if precip_forecast_final.shape[1] > 0:
                #        self.__config.callback(precip_forecast_final.squeeze())

                if self.__config.return_output:

                    for j in range(self.__config.n_ens_members):
                        self.__state.final_blended_forecast[j].append(
                            final_blended_forecast_all_members_one_timestep[j]
                        )

                final_blended_forecast_all_members_one_timestep = None
        if self.__config.measure_time:
            self.__mainloop_time = time.time() - starttime_mainloop

        return

    def __estimate_ar_parameters_radar(self):
        """
        Estimate autoregressive (AR) parameters for the radar rainfall field. If
        precipitation exists, compute temporal auto-correlations; otherwise, use
        predefined climatological values. Adjust coefficients if necessary and
        estimate AR model parameters.
        """

        # If there are values in the radar fields, compute the auto-correlations
        GAMMA = np.empty((self.__config.n_cascade_levels, self.__config.ar_order))
        # if not self.__params.zero_precip_radar:
        # compute lag-l temporal auto-correlation coefficients for each cascade level
        for i in range(self.__config.n_cascade_levels):
            GAMMA[i, :] = correlation.temporal_autocorrelation(
                self.__state.precip_cascades[i], mask=self.__params.mask_threshold
            )

        # Else, use standard values for the auto-correlations
        # else:
        #     # Get the climatological lag-1 and lag-2 auto-correlation values from Table 2
        #     # in `BPS2004`.
        #     # Hard coded, change to own (climatological) values when present.
        #     # TODO: add user warning here so users can be aware of this without reading the code?
        #     GAMMA = np.array(
        #         [
        #             [0.99805, 0.9925, 0.9776, 0.9297, 0.796, 0.482, 0.079, 0.0006],
        #             [0.9933, 0.9752, 0.923, 0.750, 0.367, 0.069, 0.0018, 0.0014],
        #         ]
        #     )

        #     # Check whether the number of cascade_levels is correct
        #     if GAMMA.shape[1] > self.__config.n_cascade_levels:
        #         GAMMA = GAMMA[:, 0 : self.__config.n_cascade_levels]
        #     elif GAMMA.shape[1] < self.__config.n_cascade_levels:
        #         # Get the number of cascade levels that is missing
        #         n_extra_lev = self.__config.n_cascade_levels - GAMMA.shape[1]
        #         # Append the array with correlation values of 10e-4
        #         GAMMA = np.append(
        #             GAMMA,
        #             [np.repeat(0.0006, n_extra_lev), np.repeat(0.0014, n_extra_lev)],
        #             axis=1,
        #         )

        #     # Finally base GAMMA.shape[0] on the AR-level
        #     if self.__config.ar_order == 1:
        #         GAMMA = GAMMA[0, :]
        #     if self.__config.ar_order > 2:
        #         for _ in range(self.__config.ar_order - 2):
        #             GAMMA = np.vstack((GAMMA, GAMMA[1, :]))

        #     # Finally, transpose GAMMA to ensure that the shape is the same as np.empty((n_cascade_levels, ar_order))
        #     GAMMA = GAMMA.transpose()
        #     assert GAMMA.shape == (
        #         self.__config.n_cascade_levels,
        #         self.__config.ar_order,
        #     )

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

    def __initialize_random_generators(self):
        """
        Initialize random generators for precipitation noise, probability matching,
        and velocity perturbations. Each ensemble member gets a separate generator,
        ensuring reproducibility and controlled randomness in forecasts.
        """
        seed = self.__config.seed
        if self.__config.noise_method is not None:
            self.__state.randgen_precip = []
            # for j in range(self.__config.n_ens_members):
            for j in range(self.__config.n_noise_fields):
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

        return

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

        self.__params.mask_rim, self.__params.struct = None, None

        if self.__config.noise_method is None:
            self.__state.final_blended_forecast_non_perturbed = [
                self.__state.precip_cascades[0][i].copy()
                for i in range(self.__config.n_cascade_levels)
            ]
        else:
            self.__state.final_blended_forecast_non_perturbed = None

        self.__params.fft_objs = []
        # for i in range(self.__config.n_ens_members):
        for i in range(self.__config.n_noise_fields):
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

        # TODO: Implement adpative inflation factor functions
        # For the moment, set inflation factors and offsets here
        self.__params.inflation_factor_bg = 1.8
        self.__params.inflation_factor_obs = 1.0
        self.__params.offset_bg = 0.0
        self.__params.offset_obs = 0.0

    def __initialize_noise_field_pool(self):
        """
        Initialize a pool of noise fields avoiding the separate generation of noise fields for each time step and ensemble member. A pool of 30 fields is sufficient to generate adequate spread in the nowcast for the combination.
        """
        self.__state.noise_field_pool = np.zeros(
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
                randstate=self.__state.randgen_precip[j],
                fft_method=self.__params.fft_objs[j],
                domain=self.__config.domain,
            )
            # decompose the noise field into a cascade
            self.__state.noise_field_pool[j] = self.__params.decomposition_method(
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
    nwp_precip,
    velocity,
    timesteps,
    timestep,
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

    # TODO: Continue cleaning up not needed parts of the code and add descriptions and
    # docstrings

    combination_config = EnKFCombinationConfig(
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        precip_threshold=precip_thr,
        norain_threshold=norain_thr,
        kmperpixel=kmperpixel,
        timestep=timestep,
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
        nwp_precip=nwp_precip,
        obs_velocity=velocity,
        fc_period=timesteps,
        fc_init=issuetime,
        enkf_combination_config=combination_config,
    )

    forecast_enkf_combination = combination_nowcaster.compute_forecast()
    return forecast_enkf_combination
