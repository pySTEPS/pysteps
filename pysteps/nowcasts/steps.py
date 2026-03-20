"""
pysteps.nowcasts.steps
======================

Implementation of the STEPS stochastic nowcasting method as described in
:cite:`Seed2003`, :cite:`BPS2006` and :cite:`SPN2013`.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.ndimage import generate_binary_structure, iterate_structure

from pysteps import cascade, extrapolation, noise, utils
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.nowcasts.utils import (
    compute_percentile_mask,
    nowcast_main_loop,
    zero_precipitation_forecast,
)
from pysteps.postprocessing import probmatching
from pysteps.timeseries import autoregression, correlation
from pysteps.utils.check_norain import check_norain

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


@dataclass(frozen=True)
class StepsNowcasterConfig:
    """
    Parameters
    ----------

    n_ens_members: int, optional
        The number of ensemble members to generate.
    n_cascade_levels: int, optional
        The number of cascade levels to use. Defaults to 6, see issue #385
         on GitHub.
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
    timestep: float, optional
        Time step of the motion vectors (minutes). Required if vel_pert_method is
        not None or mask_method is 'incremental'.
    extrapolation_method: str, optional
        Name of the extrapolation method to use. See the documentation of
        pysteps.extrapolation.interface.
    decomposition_method: {'fft'}, optional
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
    velocity_perturbation_method: {'bps',None}, optional
        Name of the noise generator to use for perturbing the advection field. See
        the documentation of pysteps.noise.interface. If set to None, the advection
        field is not perturbed.
    conditional: bool, optional
        If set to True, compute the statistics of the precipitation field
        conditionally by excluding pixels where the values are below the
        threshold precip_thr.
    mask_method: {'obs','sprog','incremental',None}, optional
        The method to use for masking no precipitation areas in the forecast
        field. The masked pixels are set to the minimum value of the observations.
        'obs' = apply precip_thr to the most recently observed precipitation
        intensity field, 'sprog' = use the smoothed forecast field from S-PROG,
        where the AR(p) model has been applied, 'incremental' = iteratively
        buffer the mask with a certain rate (currently it is 1 km/min),
        None=no masking.
    probmatching_method: {'cdf','mean',None}, optional
        Method for matching the statistics of the forecast field with those of
        the most recently observed one. 'cdf'=map the forecast CDF to the observed
        one, 'mean'=adjust only the conditional mean value of the forecast field
        in precipitation areas, None=no matching applied. Using 'mean' requires
        that precip_thr and mask_method are not None.
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
    extrapolation_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the extrapolation
        method. See the documentation of pysteps.extrapolation.
    filter_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the filter method.
        See the documentation of pysteps.cascade.bandpass_filters.py.
    noise_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the initializer of
        the noise generator. See the documentation of pysteps.noise.fftgenerators.
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

        The above parameters have been fitten by using run_vel_pert_analysis.py
        and fit_vel_pert_params.py located in the scripts directory.

        See pysteps.noise.motion for additional documentation.
    mask_kwargs: dict
        Optional dictionary containing mask keyword arguments 'mask_f' and
        'mask_rim', the factor defining the the mask increment and the rim size,
        respectively.
        The mask increment is defined as mask_f*timestep/kmperpixel.
    measure_time: bool
        If set to True, measure, print and return the computation time.
    callback: function, optional
        Optional function that is called after computation of each time step of
        the nowcast. The function takes one argument: a three-dimensional array
        of shape (n_ens_members,h,w), where h and w are the height and width
        of the input precipitation fields, respectively. This can be used, for
        instance, writing the outputs into files.
    return_output: bool, optional
        Set to False to disable returning the outputs as numpy arrays. This can
        save memory if the intermediate results are written to output files using
        the callback function.
    """

    n_ens_members: int = 24
    n_cascade_levels: int = 6
    precip_threshold: float | None = None
    norain_threshold: float = 0.0
    kmperpixel: float | None = None
    timestep: float | None = None
    extrapolation_method: str = "semilagrangian"
    decomposition_method: str = "fft"
    bandpass_filter_method: str = "gaussian"
    noise_method: str | None = "nonparametric"
    noise_stddev_adj: str | None = None
    ar_order: int = 2
    velocity_perturbation_method: str | None = "bps"
    conditional: bool = False
    probmatching_method: str | None = "cdf"
    mask_method: str | None = "incremental"
    seed: int | None = None
    num_workers: int = 1
    fft_method: str = "numpy"
    domain: str = "spatial"
    extrapolation_kwargs: dict[str, Any] = field(default_factory=dict)
    filter_kwargs: dict[str, Any] = field(default_factory=dict)
    noise_kwargs: dict[str, Any] = field(default_factory=dict)
    velocity_perturbation_kwargs: dict[str, Any] = field(default_factory=dict)
    mask_kwargs: dict[str, Any] = field(default_factory=dict)
    measure_time: bool = False
    callback: Callable[[Any], None] | None = None
    return_output: bool = True


@dataclass
class StepsNowcasterParams:
    fft: Any = None
    bandpass_filter: Any = None
    extrapolation_method: Any = None
    decomposition_method: Any = None
    recomposition_method: Any = None
    noise_generator: Callable | None = None
    perturbation_generator: Callable | None = None
    noise_std_coefficients: np.ndarray | None = None
    ar_model_coefficients: np.ndarray | None = None  # Corresponds to phi
    autocorrelation_coefficients: np.ndarray | None = None  # Corresponds to gamma
    domain_mask: np.ndarray | None = None
    structuring_element: np.ndarray | None = None
    precipitation_mean: float | None = None
    wet_area_ratio: float | None = None
    mask_rim: int | None = None
    num_ensemble_workers: int = 1
    xy_coordinates: np.ndarray | None = None
    velocity_perturbation_parallel: list[float] | None = None
    velocity_perturbation_perpendicular: list[float] | None = None
    filter_kwargs: dict | None = None
    noise_kwargs: dict | None = None
    velocity_perturbation_kwargs: dict | None = None
    mask_kwargs: dict | None = None


@dataclass
class StepsNowcasterState:
    precip_forecast: list[Any] | None = field(default_factory=list)
    precip_cascades: list[list[np.ndarray]] | None = field(default_factory=list)
    precip_decomposed: list[dict[str, Any]] | None = field(default_factory=list)
    # The observation mask (where the radar can observe the precipitation)
    precip_mask: list[Any] | None = field(default_factory=list)
    precip_mask_decomposed: dict[str, Any] | None = field(default_factory=dict)
    # The mask around the precipitation fields (to get only non-zero values)
    mask_precip: np.ndarray | None = None
    mask_threshold: np.ndarray | None = None
    random_generator_precip: list[np.random.RandomState] | None = field(
        default_factory=list
    )
    random_generator_motion: list[np.random.RandomState] | None = field(
        default_factory=list
    )
    velocity_perturbations: list[Callable] | None = field(default_factory=list)
    fft_objects: list[Any] | None = field(default_factory=list)
    extrapolation_kwargs: dict[str, Any] | None = field(default_factory=dict)


class StepsNowcaster:
    def __init__(
        self, precip, velocity, time_steps, steps_config: StepsNowcasterConfig
    ):
        # Store inputs and optional parameters
        self.__precip = precip
        self.__velocity = velocity
        self.__time_steps = time_steps

        # Store the config data:
        self.__config = steps_config

        # Store the state and params data:
        self.__state = StepsNowcasterState()
        self.__params = StepsNowcasterParams()

        # Additional variables for time measurement
        self.__start_time_init = None
        self.__init_time = None
        self.__mainloop_time = None

    def compute_forecast(self):
        """
        Generate a nowcast ensemble by using the Short-Term Ensemble Prediction
        System (STEPS) method.

        Parameters
        ----------
        precip: array-like
            Array of shape (ar_order+1,m,n) containing the input precipitation fields
            ordered by timestamp from oldest to newest. The time steps between the
            inputs are assumed to be regular.
        velocity: array-like
            Array of shape (2,m,n) containing the x- and y-components of the advection
            field. The velocities are assumed to represent one time step between the
            inputs. All values are required to be finite.
        timesteps: int or list of floats
            Number of time steps to forecast or a list of time steps for which the
            forecasts are computed (relative to the input time step). The elements
            of the list are required to be in ascending order.
        config: StepsNowcasterConfig
            Provides a set of configuration parameters for the nowcast ensemble generation.

        Returns
        -------
        out: ndarray
            If return_output is True, a four-dimensional array of shape
            (n_ens_members,num_timesteps,m,n) containing a time series of forecast
            precipitation fields for each ensemble member. Otherwise, a None value
            is returned. The time series starts from t0+timestep, where timestep is
            taken from the input precipitation fields. If measure_time is True, the
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
        """
        self.__check_inputs()
        self.__print_forecast_info()
        # Measure time for initialization
        if self.__config.measure_time:
            self.__start_time_init = time.time()

        # Slice the precipitation field to only use the last ar_order + 1 fields
        self.__precip = self.__precip[-(self.__config.ar_order + 1) :, :, :].copy()
        self.__initialize_nowcast_components()
        if check_norain(
            self.__precip,
            self.__config.precip_threshold,
            self.__config.norain_threshold,
            self.__params.noise_kwargs["win_fun"],
        ):
            return zero_precipitation_forecast(
                self.__config.n_ens_members,
                self.__time_steps,
                self.__precip,
                self.__config.callback,
                self.__config.return_output,
                self.__config.measure_time,
                self.__start_time_init,
            )

        self.__perform_extrapolation()
        self.__apply_noise_and_ar_model()
        self.__initialize_velocity_perturbations()
        self.__initialize_precipitation_mask()
        self.__initialize_fft_objects()
        # Measure and print initialization time
        if self.__config.measure_time:
            self.__init_time = self.__measure_time(
                "Initialization", self.__start_time_init
            )

        # Run the main nowcast loop
        self.__nowcast_main()

        # Unstack nowcast output if return_output is True
        if self.__config.measure_time:
            (
                self.__state.precip_forecast,
                self.__mainloop_time,
            ) = self.__state.precip_forecast

        # Stack and return the forecast output
        if self.__config.return_output:
            self.__state.precip_forecast = np.stack(
                [
                    np.stack(self.__state.precip_forecast[j])
                    for j in range(self.__config.n_ens_members)
                ]
            )
            if self.__config.measure_time:
                return (
                    self.__state.precip_forecast,
                    self.__init_time,
                    self.__mainloop_time,
                )
            else:
                return self.__state.precip_forecast
        else:
            return None

    def __nowcast_main(self):
        """
        Main nowcast loop that iterates through the ensemble members and time steps
        to generate forecasts.
        """
        # Isolate the last time slice of observed precipitation
        precip = self.__precip[
            -1, :, :
        ]  # Extract the last available precipitation field

        # Prepare state and params dictionaries, these need to be formatted a specific way for the nowcast_main_loop
        state = self.__return_state_dict()
        params = self.__return_params_dict(precip)

        print("Starting nowcast computation.")

        # Run the nowcast main loop
        self.__state.precip_forecast = nowcast_main_loop(
            precip,
            self.__velocity,
            state,
            self.__time_steps,
            self.__config.extrapolation_method,
            self.__update_state,  # Reference to the update function
            extrap_kwargs=self.__state.extrapolation_kwargs,
            velocity_pert_gen=self.__state.velocity_perturbations,
            params=params,
            ensemble=True,
            num_ensemble_members=self.__config.n_ens_members,
            callback=self.__config.callback,
            return_output=self.__config.return_output,
            num_workers=self.__params.num_ensemble_workers,
            measure_time=self.__config.measure_time,
        )

    def __check_inputs(self):
        """
        Validate the inputs to ensure consistency and correct shapes.
        """
        if self.__precip.ndim != 3:
            raise ValueError("precip must be a three-dimensional array")
        if self.__precip.shape[0] < self.__config.ar_order + 1:
            raise ValueError(
                f"precip.shape[0] must be at least ar_order+1, "
                f"but found {self.__precip.shape[0]}"
            )
        if self.__velocity.ndim != 3:
            raise ValueError("velocity must be a three-dimensional array")
        if self.__precip.shape[1:3] != self.__velocity.shape[1:3]:
            raise ValueError(
                f"Dimension mismatch between precip and velocity: "
                f"shape(precip)={self.__precip.shape}, shape(velocity)={self.__velocity.shape}"
            )
        if (
            isinstance(self.__time_steps, list)
            and not sorted(self.__time_steps) == self.__time_steps
        ):
            raise ValueError("timesteps must be in ascending order")
        if np.any(~np.isfinite(self.__velocity)):
            raise ValueError("velocity contains non-finite values")
        if self.__config.mask_method not in ["obs", "sprog", "incremental", None]:
            raise ValueError(
                f"Unknown mask method '{self.__config.mask_method}'. "
                "Must be 'obs', 'sprog', 'incremental', or None."
            )
        if self.__config.precip_threshold is None:
            if self.__config.conditional:
                raise ValueError("conditional=True but precip_thr is not specified.")
            if self.__config.mask_method is not None:
                raise ValueError("mask_method is set but precip_thr is not specified.")
            if self.__config.probmatching_method == "mean":
                raise ValueError(
                    "probmatching_method='mean' but precip_thr is not specified."
                )
            if (
                self.__config.noise_method is not None
                and self.__config.noise_stddev_adj == "auto"
            ):
                raise ValueError(
                    "noise_stddev_adj='auto' but precip_thr is not specified."
                )
        if self.__config.noise_stddev_adj not in ["auto", "fixed", None]:
            raise ValueError(
                f"Unknown noise_stddev_adj method '{self.__config.noise_stddev_adj}'. "
                "Must be 'auto', 'fixed', or None."
            )
        if self.__config.kmperpixel is None:
            if self.__config.velocity_perturbation_method is not None:
                raise ValueError("vel_pert_method is set but kmperpixel=None")
            if self.__config.mask_method == "incremental":
                raise ValueError("mask_method='incremental' but kmperpixel=None")
        if self.__config.timestep is None:
            if self.__config.velocity_perturbation_method is not None:
                raise ValueError("vel_pert_method is set but timestep=None")
            if self.__config.mask_method == "incremental":
                raise ValueError("mask_method='incremental' but timestep=None")

        # Handle None values for various kwargs
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

        if self.__config.mask_kwargs is None:
            self.__params.mask_kwargs = dict()
        else:
            self.__params.mask_kwargs = deepcopy(self.__config.mask_kwargs)

        print("Inputs validated and initialized successfully.")

    def __print_forecast_info(self):
        """
        Print information about the forecast setup, including inputs, methods, and parameters.
        """
        print("Computing STEPS nowcast")
        print("-----------------------")
        print("")

        print("Inputs")
        print("------")
        print(f"input dimensions: {self.__precip.shape[1]}x{self.__precip.shape[2]}")
        if self.__config.kmperpixel is not None:
            print(f"km/pixel:         {self.__config.kmperpixel}")
        if self.__config.timestep is not None:
            print(f"time step:        {self.__config.timestep} minutes")
        print("")

        print("Methods")
        print("-------")
        print(f"extrapolation:          {self.__config.extrapolation_method}")
        print(f"bandpass filter:        {self.__config.bandpass_filter_method}")
        print(f"decomposition:          {self.__config.decomposition_method}")
        print(f"noise generator:        {self.__config.noise_method}")
        print(
            "noise adjustment:       {}".format(
                ("yes" if self.__config.noise_stddev_adj else "no")
            )
        )
        print(f"velocity perturbator:   {self.__config.velocity_perturbation_method}")
        print(
            "conditional statistics: {}".format(
                ("yes" if self.__config.conditional else "no")
            )
        )
        print(f"precip. mask method:    {self.__config.mask_method}")
        print(f"probability matching:   {self.__config.probmatching_method}")
        print(f"FFT method:             {self.__config.fft_method}")
        print(f"domain:                 {self.__config.domain}")
        print("")

        print("Parameters")
        print("----------")
        if isinstance(self.__time_steps, int):
            print(f"number of time steps:     {self.__time_steps}")
        else:
            print(f"time steps:               {self.__time_steps}")
        print(f"ensemble size:            {self.__config.n_ens_members}")
        print(f"parallel threads:         {self.__config.num_workers}")
        print(f"number of cascade levels: {self.__config.n_cascade_levels}")
        print(f"order of the AR(p) model: {self.__config.ar_order}")

        if self.__config.velocity_perturbation_method == "bps":
            self.__params.velocity_perturbation_parallel = (
                self.__params.velocity_perturbation_kwargs.get(
                    "p_par", noise.motion.get_default_params_bps_par()
                )
            )
            self.__params.velocity_perturbation_perpendicular = (
                self.__params.velocity_perturbation_kwargs.get(
                    "p_perp", noise.motion.get_default_params_bps_perp()
                )
            )
            print(
                f"velocity perturbations, parallel:      {self.__params.velocity_perturbation_parallel[0]},{self.__params.velocity_perturbation_parallel[1]},{self.__params.velocity_perturbation_parallel[2]}"
            )
            print(
                f"velocity perturbations, perpendicular: {self.__params.velocity_perturbation_perpendicular[0]},{self.__params.velocity_perturbation_perpendicular[1]},{self.__params.velocity_perturbation_perpendicular[2]}"
            )

        if self.__config.precip_threshold is not None:
            print(f"precip. intensity threshold: {self.__config.precip_threshold}")

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

        # Determine the domain mask from non-finite values in the precipitation data
        self.__params.domain_mask = np.logical_or.reduce(
            [~np.isfinite(self.__precip[i, :]) for i in range(self.__precip.shape[0])]
        )

        print("Nowcast components initialized successfully.")

    def __perform_extrapolation(self):
        """
        Extrapolate (advect) precipitation fields based on the velocity field to align
        them in time. This prepares the precipitation fields for autoregressive modeling.
        """
        # Determine the precipitation threshold mask if conditional is set
        if self.__config.conditional:
            self.__state.mask_threshold = np.logical_and.reduce(
                [
                    self.__precip[i, :, :] >= self.__config.precip_threshold
                    for i in range(self.__precip.shape[0])
                ]
            )
        else:
            self.__state.mask_threshold = None

        extrap_kwargs = self.__state.extrapolation_kwargs.copy()
        extrap_kwargs["xy_coords"] = self.__params.xy_coordinates
        extrap_kwargs["allow_nonfinite_values"] = (
            True if np.any(~np.isfinite(self.__precip)) else False
        )

        res = []

        def __extrapolate_single_field(precip, i):
            # Extrapolate a single precipitation field using the velocity field
            return self.__params.extrapolation_method(
                precip[i, :, :],
                self.__velocity,
                self.__config.ar_order - i,
                "min",
                **extrap_kwargs,
            )[-1]

        for i in range(self.__config.ar_order):
            if (
                not DASK_IMPORTED
            ):  # If Dask is not available, perform sequential extrapolation
                self.__precip[i, :, :] = __extrapolate_single_field(self.__precip, i)
            else:
                # If Dask is available, accumulate delayed computations for parallel execution
                res.append(dask.delayed(__extrapolate_single_field)(self.__precip, i))

        # If Dask is available, perform the parallel computation
        if DASK_IMPORTED and res:
            num_workers_ = min(self.__params.num_ensemble_workers, len(res))
            self.__precip = np.stack(
                list(dask.compute(*res, num_workers=num_workers_))
                + [self.__precip[-1, :, :]]
            )

        print("Extrapolation complete and precipitation fields aligned.")

    def __apply_noise_and_ar_model(self):
        """
        Apply noise and autoregressive (AR) models to precipitation cascades.
        This method applies the AR model to the decomposed precipitation cascades
        and adds noise perturbations if necessary.
        """
        # Make a copy of the precipitation data and replace non-finite values
        precip = self.__precip.copy()
        for i in range(self.__precip.shape[0]):
            # Replace non-finite values with the minimum finite value of the precipitation field
            precip[i, ~np.isfinite(precip[i, :])] = np.nanmin(precip[i, :])
        # Store the precipitation data back in the object
        self.__precip = precip

        # Initialize the noise generator if the noise_method is provided
        if self.__config.noise_method is not None:
            np.random.seed(
                self.__config.seed
            )  # Set the random seed for reproducibility
            init_noise, generate_noise = noise.get_method(self.__config.noise_method)
            self.__params.noise_generator = generate_noise

            self.__params.perturbation_generator = init_noise(
                self.__precip,
                fft_method=self.__params.fft,
                **self.__params.noise_kwargs,
            )

            # Handle noise standard deviation adjustments if necessary
            if self.__config.noise_stddev_adj == "auto":
                print("Computing noise adjustment coefficients... ", end="", flush=True)
                if self.__config.measure_time:
                    starttime = time.time()

                # Compute noise adjustment coefficients
                self.__params.noise_std_coefficients = (
                    noise.utils.compute_noise_stddev_adjs(
                        self.__precip[-1, :, :],
                        self.__config.precip_threshold,
                        np.min(self.__precip),
                        self.__params.bandpass_filter,
                        self.__params.decomposition_method,
                        self.__params.perturbation_generator,
                        self.__params.noise_generator,
                        20,
                        conditional=self.__config.conditional,
                        num_workers=self.__config.num_workers,
                        seed=self.__config.seed,
                    )
                )

                # Measure and print time taken
                if self.__config.measure_time:
                    __ = self.__measure_time(
                        "Noise adjustment coefficient computation", starttime
                    )
                else:
                    print("done.")

            elif self.__config.noise_stddev_adj == "fixed":
                # Set fixed noise adjustment coefficients
                func = lambda k: 1.0 / (0.75 + 0.09 * k)
                self.__params.noise_std_coefficients = [
                    func(k) for k in range(1, self.__config.n_cascade_levels + 1)
                ]

            else:
                # Default to no adjustment
                self.__params.noise_std_coefficients = np.ones(
                    self.__config.n_cascade_levels
                )

            if self.__config.noise_stddev_adj is not None:
                # Print noise std deviation coefficients if adjustments were made
                print(
                    f"noise std. dev. coeffs:   {str(self.__params.noise_std_coefficients)}"
                )

        else:
            # No noise, so set perturbation generator and noise_std_coefficients to None
            self.__params.perturbation_generator = None
            self.__params.noise_std_coefficients = np.ones(
                self.__config.n_cascade_levels
            )  # Keep default as 1.0 to avoid breaking AR model

        # Decompose the input precipitation fields
        self.__state.precip_decomposed = []
        for i in range(self.__config.ar_order + 1):
            precip_ = self.__params.decomposition_method(
                self.__precip[i, :, :],
                self.__params.bandpass_filter,
                mask=self.__state.mask_threshold,
                fft_method=self.__params.fft,
                output_domain=self.__config.domain,
                normalize=True,
                compute_stats=True,
                compact_output=True,
            )
            self.__state.precip_decomposed.append(precip_)

        # Normalize the cascades and rearrange them into a 4D array
        self.__state.precip_cascades = nowcast_utils.stack_cascades(
            self.__state.precip_decomposed, self.__config.n_cascade_levels
        )
        self.__state.precip_decomposed = self.__state.precip_decomposed[-1]
        self.__state.precip_decomposed = [
            self.__state.precip_decomposed.copy()
            for _ in range(self.__config.n_ens_members)
        ]

        # Compute temporal autocorrelation coefficients for each cascade level
        self.__params.autocorrelation_coefficients = np.empty(
            (self.__config.n_cascade_levels, self.__config.ar_order)
        )
        for i in range(self.__config.n_cascade_levels):
            self.__params.autocorrelation_coefficients[i, :] = (
                correlation.temporal_autocorrelation(
                    self.__state.precip_cascades[i], mask=self.__state.mask_threshold
                )
            )

        nowcast_utils.print_corrcoefs(self.__params.autocorrelation_coefficients)

        # Adjust the lag-2 correlation coefficient if AR(2) model is used
        if self.__config.ar_order == 2:
            for i in range(self.__config.n_cascade_levels):
                self.__params.autocorrelation_coefficients[i, 1] = (
                    autoregression.adjust_lag2_corrcoef2(
                        self.__params.autocorrelation_coefficients[i, 0],
                        self.__params.autocorrelation_coefficients[i, 1],
                    )
                )

        # Estimate the parameters of the AR model using auto-correlation coefficients
        self.__params.ar_model_coefficients = np.empty(
            (self.__config.n_cascade_levels, self.__config.ar_order + 1)
        )
        for i in range(self.__config.n_cascade_levels):
            self.__params.ar_model_coefficients[i, :] = (
                autoregression.estimate_ar_params_yw(
                    self.__params.autocorrelation_coefficients[i, :]
                )
            )

        nowcast_utils.print_ar_params(self.__params.ar_model_coefficients)

        # Discard all except the last ar_order cascades for AR model
        self.__state.precip_cascades = [
            self.__state.precip_cascades[i][-self.__config.ar_order :]
            for i in range(self.__config.n_cascade_levels)
        ]

        # Stack the cascades into a list containing all ensemble members
        self.__state.precip_cascades = [
            [
                self.__state.precip_cascades[j].copy()
                for j in range(self.__config.n_cascade_levels)
            ]
            for _ in range(self.__config.n_ens_members)
        ]

        # Initialize random generators if noise_method is provided
        if self.__config.noise_method is not None:
            self.__state.random_generator_precip = []
            self.__state.random_generator_motion = []
            seed = self.__config.seed
            for _ in range(self.__config.n_ens_members):
                # Create random state for precipitation noise generator
                rs = np.random.RandomState(seed)
                self.__state.random_generator_precip.append(rs)
                seed = rs.randint(0, high=int(1e9))
                # Create random state for motion perturbations generator
                rs = np.random.RandomState(seed)
                self.__state.random_generator_motion.append(rs)
                seed = rs.randint(0, high=int(1e9))
        else:
            self.__state.random_generator_precip = None
            self.__state.random_generator_motion = None
        print("AR model and noise applied to precipitation cascades.")

    def __initialize_velocity_perturbations(self):
        """
        Initialize the velocity perturbators for each ensemble member if the velocity
        perturbation method is specified.
        """
        if self.__config.velocity_perturbation_method is not None:
            init_vel_noise, generate_vel_noise = noise.get_method(
                self.__config.velocity_perturbation_method
            )

            self.__state.velocity_perturbations = []
            for j in range(self.__config.n_ens_members):
                kwargs = {
                    "randstate": self.__state.random_generator_motion[j],
                    "p_par": self.__params.velocity_perturbation_kwargs.get(
                        "p_par", self.__params.velocity_perturbation_parallel
                    ),
                    "p_perp": self.__params.velocity_perturbation_kwargs.get(
                        "p_perp", self.__params.velocity_perturbation_perpendicular
                    ),
                }
                vp = init_vel_noise(
                    self.__velocity,
                    1.0 / self.__config.kmperpixel,
                    self.__config.timestep,
                    **kwargs,
                )
                self.__state.velocity_perturbations.append(
                    lambda t, vp=vp: generate_vel_noise(vp, t * self.__config.timestep)
                )
        else:
            self.__state.velocity_perturbations = None
        print("Velocity perturbations initialized successfully.")

    def __initialize_precipitation_mask(self):
        """
        Initialize the precipitation mask and handle different mask methods (sprog, incremental).
        """
        self.__state.precip_forecast = [[] for _ in range(self.__config.n_ens_members)]

        if self.__config.probmatching_method == "mean":
            self.__params.precipitation_mean = np.mean(
                self.__precip[-1, :, :][
                    self.__precip[-1, :, :] >= self.__config.precip_threshold
                ]
            )
        else:
            self.__params.precipitation_mean = None

        if self.__config.mask_method is not None:
            self.__state.mask_precip = (
                self.__precip[-1, :, :] >= self.__config.precip_threshold
            )

            if self.__config.mask_method == "sprog":
                # Compute the wet area ratio and the precipitation mask
                self.__params.wet_area_ratio = np.sum(self.__state.mask_precip) / (
                    self.__precip.shape[1] * self.__precip.shape[2]
                )
                self.__state.precip_mask = [
                    self.__state.precip_cascades[0][i].copy()
                    for i in range(self.__config.n_cascade_levels)
                ]
                self.__state.precip_mask_decomposed = self.__state.precip_decomposed[
                    0
                ].copy()

            elif self.__config.mask_method == "incremental":
                # Get mask parameters
                self.__params.mask_rim = self.__params.mask_kwargs.get("mask_rim", 10)
                mask_f = self.__params.mask_kwargs.get("mask_f", 1.0)
                # Initialize the structuring element
                self.__params.structuring_element = generate_binary_structure(2, 1)
                # Expand the structuring element based on mask factor and timestep
                n = mask_f * self.__config.timestep / self.__config.kmperpixel
                self.__params.structuring_element = iterate_structure(
                    self.__params.structuring_element, int((n - 1) / 2.0)
                )
                # Compute and apply the dilated mask for each ensemble member
                self.__state.mask_precip = nowcast_utils.compute_dilated_mask(
                    self.__state.mask_precip,
                    self.__params.structuring_element,
                    self.__params.mask_rim,
                )
                self.__state.mask_precip = [
                    self.__state.mask_precip.copy()
                    for _ in range(self.__config.n_ens_members)
                ]
        else:
            self.__state.mask_precip = None

        if self.__config.noise_method is None and self.__state.precip_mask is None:
            self.__state.precip_mask = [
                self.__state.precip_cascades[0][i].copy()
                for i in range(self.__config.n_cascade_levels)
            ]
        print("Precipitation mask initialized successfully.")

    def __initialize_fft_objects(self):
        """
        Initialize FFT objects for each ensemble member.
        """
        self.__state.fft_objs = []
        for _ in range(self.__config.n_ens_members):
            fft_obj = utils.get_method(
                self.__config.fft_method, shape=self.__precip.shape[1:]
            )
            self.__state.fft_objs.append(fft_obj)
        print("FFT objects initialized successfully.")

    def __return_state_dict(self):
        """
        Initialize the state dictionary used during the nowcast iteration.
        """
        return {
            "fft_objs": self.__state.fft_objs,
            "mask_prec": self.__state.mask_precip,
            "precip_cascades": self.__state.precip_cascades,
            "precip_decomp": self.__state.precip_decomposed,
            "precip_m": self.__state.precip_mask,
            "precip_m_d": self.__state.precip_mask_decomposed,
            "randgen_prec": self.__state.random_generator_precip,
        }

    def __return_params_dict(self, precip):
        """
        Initialize the params dictionary used during the nowcast iteration.
        """
        return {
            "decomp_method": self.__params.decomposition_method,
            "domain": self.__config.domain,
            "domain_mask": self.__params.domain_mask,
            "filter": self.__params.bandpass_filter,
            "fft": self.__params.fft,
            "generate_noise": self.__params.noise_generator,
            "mask_method": self.__config.mask_method,
            "mask_rim": self.__params.mask_rim,
            "mu_0": self.__params.precipitation_mean,
            "n_cascade_levels": self.__config.n_cascade_levels,
            "n_ens_members": self.__config.n_ens_members,
            "noise_method": self.__config.noise_method,
            "noise_std_coeffs": self.__params.noise_std_coefficients,
            "num_ensemble_workers": self.__params.num_ensemble_workers,
            "phi": self.__params.ar_model_coefficients,
            "pert_gen": self.__params.perturbation_generator,
            "probmatching_method": self.__config.probmatching_method,
            "precip": precip,
            "precip_thr": self.__config.precip_threshold,
            "recomp_method": self.__params.recomposition_method,
            "struct": self.__params.structuring_element,
            "war": self.__params.wet_area_ratio,
        }

    def __update_state(self, state, params):
        """
        Update the state during the nowcasting loop. This function handles the AR model iteration,
        noise generation, recomposition, and mask application for each ensemble member.
        """
        precip_forecast_out = [None] * params["n_ens_members"]

        # Update the deterministic AR(p) model if noise or sprog mask is used
        if params["noise_method"] is None or params["mask_method"] == "sprog":
            self.__update_deterministic_ar_model(state, params)

        # Worker function for each ensemble member
        def worker(j):
            self.__apply_ar_model_to_cascades(j, state, params)
            precip_forecast_out[j] = self.__recompose_and_apply_mask(j, state, params)

        # Use Dask for parallel execution if available
        if (
            DASK_IMPORTED
            and params["n_ens_members"] > 1
            and params["num_ensemble_workers"] > 1
        ):
            res = []
            for j in range(params["n_ens_members"]):
                res.append(dask.delayed(worker)(j))
            dask.compute(*res, num_workers=params["num_ensemble_workers"])
        else:
            for j in range(params["n_ens_members"]):
                worker(j)

        return np.stack(precip_forecast_out), state

    def __update_deterministic_ar_model(self, state, params):
        """
        Update the deterministic AR(p) model for each cascade level if noise is disabled
        or if the sprog mask is used.
        """
        for i in range(params["n_cascade_levels"]):
            state["precip_m"][i] = autoregression.iterate_ar_model(
                state["precip_m"][i], params["phi"][i, :]
            )

        state["precip_m_d"]["cascade_levels"] = [
            state["precip_m"][i][-1] for i in range(params["n_cascade_levels"])
        ]

        if params["domain"] == "spatial":
            state["precip_m_d"]["cascade_levels"] = np.stack(
                state["precip_m_d"]["cascade_levels"]
            )

        precip_m_ = params["recomp_method"](state["precip_m_d"])

        if params["domain"] == "spectral":
            precip_m_ = params["fft"].irfft2(precip_m_)

        if params["mask_method"] == "sprog":
            state["mask_prec"] = compute_percentile_mask(precip_m_, params["war"])

    def __apply_ar_model_to_cascades(self, j, state, params):
        """
        Apply the AR(p) model to the cascades for each ensemble member, including
        noise generation and normalization.
        """
        # Generate noise if enabled
        if params["noise_method"] is not None:
            eps = self.__generate_and_decompose_noise(j, state, params)
        else:
            eps = None

        # Iterate the AR(p) model for each cascade level
        for i in range(params["n_cascade_levels"]):
            if eps is not None:
                eps_ = eps["cascade_levels"][i]
                eps_ *= params["noise_std_coeffs"][i]
            else:
                eps_ = None

            # Apply the AR(p) model with or without perturbations
            if eps is not None or params["vel_pert_method"] is not None:
                state["precip_cascades"][j][i] = autoregression.iterate_ar_model(
                    state["precip_cascades"][j][i], params["phi"][i, :], eps=eps_
                )
            else:
                # use the deterministic AR(p) model computed above if
                # perturbations are disabled
                state["precip_cascades"][j][i] = state["precip_m"][i]

        eps = None
        eps_ = None

    def __generate_and_decompose_noise(self, j, state, params):
        """
        Generate and decompose the noise field into cascades for a given ensemble member.
        """
        eps = params["generate_noise"](
            params["pert_gen"],
            randstate=state["randgen_prec"][j],
            fft_method=state["fft_objs"][j],
            domain=params["domain"],
        )

        eps = params["decomp_method"](
            eps,
            params["filter"],
            fft_method=state["fft_objs"][j],
            input_domain=params["domain"],
            output_domain=params["domain"],
            compute_stats=True,
            normalize=True,
            compact_output=True,
        )

        return eps

    def __recompose_and_apply_mask(self, j, state, params):
        """
        Recompose the precipitation field from cascades and apply the precipitation mask.
        """
        state["precip_decomp"][j]["cascade_levels"] = [
            state["precip_cascades"][j][i][-1, :]
            for i in range(params["n_cascade_levels"])
        ]

        if params["domain"] == "spatial":
            state["precip_decomp"][j]["cascade_levels"] = np.stack(
                state["precip_decomp"][j]["cascade_levels"]
            )

        precip_forecast = params["recomp_method"](state["precip_decomp"][j])

        if params["domain"] == "spectral":
            precip_forecast = state["fft_objs"][j].irfft2(precip_forecast)

        # Apply the precipitation mask
        if params["mask_method"] is not None:
            precip_forecast = self.__apply_precipitation_mask(
                precip_forecast, j, state, params
            )

        # Adjust the CDF of the forecast to match the observed precipitation field
        if params["probmatching_method"] == "cdf":
            precip_forecast = probmatching.nonparam_match_empirical_cdf(
                precip_forecast, params["precip"]
            )
        # Adjust the mean of the forecast to match the observed mean
        elif params["probmatching_method"] == "mean":
            mask = precip_forecast >= params["precip_thr"]
            mu_fct = np.mean(precip_forecast[mask])
            precip_forecast[mask] = precip_forecast[mask] - mu_fct + params["mu_0"]

        # Update the mask for incremental method
        if params["mask_method"] == "incremental":
            state["mask_prec"][j] = nowcast_utils.compute_dilated_mask(
                precip_forecast >= params["precip_thr"],
                params["struct"],
                params["mask_rim"],
            )

        # Apply the domain mask (set masked areas to NaN)
        precip_forecast[params["domain_mask"]] = np.nan

        return precip_forecast

    def __apply_precipitation_mask(self, precip_forecast, j, state, params):
        """
        Apply the precipitation mask to prevent new precipitation from generating
        in areas where it was not observed.
        """
        precip_forecast_min = precip_forecast.min()

        if params["mask_method"] == "incremental":
            precip_forecast = (
                precip_forecast_min
                + (precip_forecast - precip_forecast_min) * state["mask_prec"][j]
            )
            mask_prec_ = precip_forecast > precip_forecast_min
        else:
            mask_prec_ = state["mask_prec"]

        # Set to min value outside the mask
        precip_forecast[~mask_prec_] = precip_forecast_min

        return precip_forecast

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

    def reset_states_and_params(self):
        """
        Reset the internal state and parameters of the nowcaster to allow multiple forecasts.
        This method resets the state and params to their initial conditions without reinitializing
        the inputs like precip, velocity, time_steps, or config.
        """
        # Re-initialize the state and parameters
        self.__state = StepsNowcasterState()
        self.__params = StepsNowcasterParams()

        # Reset time measurement variables
        self.__start_time_init = None
        self.__init_time = None
        self.__mainloop_time = None


# Wrapper function to preserve backward compatibility
def forecast(
    precip,
    velocity,
    timesteps,
    n_ens_members=24,
    n_cascade_levels=6,
    precip_thr=None,
    norain_thr=0.0,
    kmperpixel=None,
    timestep=None,
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
    callback=None,
    return_output=True,
):
    """
    Generate a nowcast ensemble by using the Short-Term Ensemble Prediction
    System (STEPS) method.

    Parameters
    ----------
    precip: array-like
        Array of shape (ar_order+1,m,n) containing the input precipitation fields
        ordered by timestamp from oldest to newest. The time steps between the
        inputs are assumed to be regular.
    velocity: array-like
        Array of shape (2,m,n) containing the x- and y-components of the advection
        field. The velocities are assumed to represent one time step between the
        inputs. All values are required to be finite.
    timesteps: int or list of floats
        Number of time steps to forecast or a list of time steps for which the
        forecasts are computed (relative to the input time step). The elements
        of the list are required to be in ascending order.
    n_ens_members: int, optional
        The number of ensemble members to generate.
    n_cascade_levels: int, optional
        The number of cascade levels to use. Defaults to 6, see issue #385
         on GitHub.
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
    timestep: float, optional
        Time step of the motion vectors (minutes). Required if vel_pert_method is
        not None or mask_method is 'incremental'.
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
        conditionally by excluding pixels where the values are below the
        threshold precip_thr.
    mask_method: {'obs','sprog','incremental',None}, optional
        The method to use for masking no precipitation areas in the forecast
        field. The masked pixels are set to the minimum value of the observations.
        'obs' = apply precip_thr to the most recently observed precipitation
        intensity field, 'sprog' = use the smoothed forecast field from S-PROG,
        where the AR(p) model has been applied, 'incremental' = iteratively
        buffer the mask with a certain rate (currently it is 1 km/min),
        None=no masking.
    probmatching_method: {'cdf','mean',None}, optional
        Method for matching the statistics of the forecast field with those of
        the most recently observed one. 'cdf'=map the forecast CDF to the observed
        one, 'mean'=adjust only the conditional mean value of the forecast field
        in precipitation areas, None=no matching applied. Using 'mean' requires
        that precip_thr and mask_method are not None.
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
    mask_kwargs: dict
        Optional dictionary containing mask keyword arguments 'mask_f' and
        'mask_rim', the factor defining the the mask increment and the rim size,
        respectively.
        The mask increment is defined as mask_f*timestep/kmperpixel.
    measure_time: bool
        If set to True, measure, print and return the computation time.
    callback: function, optional
        Optional function that is called after computation of each time step of
        the nowcast. The function takes one argument: a three-dimensional array
        of shape (n_ens_members,h,w), where h and w are the height and width
        of the input precipitation fields, respectively. This can be used, for
        instance, writing the outputs into files.
    return_output: bool, optional
        Set to False to disable returning the outputs as numpy arrays. This can
        save memory if the intermediate results are written to output files using
        the callback function.

    Returns
    -------
    out: ndarray
        If return_output is True, a four-dimensional array of shape
        (n_ens_members,num_timesteps,m,n) containing a time series of forecast
        precipitation fields for each ensemble member. Otherwise, a None value
        is returned. The time series starts from t0+timestep, where timestep is
        taken from the input precipitation fields. If measure_time is True, the
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
    """

    nowcaster_config = StepsNowcasterConfig(
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
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
        conditional=conditional,
        probmatching_method=probmatching_method,
        mask_method=mask_method,
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
    )

    # Create an instance of the new class with all the provided arguments
    nowcaster = StepsNowcaster(
        precip, velocity, timesteps, steps_config=nowcaster_config
    )
    forecast_steps_nowcast = nowcaster.compute_forecast()
    nowcaster.reset_states_and_params()
    # Call the appropriate methods within the class
    return forecast_steps_nowcast
