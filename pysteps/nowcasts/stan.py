"""
pysteps.nowcasts.stan
======================

Implementation of the STAN stochastic nowcasting method, it is based in
- ANVIL described in :cite:`PCLH2020`,
- SPROG-LOC described in :cite:`RRR2022`,
- STEPS described in :cite:`Seed2003`, :cite:`BPS2006` and :cite:`SPN2013`.

.. autosummary::
    :toctree: ../generated/
    
    forecast
"""

import numpy as np
from scipy.ndimage import generate_binary_structure, iterate_structure, gaussian_filter, uniform_filter
import time

from pysteps import cascade
from pysteps import extrapolation
from pysteps import noise
from pysteps import utils
from pysteps.decorators import deprecate_args
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.postprocessing import probmatching
from pysteps.timeseries import autoregression
from pysteps.nowcasts.utils import compute_percentile_mask, nowcast_main_loop
from pysteps.utils import spectral

from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False

@dataclass
class StepsNowcasterConfig:
    """
    Parameters
    ----------
    
    n_ens_members: int, optional
        The number of ensemble members to generate.
    n_cascade_levels: int, optional
        The number of cascade levels to use. Defaults to 6, see issue #385
        on GitHub.
    var1_threshold: float, optional
        Specifies the threshold value for minimum observable variable 1.
        Required if mask_method is not None or conditional is True.
    var2_threshold: float, optional
        Specifies the threshold value for minimum observable variable 2.
        Required if var2 is not None.
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
    decomp_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the decomposition
        method.
    extrapolation_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the extrapolation
        method. See the documentation of pysteps.extrapolation.
    motion_field_general: array_like or str or None, optional
        Optional externally provided motion field. If None, the input velocity
        field is used.
    velocity_perturbation_method: {'bps',None}, optional
        Name of the noise generator to use for perturbing the advection field. See
        the documentation of pysteps.noise.interface. If set to None, the advection
        field is not perturbed.
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
    noise_method: {'parametric','nonparametric','ssft','nested',None}, optional
        Name of the noise generator to use for perturbating the variable 1
        field. See the documentation of pysteps.noise.interface. If set to None,
        no noise is generated.
    noise_stddev_adj: {'auto','fixed',None}, optional
        Optional adjustment for the standard deviations of the noise fields added
        to each cascade level. This is done to compensate incorrect std. dev.
        estimates of cascade levels due to presence of no-rain areas. 'auto'=use
        the method implemented in pysteps.noise.utils.compute_noise_stddev_adjs.
        'fixed'= use the formula given in :cite:`BPS2006` (eq. 6), None=disable
        noise std. dev adjustment.
    noise_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the initializer of
        the noise generator. See the documentation of pysteps.noise.fftgenerators.
    noise_method_var2: {'parametric','nonparametric','ssft','nested',None}, optional
        Same as `noise_method`, but applied to variable 2.
    noise_stddev_adj_var2: {'auto','fixed',None}, optional
        Same as `noise_stddev_adj`, but applied to variable 2.
    noise_kwargs_var2: dict, optional
        Same as `noise_kwargs`, but applied to variable 2.
    noise_levels: int or None, optional
        Optional cascade level index above which noise is applied for variable 1.
    noise_levels_var2: int or None, optional
        Optional cascade level index above which noise is applied for variable 2.
    ar_order: int, optional
        The order of the autoregressive model to use. Must be >= 1.
    d_order: int, optional
        The order of the differencing in the autoregressive AR(p,0) (d_order = 0) or
        autoregressive integrated ARI(p,1) (d_order = 1) model to use.
    ar_window_radius: int or list or str or None, optional
        The radius of the window to use for determining the parameters of the
        autoregressive model. Set to None to disable localization. String values
        enable adaptive localization modes (e.g., 'central', 'sprogloc').
    var1_var2_window_radius: int, optional
        The radius of the window to use for determining the var2(var1) relation.
        Applicable if var2 is not None.
    autocorrelation_coefficients_factor: float, optional
        Adjusting factor for the autocorrelation coefficients.
    conditional: bool, optional
        If set to True, compute the statistics of the variable 1 field
        conditionally by excluding pixels where the values are below the
        threshold var1_thr.
    conditional_var2: bool, optional
        Same as `conditional`, but applied to variable 2.
    noise_with_var2: bool, optional
        If True, noise for variable 1 is generated using variable 2 statistics.
    phi_with_var2: bool, optional
        If True, AR(1) phi0 for variable 1 is replaced by phi0 estimated from variable 2.
    r_vil_conversion_method: {'glar',None}, optional
        Method for converting VIL to rain rate. 'glar' applies the Gaussian
        Linear Autoregressive Regression model.
    compute_glar_params: bool, optional
        If True, compute global GLAR parameters from the last available fields.
    a_glar: float, optional
        GLAR regression coefficient.
    b_glar: float, optional
        GLAR regression coefficient.
    phi_glar: float, optional
        AR(1) persistence coefficient for GLAR residuals.
    alpha: float, optional
        Logistic regression parameter for probabilistic rain occurrence.
    beta: float, optional
        Logistic regression parameter for probabilistic rain occurrence.
    prob_conversion: bool, optional
        If True, apply the probabilistic occurrence model for variable 2.
    var1_name: str, optional
        Name of variable 1.
    var2_name: str, optional
        Name of variable 2.
    mask_method: {'obs','sprog','incremental',None}, optional
        The method to use for masking no variable 1 areas in the forecast
        field. The masked pixels are set to the minimum value of the observations.
        'obs' = apply var1_thr to the most recently observed variable 1
        field, 'sprog' = use the smoothed forecast field from S-PROG,
        where the ARI(p,d) model has been applied, 'incremental' = iteratively
        buffer the mask with a certain rate (currently it is 1 km/min),
        None=no masking.
    mask_method_var2: {'obs','sprog','incremental',None}, optional
        Same as `mask_method`, but applied to variable 2.
    mask_kwargs: dict
        Optional dictionary containing mask keyword arguments 'mask_f' and
        'mask_rim', the factor defining the the mask increment and the rim size,
        respectively.
        The mask increment is defined as mask_f*timestep/kmperpixel.
    probmatching_method: {'cdf','mean',None}, optional
        Method for matching the statistics of the forecast field with those of
        the most recently observed one. 'cdf'=map the forecast CDF to the observed
        one, 'mean'=adjust only the conditional mean value of the forecast field
        in variable 1 areas, None=no matching applied. Using 'mean' requires
        that var1_thr and mask_method are not None.
    probmatching_method_var2: {'cdf','mean',None}, optional
        Same as `probmatching_method`, but applied to variable 2.
    fft_method: str, optional
        A string defining the FFT method to use (see utils.fft.get_method).
        Defaults to 'numpy' for compatibility reasons. If pyFFTW is installed,
        the recommended method is 'pyfftw'.
    domain: {"spatial", "spectral"}
        If "spatial", all computations are done in the spatial domain (the
        classical STEPS model). If "spectral", the ARI(2,d) models and stochastic
        perturbations are applied directly in the spectral domain to reduce
        memory footprint and improve performance :cite:`PCH2019b`.
    filter_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the filter method.
        See the documentation of pysteps.cascade.bandpass_filters.py.
    gamma_filter_method: {'gaussian','uniform'}, optional
        Filter used for smoothing autocorrelation coefficients.
    regression_filter_method: {'gaussian','uniform'}, optional
        Filter used for smoothing localized regression coefficients.
    fill_global_regression: bool, optional
        If True, fill missing localized regression coefficients with global ones.
    adaptive_localization: bool or int, optional
        Enables adaptive selection of cascade levels for AR localization.
    constant_adaptive_localization: bool, optional
        If True, adaptive localization uses a fixed cascade ordering.
    fill_autocorrelation_coefficients_method: {'first','global',int}, optional
        Method for filling missing autocorrelation coefficients.
    global_transition: bool, optional
        If True, gradually transition from localized to global regression.
    num_workers: int, optional
        The number of workers to use for parallel computation. Applicable if dask
        is enabled or pyFFTW is used for computing the FFT. When num_workers>1, it
        is advisable to disable OpenMP by setting the environment variable
        OMP_NUM_THREADS to 1. This avoids slowdown caused by too many simultaneous
        threads.
    measure_time: bool
        If set to True, measure, print and return the computation time.
    callback: function, optional
        Optional function that is called after computation of each time step of
        the nowcast. The function takes one argument: a three-dimensional array
        of shape (n_ens_members,h,w), where h and w are the height and width
        of the input variable 1 fields, respectively. This can be used, for
        instance, writing the outputs into files.
    return_output: bool, optional
        Set to False to disable returning the outputs as numpy arrays. This can
        save memory if the intermediate results are written to output files using
        the callback function.
    """

    # --- Ensemble & Cascade Settings ---
    n_ens_members: int = 24
    n_cascade_levels: int = 6
    decomposition_method: str = "fft"
    bandpass_filter_method: str = "gaussian"
    decomp_kwargs: dict[str, Any] = field(default_factory=dict)
    
    # --- Extrapolation & Motion ---
    extrapolation_method: str = "semilagrangian"
    extrapolation_kwargs: dict[str, Any] = field(default_factory=dict)
    motion_field_general: np.ndarray | str | None = None
    velocity_perturbation_method: str | None = "bps"
    velocity_perturbation_kwargs: dict[str, Any] = field(default_factory=dict)
    
    # --- AR Model & Noise ---
    ar_order: int = 2
    d_order: int = 0
    ar_window_radius: int | list | None = None
    noise_method: str | None = "nonparametric"
    noise_stddev_adj: str | None = None
    noise_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    noise_levels: int | None = None
    noise_levels_var2: int | None = None
    
    # --- Variable 2 (Dual-Pol/Satellite) Settings ---
    var2_name: str = "var2"
    var2_threshold: float | None = None
    noise_method_var2: str | None = None
    noise_stddev_adj_var2: str | None = None
    noise_kwargs_var2: dict[str, Any] = field(default_factory=dict)
    var1_var2_window_radius: int = 3
    conditional: bool = False
    conditional_var2: bool = False
    noise_with_var2: bool = False
    phi_with_var2: bool = False
    
    # --- VIL to Rain Conversion (Gaussian Parameters) ---
    r_vil_conversion_method: str | None = "glar"
    compute_glar_params: bool = False
    a_glar: float = 1.01104088
    b_glar: float = 7.06145853
    phi_glar: float = 0.96562279
    alpha: float = 4.73652705
    beta: float = 0.31451517
    prob_conversion: bool = False
    
    # --- Masking & Probability Matching ---
    var1_name: str = "var1"
    var1_threshold: float | None = None
    mask_method: str | None = "incremental"
    mask_method_var2: str | None = None
    mask_kwargs: dict[str, Any] = field(default_factory=dict)
    probmatching_method: str | None = "cdf"
    probmatching_method_var2: str | None = "cdf"
    
    # --- Spatial/Temporal Metadata ---
    kmperpixel: float | None = None
    timestep: float | None = None
    domain: str = "spatial"
    
    # --- Advanced/Misc Flags ---
    gamma_filter_method: str = "gaussian"
    regression_filter_method: str = "gaussian"
    autocorrelation_coefficients_factor: float = 1
    fill_global_regression: bool = False
    adaptive_localization: bool | int = False
    constant_adaptive_localization: bool = True
    fill_autocorrelation_coefficients_method: str | int = "first"
    global_transition: bool = False
    fft_method: str = "numpy"
    filter_kwargs: dict[str, Any] = field(default_factory=dict)
    
    # --- Execution Flags ---
    num_workers: int = 1
    measure_time: bool = False
    callback: Callable[[Any], None] | None = None
    return_output: bool = True


@dataclass
class StepsNowcasterParams:
    # --- Decomposition Objects ---
    fft: Any = None
    bandpass_filter: Any = None
    decomposition_method: Any = None
    recomposition_method: Any = None
    
    # --- Calculated Coefficients (Var 1) ---
    ar_model_coefficients: np.ndarray | None = None  # phi
    phi0: np.ndarray | None = None
    autocorrelation_coefficients: np.ndarray | None = None  # gamma
    noise_std_coefficients: np.ndarray | None = None
    
    # --- Calculated Coefficients (Var 2) ---
    ar_model_coefficients_var2: np.ndarray | None = None 
    phi0_var2: np.ndarray | None = None
    autocorrelation_coefficients_var2: np.ndarray | None = None
    noise_std_coefficients_var2: np.ndarray | None = None
    
    # --- Regression / Interaction Params ---
    var1_var2_a: np.ndarray | None = None
    var1_var2_b: np.ndarray | None = None
    global_var1_var2_a: np.ndarray | None = None
    global_var1_var2_b: np.ndarray | None = None
    
    # --- Generators ---
    noise_generator: Callable | None = None
    perturbation_generator: Callable | None = None
    noise_generator_var2: Callable | None = None
    perturbation_generator_var2: Callable | None = None
    is_probabilistic: bool = False
    is_var1_probabilistic: bool = False
    is_var2_probabilistic: bool = False
    is_vel_probabilistic: bool = False
    
    # --- Masking & Geometry ---
    domain_mask: np.ndarray | None = None
    xy_coordinates: np.ndarray | None = None
    structuring_element: np.ndarray | None = None
    structuring_element_var2: np.ndarray | None = None
    mask_rim: int | None = None
    mask_rim_var2: int | None = None
    
    # --- Statistics ---
    variable1_mean: float | None = None
    variable2_mean: float | None = None
    wet_area_ratio: float | None = None
    wet_area_ratio_var2: float | None = None
    var1_min: float | None = None
    var2_min: float | None = None
    
    # --- Motion Perturbation ---
    velocity_perturbation_method: Any = None
    velocity_perturbation_parallel: list[float] | None = None
    velocity_perturbation_perpendicular: list[float] | None = None
    num_ensemble_workers: int = 1


@dataclass
class StepsNowcasterState:
    # --- Variable 1 State ---
    var1_forecast: list[np.ndarray] | None = field(default_factory=list)
    var1_cascades: list[list[np.ndarray]] | None = field(default_factory=list)
    var1_cascades_adaptive: list[np.ndarray] | None = field(default_factory=list)
    var1_decomposed: list[dict[str, Any]] | None = field(default_factory=list)
    
    # Masks for Var 1
    var1_mask: list[Any] | None = None 
    var1_mask_decomposed: dict[str, Any] | None = field(default_factory=dict)
    mask_var1: np.ndarray | None = None
    mask_threshold: np.ndarray | None = None
    
    # --- Variable 2 State ---
    var2_forecast: list[np.ndarray] | None = field(default_factory=list)
    var2_cascades: list[list[np.ndarray]] | None = field(default_factory=list)
    var2_decomposed: list[dict[str, Any]] | None = field(default_factory=list)
    
    # Masks for Var 2
    var1_cascades_mask: list[list[np.ndarray]] | None = field(default_factory=list)
    var2_cascades_mask: list[list[np.ndarray]] | None = field(default_factory=list)
    var2_mask: list[Any] | None = None 
    var2_mask_decomposed: dict[str, Any] | None = field(default_factory=dict)
    mask_var2: np.ndarray | None = None
    
    # --- Generators & Objects ---
    random_generator_var1: list[np.random.RandomState] | None = field(default_factory=list)
    random_generator_var2: list[np.random.RandomState] | None = field(default_factory=list)
    random_generator_motion: list[np.random.RandomState] | None = field(default_factory=list)
    velocity_perturbations: list[Callable] | None = field(default_factory=list)
    fft_objects: list[Any] | None = field(default_factory=list)

class StepsNowcaster:
    def __init__(
        self, 
        var1: np.ndarray, 
        velocity: np.ndarray, 
        time_steps: int | list, 
        steps_config: StepsNowcasterConfig,
        var2: np.ndarray | None = None
    ):
        # Store inputs and optional parameters
        self.__var1 = var1
        self.__velocity = velocity
        self.__time_steps = time_steps
        self.__var2 = var2
        
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
        Generate a nowcast ensemble using the STAN stochastic nowcasting method.
        STAN integrates components from ANVIL, SPROG‑LOC,and STEPS to produce
        spatially and temporally consistent ensemble nowcasts of VIL or rain rate.
    
        Parameters
        ----------
        var1: array_like
            Array of shape (ar_order+d_order+1, m, n) containing the input fields
            ordered from oldest to newest. Typically VIL or rain rate, but any
            variable can be used. The time steps between inputs are assumed to be
            regular.
        velocity: array_like
            Array of shape (2, m, n) or (t, 2, m, n) containing the x- and y-components
            of the advection field. If a time‑varying motion field is provided, the first
            dimension must match the number of forecast timesteps. Velocities are assumed
            to represent one input timestep. All values must be finite.
        timesteps: int or list of floats
            Number of lead times to forecast, or a list of lead times (expressed in
            multiples of the input timestep). If a list is provided, its elements
            must be in ascending order.
        config: StepsNowcasterConfig
            Configuration object containing all parameters required for STAN,
            including AR(p,d) settings, cascade decomposition, noise generation,
            VIL–rain conversion, and masking options.
        var2: array_like, optional
            Optional second variable (e.g., rain rate when var1 is VIL). If
            provided, STAN performs localized VIL→rain regression and dual‑variable
            AR modeling.
    
        Returns
        -------
        out: ndarray or None
            If ``return_output`` is True, returns an array of shape
            (n_ens_members, num_timesteps, m, n) containing the ensemble forecasts
            of variable 1. If ``measure_time`` is True, returns a tuple containing
            (forecast, initialization_time, mainloop_time). If ``return_output`` is
            False, returns None.
    
        See also
        --------
        pysteps.extrapolation.interface
        pysteps.cascade.interface
        pysteps.noise.interface
        pysteps.noise.utils.compute_noise_stddev_adjs
    
        References
        ----------
        PCLH2020, RRR2022, Seed2003, BPS2006, SPN2013
        """

        self.__check_inputs()
        self.__print_forecast_info()
        # Measure time for initialization
        if self.__config.measure_time:
            self.__start_time_init = time.time()
        
        # Required mainly if VIL and Rain are estimated from different
        # preprocessing steps to have the same extent
        if self.__var2 is not None and self.__var2.ndim == 3:
            if self.__config.compute_glar_params:
                self.__config.a_glar, self.__config.b_glar = self.__get_global_var1_var2_regression()
            self.__fix_var1_from_var2_extent(self.__var1, self.__var2)
            
        self.__initialize_nowcast_components()
        self.__perform_extrapolation()
        if self.__var2 is not None and self.__var2.ndim == 3:
            self.__perform_extrapolation(is_var2=True)
            if self.__config.compute_glar_params:
                self.__config.phi_glar = self.__get_phi_var1_var2_regression()
            
        self.__initialize_noise_and_ar_model()
        self.__initialize_velocity_perturbations()
        self.__initialize_variable_mask()
        self.__initialize_fft_objects()
        # Measure and print initialization time
        if self.__config.measure_time:
            self.__measure_time("Initialization", self.__start_time_init)

        # Run the main nowcast loop
        self.__nowcast_main()

        if self.__config.measure_time:
            self.__state.var1_forecast, self.__mainloop_time = (
                self.__state.var1_forecast
            )

        # Stack and return the forecast output
        if self.__config.return_output:
            self.__state.var1_forecast = np.stack(
                [
                    np.stack(self.__state.var1_forecast[j])
                    for j in range(self.__config.n_ens_members)
                ]
            )
            if self.__config.measure_time:
                return (
                    self.__state.var1_forecast,
                    self.__init_time,
                    self.__mainloop_time,
                )
            else:
                return self.__state.var1_forecast
        else:
            return None

    def __nowcast_main(self):
        """
        Main nowcast loop that iterates through the ensemble members and time steps
        to generate forecasts.
        """
        # Isolate the last time slice of variable 1
        var1 = self.__var1[
            -1, :, :
        ]  # Extract the last available variable 1 field
        if self.__var2 is not None:
            var2 = self.__var2[
                -1, :, :
            ]  # Extract the last available variable 2 field
        else: var2 = None
        
        # Prepare state and params dictionaries, these need to be formatted a specific way for the nowcast_main_loop
        state = self.__initialize_state()
        params = self.__initialize_params(var1, var2=var2)

        print("Starting nowcast computation.")
        
        # Run the nowcast main loop
        self.__state.var1_forecast = nowcast_main_loop(
            var1,
            self.__velocity,
            state,
            self.__time_steps,
            self.__config.extrapolation_method,
            self.__update_state,  # Reference to the update function
            extrap_kwargs=self.__config.extrapolation_kwargs,
            motion_field_general=self.__config.motion_field_general,
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
        if self.__var1.ndim != 3:
            raise ValueError(f"{self.__config.var1_name} must be a three-dimensional array")
        if self.__var1.shape[0] < self.__config.ar_order + self.__config.d_order + 1:
            raise ValueError(
                f"var1.shape[0] must be at least ar_order+d_order+1, "
                f"but found {self.__var1.shape[0]}"
            )
        if self.__var2 is not None and self.__var2.ndim not in [2, 3]:
            raise ValueError(f"{self.__config.var2_name} must be a two- or three-dimensional array")
        if self.__velocity.ndim != 3:
            raise ValueError("velocity must be a three-dimensional array")
        if self.__var1.shape[1:3] != self.__velocity.shape[1:3]:
            raise ValueError(
                f"Dimension mismatch between {self.__config.var1_name} and velocity: "
                f"shape({self.__config.var1_name})={self.__var1.shape}, shape(velocity)={self.__velocity.shape}"
            )
        if (
            isinstance(self.__time_steps, list)
            and not sorted(self.__time_steps) == self.__time_steps
        ):
            raise ValueError("timesteps must be in ascending order")
        if np.any(~np.isfinite(self.__velocity)):
            raise ValueError("velocity contains non-finite values")
        if self.__config.mask_method not in ["obs", "sprog", "incremental", "stepsincremental", "obsincremental", None]:
            raise ValueError(
                f"Unknown mask method '{self.__config.mask_method}'. "
                "Must be 'obs', 'sprog', 'incremental', or None."
            )
        if self.__config.var1_threshold is None:
            if self.__config.conditional:
                raise ValueError("conditional=True but var1_thr is not specified.")
            if self.__config.mask_method is not None:
                raise ValueError("mask_method is set but var1_thr is not specified.")
            if self.__config.probmatching_method == "mean":
                raise ValueError(
                    "probmatching_method='mean' but var1_thr is not specified."
                )
            if self.__config.probmatching_method_var2 == "mean":
                raise ValueError(
                    "probmatching_method_var2='mean' but var2_thr is not specified."
                )
            if (
                self.__config.noise_method is not None
                and self.__config.noise_stddev_adj == "auto"
            ):
                raise ValueError(
                    "noise_stddev_adj='auto' but var1_thr is not specified."
                )
        if self.__config.noise_stddev_adj not in ["auto", "fixed", None]:
            raise ValueError(
                f"Unknown noise_stddev_adj method '{self.__config.noise_stddev_adj}'. "
                "Must be 'auto', 'fixed', or None."
            )
        if self.__config.kmperpixel is None:
            if self.__config.velocity_perturbation_method is not None:
                raise ValueError("vel_pert_method is set but kmperpixel=None")
            if "incremental" in self.__config.mask_method:
                raise ValueError("mask_method='incremental' but kmperpixel=None")
            if self.__config.adaptive_localization:
                raise ValueError("adaptive_localization is set but kmperpixel=None")
        if self.__config.timestep is None:
            if self.__config.velocity_perturbation_method is not None:
                raise ValueError("vel_pert_method is set but timestep=None")
            if "incremental" in self.__config.mask_method:
                raise ValueError("mask_method='incremental' but timestep=None")
        
        if self.__config.ar_window_radius is not None:
            if not isinstance(self.__config.ar_window_radius, (str, int, list)):
                raise ValueError("ar_window_radius type must be None, str, int or list")
            if self.__config.adaptive_localization:
                if isinstance(self.__config.ar_window_radius, str):
                    if self.__config.ar_window_radius.split('_')[0] not in ["central", "sprogloc"]: # "central_r", "central_sqr", "central_L", "central_L_r", "sprogloc_r", "sprogloc_L", "sprogloc_L_r"]:
                        raise ValueError("ar_window_radius must be 'central', or 'sprogloc' with possible additions to scale '_L', reverse '_r' or set minimum size '_m##'.")
                elif isinstance(self.__config.ar_window_radius, list):
                    if len(self.__config.ar_window_radius) != self.__config.n_cascade_levels:
                        raise ValueError(
                            "adaptive_localization = True but length msimatch between ar_window_radius and n_cascade_levels"
                        )
                if isinstance(self.__config.adaptive_localization, int) & self.__config.adaptive_localization > self.__config.n_cascade_levels:
                    raise ValueError("adaptive_localization is int but higher than n_cascade_levels")
        
        # Handle None values for various kwargs
        if self.__config.extrapolation_kwargs is None:
            self.__config.extrapolation_kwargs = {}
        if self.__config.decomp_kwargs is None:
            self.__config.decomp_kwargs = {
                'normalize':True, 
                'compute_stats':True, 
                'compact_output':True
                }
        if self.__config.filter_kwargs is None:
            self.__config.filter_kwargs = {}
        if self.__config.noise_kwargs is None:
            self.__config.noise_kwargs = {}
        if self.__config.noise_kwargs_var2 is None:
            self.__config.noise_kwargs_var2 = {}
        if self.__config.velocity_perturbation_kwargs is None:
            self.__config.velocity_perturbation_kwargs = {}
        if self.__config.mask_kwargs is None:
            self.__config.mask_kwargs = {}

        print("Inputs validated and initialized successfully.")

    def __print_forecast_info(self):
        """
        Print information about the forecast setup, including inputs, methods, and parameters.
        """
        print("Computing STAN nowcast")
        print("-----------------------")
        print("")

        print("Inputs")
        print("------")
        print(f"input dimensions: {self.__var1.shape[1]}x{self.__var1.shape[2]}")
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
        if self.__var2 is not None:
            print(f"regression filter:        {self.__config.regression_filter_method}")
        
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
        print(f"{self.__config.var1_name} mask method:    {self.__config.mask_method}")
        print(f"{self.__config.var2_name} mask method:    {self.__config.mask_method_var2}")
        print(f"probability matching:   {self.__config.probmatching_method}")
        print(f"probability matching var2:   {self.__config.probmatching_method_var2}")
        print(f"FFT method:             {self.__config.fft_method}")
        print(f"domain:                 {self.__config.domain}")
        print(f"adaptive localization:  {self.__config.adaptive_localization}")
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
        print(f"order of the ARI(p,d) model: {self.__config.ar_order}")
        print(f"differencing order of the ARI(p,d) model: {self.__config.d_order}")
        print(f"ARI(p,d) window radius:      {self.__config.ar_window_radius}")
        
        if self.__config.velocity_perturbation_method == "bps":
            self.__params.velocity_perturbation_parallel = (
                self.__config.velocity_perturbation_kwargs.get(
                    "p_par", noise.motion.get_default_params_bps_par()
                )
            )
            self.__params.velocity_perturbation_perpendicular = (
                self.__config.velocity_perturbation_kwargs.get(
                    "p_perp", noise.motion.get_default_params_bps_perp()
                )
            )
            print(
                f"velocity perturbations, parallel:      {self.__params.velocity_perturbation_parallel[0]},{self.__params.velocity_perturbation_parallel[1]},{self.__params.velocity_perturbation_parallel[2]}"
            )
            print(
                f"velocity perturbations, perpendicular: {self.__params.velocity_perturbation_perpendicular[0]},{self.__params.velocity_perturbation_perpendicular[1]},{self.__params.velocity_perturbation_perpendicular[2]}"
            )

        if self.__config.var1_threshold is not None:
            print(f"{self.__config.var1_name}. threshold: {self.__config.var1_threshold}")
        if self.__config.var2_threshold is not None:
            print(f"{self.__config.var2_name}. threshold: {self.__config.var2_threshold}")
        
    def __initialize_nowcast_components(self):
        """
        Initialize the FFT, bandpass filters, decomposition methods, and extrapolation method.
        """
        # Initialize number of ensemble workers
        self.__params.num_ensemble_workers = min(
            self.__config.n_ens_members, self.__config.num_workers
        )
        
        # Slice the variable 1 field to only use the last ar_order + d_order + 1 fields
        self.__var1 = self.__var1[-(self.__config.ar_order + self.__config.d_order + 1) :, :, :].copy()
        
        # Slice the variable 2 field and save the last fields
        if self.__var2 is not None:
            if self.__var2.ndim == 2:
                self.__var2 = self.__var2[np.newaxis, :, :]
            self.__var2 = self.__var2[-(self.__config.ar_order + self.__config.d_order + 1) :, :, :].copy()
                
        M, N = self.__var1.shape[1:]  # Extract the spatial dimensions (height, width)

        # Initialize FFT method
        self.__params.fft = utils.get_method(
            self.__config.fft_method, shape=(M, N), n_threads=self.__config.num_workers
        )

        # Initialize the band-pass filter for the cascade decomposition
        filter_method = cascade.get_method(self.__config.bandpass_filter_method)
        self.__params.bandpass_filter = filter_method(
            (M, N),
            self.__config.n_cascade_levels,
            **(self.__config.filter_kwargs or {}),
        )
        
        # Get the decomposition method (e.g., FFT)
        self.__params.decomposition_method, self.__params.recomposition_method = (
            cascade.get_method(self.__config.decomposition_method)
        )
        
        # Get the extrapolation method (e.g., semilagrangian)
        self.__params.extrapolation_method = extrapolation.get_method(
            self.__config.extrapolation_method
        )
        
        # Generate the mesh grid for spatial coordinates
        x_values, y_values = np.meshgrid(np.arange(N), np.arange(M))
        self.__params.xy_coordinates = np.stack([x_values, y_values])

        # Determine the domain mask from non-finite values in the variable 1 data
        self.__params.domain_mask = np.logical_or.reduce(
            [~np.isfinite(self.__var1[i, :]) for i in range(self.__var1.shape[0])]
        )
        
        # Get minimum values for var1 and var2
        self.__params.var1_min = np.nanmin(self.__var1)
        self.__params.var2_min = np.nanmin(self.__var2) if self.__var2 is not None else self.__params.var1_min
        
        # Determine the variable 1 threshold mask if conditional is set
        if self.__config.conditional:
            self.__state.mask_threshold = np.logical_and.reduce(
                [
                    self.__var1[i, :, :] >= self.__config.var1_threshold
                    for i in range(self.__var1.shape[0])
                ]
            )
        else:
            self.__state.mask_threshold = None
        
        if self.__var2 is not None and self.__config.conditional_var2:
            self.__state.mask_threshold_var2 = np.logical_and.reduce(
                [
                    self.__var2[i, :, :] >= self.__config.var2_threshold
                    for i in range(self.__var2.shape[0])
                ]
            )
        else:
            self.__state.mask_threshold_var2 = None
        
        # Determine the coefficients fields of the relation var2 =a*var1+b by
        # localized linear regression
        if self.__var2 is not None and self.__config.r_vil_conversion_method == "loclinreg":
            self.__params.var1_var2_a, self.__params.var1_var2_b = self.__var1_var2_regression(self.__var1[-1, :])
            self.__params.global_var1_var2_a, self.__params.global_var1_var2_b = self.__get_global_var1_var2_regression()
        else:
            self.__params.var1_var2_a, self.__params.var1_var2_b = None, None
            self.__params.global_var1_var2_a, self.__params.global_var1_var2_b = None, None
            
        # Save last timestep to keep track
        self.__state.last_var1 = [self.__var1[-1, :]] * self.__config.n_ens_members
        if self.__var2 is not None:
            self.__state.last_var2 = [self.__var2[-1, :]] * self.__config.n_ens_members
        else:
            self.__state.last_var2 = self.__state.last_var1
        
        print("Nowcast components initialized successfully.")


    def __perform_extrapolation(self, is_var2=False):
        """
        Extrapolate (advect) variable fields based on the velocity field to align
        them in time. This prepares the fields for autoregressive modeling.
        """
        # 1. Select target variable
        if is_var2:
            var = self.__var2
            var_name = self.__config.var2_name
            if var.shape[0] < self.__config.ar_order + self.__config.d_order:
                return
        else:
            var = self.__var1
            var_name = self.__config.var1_name
            
        # 2. Setup arguments
        extrap_kwargs = self.__config.extrapolation_kwargs.copy()
        extrap_kwargs["xy_coords"] = self.__params.xy_coordinates
        extrap_kwargs["allow_nonfinite_values"] = (
            True if np.any(~np.isfinite(var)) else False
        )
        
        res = []
        
        # 3. Define nested worker function (Exact original structure)
        def __extrapolate_single_field(var, i):
            # Extrapolate a single variable field using the velocity field
            return self.__params.extrapolation_method(
                var[i, :, :],
                self.__velocity,
                self.__config.ar_order + self.__config.d_order - i,
                "min",
                **extrap_kwargs,
            )[-1]
        
        # 4. Loop and Execute (Exact original structure)
        for i in range(self.__config.ar_order + self.__config.d_order):
            if not DASK_IMPORTED:
                # If Dask is not available, perform sequential extrapolation
                var[i, :, :] = __extrapolate_single_field(var, i)
            else:
                # If Dask is available, accumulate delayed computations for parallel execution
                res.append(dask.delayed(__extrapolate_single_field)(var, i))

        # 5. Parallel Compute Block (Exact original structure)
        if DASK_IMPORTED and res:
            num_workers_ = min(self.__params.num_ensemble_workers, len(res))
            var = np.stack(
                list(dask.compute(*res, num_workers=num_workers_))
                + [var[-1, :, :]]
            )
            
        # 6. Save back to class
        if is_var2:
            self.__var2 = var
        else:
            self.__var1 = var

        print(f"Extrapolation complete and {var_name} fields aligned.")

    
    def __fix_var1_from_var2_extent(self, var1, var2, update=True, j=None, state=None, params=None):
        var1_threshold = self.__config.var1_threshold
        var2_threshold = self.__config.var2_threshold
        a_glar = self.__config.a_glar
        b_glar = self.__config.b_glar
        
        var1_dry = var1 < var1_threshold
        var2_wet = var2 >= var2_threshold
        mask_drywet = var1_dry & var2_wet
        
        var1[mask_drywet] = (var2[mask_drywet] - b_glar)/a_glar
        
        if update:
            self.__var1 = var1
        else:
            return var1
        

    def __decompose_field(self, field, mask):
        """
        Break a stack of fields (time, y, x) into cascade levels.
        
        Args:
            field: array (time, y, x) - input radar data
            mask: array (y, x) - valid observation areas
        
        Returns:
            cascades: array (time, levels, y, x) - decomposed fields
            decomposed: list - stats for last timestep (copied for ensemble)
        """
        decomposed_list = []
        # Process each time step
        for i in range(field.shape[0]):
            field_decomposed = self.__params.decomposition_method(
                field[i],  # Current time step
                self.__params.bandpass_filter,  # Filter for each cascade level
                mask=mask,
                fft_method=self.__params.fft,
                output_domain=self.__config.domain,
                **self.__config.decomp_kwargs,
            )
            decomposed_list.append(field_decomposed)
    
        # Stack into (time, levels, y, x) format
        cascades_4d = nowcast_utils.stack_cascades(decomposed_list, self.__config.n_cascade_levels)
        
        # Copy last timestep for all ensemble members
        last_decomposed = [decomposed_list[-1].copy() for _ in range(self.__config.n_ens_members)]
        return cascades_4d, last_decomposed

    
    def __initialize_noise_and_ar_model(self):
        """
        Main function to prepare data for nowcasting.
        
        Steps:
        1. Clean input data (fix NaN values)
        2. Setup noise generators for var1 and var2
        3. Break fields into cascade levels
        4. Setup AR model windows and masks
        5. Compute AR model coefficients
        6. Setup random number generators for ensemble
        """
        self.__clean_var_data()
        self.__setup_noise_generators()
    
        # Break var1 into cascade levels
        self.__state.var1_cascades, self.__state.var1_decomposed = \
            self.__decompose_field(self.__var1, self.__state.mask_threshold)
            
        # Break var2 into cascade levels (if used)
        if self.__var2 is not None:
            self.__state.var2_cascades, self.__state.var2_decomposed = \
                self.__decompose_field(self.__var2, self.__state.mask_threshold_var2)
                
        # Setup AR windows and adaptive masks
        self.__setup_ar_window_radius()
        if self.__config.adaptive_localization:
            var1_cascades_adaptive = self.__compute_adaptive_cascades_mask()
            if self.__var2 is not None:
                var2_cascades_adaptive = self.__compute_adaptive_cascades_mask(is_var2=True)
                var_cascades_adaptive = var2_cascades_adaptive.copy()
                var_cascades_adaptive[var2_cascades_adaptive == 0] = var1_cascades_adaptive[var2_cascades_adaptive == 0]
            else:
                var_cascades_adaptive = var1_cascades_adaptive
            self.__state.var1_cascades_adaptive = var_cascades_adaptive
            
        starttime = time.time() if self.__config.measure_time else None
        # Compute AR model parameters for var1
        self.__params.autocorrelation_coefficients, self.__params.ar_model_coefficients = \
            self.__compute_ar_coefficients(
                self.__state.var1_cascades,
                self.__state.mask_threshold,
                is_var2=False,
                update_phi0=self.__config.noise_method is not None and self.__config.d_order == 1,
            )
        self.__params.phi0 = self.__params.ar_model_coefficients[:, -1]
        if self.__config.measure_time:
            self.__measure_time(f"AR coefficients {self.__config.var1_name}", starttime)
        
        # Compute AR model parameters for var2 (if needed)
        if (self.__var2 is not None and 
            (self.__config.noise_method_var2 is not None or self.__config.mask_method_var2 or self.__config.phi_with_var2)): # == "sprog"
            starttime = time.time() if self.__config.measure_time else None
            self.__params.autocorrelation_coefficients_var2, self.__params.ar_model_coefficients_var2 = \
                self.__compute_ar_coefficients(
                    self.__state.var2_cascades,
                    self.__state.mask_threshold_var2,
                    is_var2=True,
                    update_phi0=self.__config.noise_method_var2 is not None and self.__config.d_order == 1,
                )
            
            self.__params.phi0_var2 = self.__params.ar_model_coefficients_var2[:, -1]
            if self.__config.measure_time:
                self.__measure_time(f"AR coefficients {self.__config.var2_name}", starttime)
                
            # Replace phi0 from var1 to phi0 from var2 (rainfall)
            if self.__config.phi_with_var2:
                print('phi0 replaced for phi0_var2')
                self.__params.phi0 = self.__params.phi0_var2 #/ self.__config.a_glar
                self.__params.ar_model_coefficients[:, -1] = self.__params.phi0
            
        # Trim cascades and setup random generators
        self.__finalize_cascades_and_rngs()
        
        print(f"AR model applied to {self.__config.var1_name} cascades.")
        if self.__config.noise_method is not None:
            print(f"Noise applied to {self.__config.var1_name} cascades.")
        if self.__var2 is not None and self.__config.noise_method_var2 is not None:
            print(f"Noise applied to {self.__config.var2_name} cascades.")


    def __clean_var_data(self):
        """
        Fix NaN and infinite values in var1 data.
        
        Replace bad values with the minimum valid value from each time step.
        """
        var1 = self.__var1.copy()
        for i in range(var1.shape[0]):  # Each time step
            nan_mask = ~np.isfinite(var1[i])
            var1[i, nan_mask] = np.nanmin(var1[i])  # Use min valid value
        self.__var1 = var1
        
        if self.__var2 is not None:
            var2 = self.__var2.copy()
            for i in range(var2.shape[0]):  # Each time step
                nan_mask = ~np.isfinite(var2[i])
                var2[i, nan_mask] = np.nanmin(var2[i])  # Use min valid value
            self.__var2 = var2
            
    
    def __setup_noise_generators(self):
        """
        Create noise generators for var1 (rain) and var2 (VIL).
        
        Sets up noise method and computes scaling factors for each cascade level.
        """
        # Setup var1 noise (main precipitation field)
        is_var1_probabilistic = self.__config.noise_method is not None
        self.__params.is_var1_probabilistic = is_var1_probabilistic
        if is_var1_probabilistic:
            np.random.seed(self.__config.seed)
            init_noise, generate_noise = noise.get_method(self.__config.noise_method)
            self.__params.noise_generator = generate_noise  # Used during forecast
            
            if self.__var2 is not None and self.__config.noise_with_var2: #noise correlated to var2
                self.__params.perturbation_generator = init_noise(  # Setup with current data
                    self.__var2, fft_method=self.__params.fft, **self.__config.noise_kwargs
                )
                # Compute noise scaling per cascade level
                self.__params.noise_std_coefficients = self.__compute_noise_std_coeffs(
                    self.__var2[-1], self.__config.var2_threshold, self.__config.noise_stddev_adj,
                    conditional=self.__config.conditional
                )
            else: #Normal STEPS noise correlated to var1
                self.__params.perturbation_generator = init_noise(  # Setup with current data
                    self.__var1, fft_method=self.__params.fft, **self.__config.noise_kwargs
                )
                # Compute noise scaling per cascade level
                self.__params.noise_std_coefficients = self.__compute_noise_std_coeffs(
                    self.__var1[-1], self.__config.var1_threshold, self.__config.noise_stddev_adj,
                    conditional=self.__config.conditional
                )
                
        else:
            # No noise - use defaults
            self.__params.perturbation_generator = None
            self.__params.noise_std_coefficients = None
    
        # Setup var2 noise (for VIL → rain conversion)
        is_var2_probabilistic = (
            self.__config.noise_method_var2 is not None or
            self.__config.mask_method_var2 == "stepsincremental" or
            self.__config.prob_conversion
        )
        self.__params.is_var2_probabilistic = is_var2_probabilistic
        if self.__var2 is not None and is_var2_probabilistic:
            if self.__config.noise_method_var2 is None:
                noise_method_var2 = "nonparametric"
            else:
                noise_method_var2 = self.__config.noise_method_var2
            np.random.seed(self.__config.seed)
            init_noise_var2, generate_noise_var2 = noise.get_method(noise_method_var2)
            self.__params.noise_generator_var2 = generate_noise_var2
            self.__params.perturbation_generator_var2 = init_noise_var2(
                self.__var2, fft_method=self.__params.fft, **self.__config.noise_kwargs_var2
            )
            self.__params.noise_std_coefficients_var2 = self.__compute_noise_std_coeffs(
                self.__var2[-1], self.__config.var2_threshold, self.__config.noise_stddev_adj_var2,
                conditional=self.__config.conditional_var2
            )
        else:
            self.__params.perturbation_generator_var2 = None
            self.__params.noise_std_coefficients_var2 = None
        
    
    def __compute_noise_std_coeffs(self, field, threshold, stddev_adj, conditional):
        """
        Compute noise scaling factors for each cascade level.
        
        Options:
        - 'auto': Measure from data (handles no-rain bias)
        - 'fixed': Use formula 1/(0.75 + 0.09*k)
        - None: Use 1.0 for all levels
        
        Args:
            field: array (y, x) - latest observation
            threshold: float - minimum rain value
            stddev_adj: str - adjustment method
            conditional: bool - only use pixels above threshold
        
        Returns:
            array (n_cascade_levels,) - scaling factors per cascade level
        """
        if stddev_adj == "auto":
            print("Computing noise adjustment coefficients... ", end="", flush=True)
            starttime = time.time() if self.__config.measure_time else None
            coeffs = noise.utils.compute_noise_stddev_adjs(
                field, threshold, np.min(field),
                self.__params.bandpass_filter, self.__params.decomposition_method,
                self.__params.perturbation_generator, self.__params.noise_generator,
                20, conditional=conditional, num_workers=self.__config.num_workers,
                seed=self.__config.seed
            )
            print(f"noise std. dev. coeffs:   {str(coeffs)}")
            if self.__config.measure_time and starttime:
                self.__measure_time("Noise adjustment", starttime)
            else:
                print("done.")
            return coeffs
        
        elif stddev_adj == "fixed":
            # Formula from BPS2006 paper
            func = lambda k: 1.0 / (0.75 + 0.09 * k)
            coeffs = np.array([func(k) for k in range(1, self.__config.n_cascade_levels + 1)])
            print(f"noise std. dev. coeffs:   {str(coeffs)}")
            return coeffs
        
        return np.ones(self.__config.n_cascade_levels)  # No adjustment
    
    
    def __setup_ar_window_radius(self):
        """
        Convert ar_window_radius config to array of window sizes.
        
        Handles these cases:
        - None → infinite (no localization)
        - int → same size for all levels  
        - list → per-level sizes
        - str → 'central' or 'sprogloc' formulas
        """
        # Handle adaptive localization defaults
        if self.__config.adaptive_localization:
            if self.__config.adaptive_localization == 1:
                self.__config.adaptive_localization = self.__config.n_cascade_levels
            if self.__config.ar_window_radius is None:
                self.__config.ar_window_radius = "central"
    
        if self.__config.ar_window_radius is None:
            self.__config.ar_window_radius = np.full(self.__config.n_cascade_levels, np.inf)
        elif isinstance(self.__config.ar_window_radius, (int, np.integer)):
            self.__config.ar_window_radius = np.full(self.__config.n_cascade_levels, self.__config.ar_window_radius)
        elif isinstance(self.__config.ar_window_radius, list):
            self.__config.ar_window_radius = np.array(self.__config.ar_window_radius)
        elif isinstance(self.__config.ar_window_radius, str):
            self.__config.ar_window_radius = self.__compute_window_sizes_from_string()
    
    def __compute_window_sizes_from_string(self):
        """
        Convert string window specs ('central', 'sprogloc') to radius array.
        
        'central': Uses bandpass filter wavenumbers → pixel radii
        'sprogloc': Geometric progression from SPROG paper
        
        Supports suffixes: _L, _r, _m# for variations.
        """
        bp_filter = self.__params.bandpass_filter
        L = max(bp_filter["shape"])  # Domain size
        ar_window_radius = self.__config.ar_window_radius
        
        if "central" in ar_window_radius:
            res = self.__config.kmperpixel
            wavenumbers = bp_filter["central_wavenumbers"]
            lk = L * res / (2 * wavenumbers)  # Wavelength → pixels
            
            if "_L" in ar_window_radius:
                lk = lk * L / lk[0]  # Normalize to domain size
            lk = [max(1, x) for x in lk]  # Minimum radius = 1
            
        else:  # sprogloc
            kmin = 2  # Starting scale
            K = self.__config.n_cascade_levels
            kw_2 = bp_filter["central_wavenumbers"][kmin - 1]
            q = (L / (kmin * kw_2)) ** (1 / (K - kmin))  # Geometric factor
            
            if "_L" in ar_window_radius:
                lk = [L * q ** (1 - k) for k in range(1, K + 1)]
            else:
                lk = [kmin * q ** (K - k) for k in range(1, K + 1)]
        
        if "_r" in ar_window_radius:
            lk = lk[::-1]  # Reverse order
        
        if "_m" in ar_window_radius: #set minimum window size e.g. central_m10, min size = 10
            wr_parts = ar_window_radius.split("_")
            m = [int(part[1:]) for part in wr_parts if "m" in part][0]
            lk = [l if l > m else m for l in lk]
                
        window_radius = np.array(lk, dtype=int)
        print("ARI(p,d) window radius:", window_radius)
        return window_radius

    def __compute_adaptive_cascades_mask(self, is_var2=False):
        """
        Create pixel-wise cascade selection map for adaptive localization.
        
        For each pixel, pick "best" cascade level (strongest signal).
        Uses only first N levels where N = adaptive_localization.
        """
        n_adaptive_cascades = self.__config.adaptive_localization
        
        if is_var2:
            var_cascades = self.__state.var2_cascades
            var_decomposed = self.__state.var2_decomposed[-1]
            mask_threshold = self.__state.mask_threshold_var2
        else:
            var_cascades = self.__state.var1_cascades
            var_decomposed = self.__state.var1_decomposed[-1]
            mask_threshold = self.__state.mask_threshold
        var_cascades_adaptive = var_cascades[:n_adaptive_cascades, -1].copy()
        
        # Normalize cascades if not already done
        if not var_decomposed["normalized"]:
            for i in range(n_adaptive_cascades):
                mu = var_decomposed["means"][i]
                sigma = var_decomposed["stds"][i]
                var_cascades_adaptive[i] = (var_cascades_adaptive[i] - mu) / sigma
    
        # Convert positive values → cascade level index (1,2,3...)
        for i in range(n_adaptive_cascades):
            level = var_cascades_adaptive[i]
            level[level <= 0] = 0      # No signal → 0
            level[level > 0] = i + 1   # Signal → cascade index
    
        # Assign highest-signal cascade per pixel (first-come priority)
        cascades_adaptive = np.zeros(self.__var1.shape[-2:])
        for i in range(n_adaptive_cascades):
            mask = (
                (var_cascades_adaptive[i] == i + 1) &  # This cascade's domain
                (cascades_adaptive == 0) &               # Not yet assigned
                (mask_threshold)            # Valid observation
            )
            cascades_adaptive[mask] = var_cascades_adaptive[i][mask]
        
        return cascades_adaptive
    
    
    def __compute_ar_coefficients(self, cascades, mask_threshold, is_var2=False, update_phi0=False):
        """
        Compute AR(p,d) model parameters from cascade autocorrelations.
        """
        var_name = self.__config.var2_name if is_var2 else self.__config.var1_name
        n_cascade_levels = self.__config.n_cascade_levels
        ar_order = self.__config.ar_order
        d_order = self.__config.d_order
        domain = self.__config.domain
    
        # Short-hand variables
        ad_loc = self.__config.adaptive_localization
        cad_loc = self.__config.constant_adaptive_localization
        ar_window_radius = self.__config.ar_window_radius
        gamma_filter_method = self.__config.gamma_filter_method
        fill_gamma_method = self.__config.fill_autocorrelation_coefficients_method
    
        M, N = cascades.shape[2], cascades.shape[3]
        
        # 1. Initialize Autocorrelation Arrays (gamma)
        gamma = np.empty((n_cascade_levels, ar_order, M, N))
        
        # Helper to broadcast 1D gamma to (ar_order, M, N)
        def _to_spatial_grid(gamma_in):
            if gamma_in.ndim == 1:
                # Shape (ar_order,) -> (ar_order, M, N)
                return np.tile(gamma_in[:, None, None], (1, M, N))
            return gamma_in
    
        # 2. adaptive_localization Logic
        if ad_loc:
            # Initialize array to save global autocorrelation coefficients (first window size)
            gamma_global = np.empty((n_cascade_levels, ar_order, M, N))
    
            ad_loc_len = len(ar_window_radius) if ad_loc == 1 else len(ar_window_radius[:ad_loc])
    
            # Iterate through all window radii
            for iwr in range(ad_loc_len):
                # Parallel (over k) computation of gamma_wr for this window radius
                if DASK_IMPORTED and self.__config.num_workers > 1:
                    tasks = [
                        dask.delayed(self.__temporal_autocorrelation)(
                            cascades[k],
                            d=d_order,
                            domain=domain,
                            x_shape=(M, N),
                            mask=mask_threshold,
                            window=gamma_filter_method,
                            window_radius=ar_window_radius[iwr],
                        )
                        for k in range(n_cascade_levels)
                    ]
                    results = dask.compute(*tasks, num_workers=self.__config.num_workers)
                    gamma_wr_list = [np.array(r) for r in results]
                else:
                    gamma_wr_list = []
                    for k in range(n_cascade_levels):
                        gamma_wr = np.array(
                            self.__temporal_autocorrelation(
                                cascades[k],
                                d=d_order,
                                domain=domain,
                                x_shape=(M, N),
                                mask=mask_threshold,
                                window=gamma_filter_method,
                                window_radius=ar_window_radius[iwr],
                            )
                        )
                        gamma_wr_list.append(gamma_wr)
    
                # Assign gamma to each pixel using adaptive localization logic from gamma_wr_list[k]
                for k in range(n_cascade_levels):
                    gamma_wr = gamma_wr_list[k]
    
                    # Store global fallback (Index 0 is considered global/first)
                    if iwr == 0:
                        if fill_gamma_method in ["global", 0]:
                            # Re-compute with infinite radius for true global
                            gamma_inf = np.array(
                                self.__temporal_autocorrelation(
                                    cascades[k],
                                    d=d_order,
                                    domain=domain,
                                    x_shape=(M, N),
                                    mask=mask_threshold,
                                    window=gamma_filter_method,
                                    window_radius=np.inf,
                                )
                            )
                            gamma_global[k] = _to_spatial_grid(gamma_inf)
    
                        elif fill_gamma_method in ["first", 1]:
                            # Use the first window result as global
                            gamma_global[k] = _to_spatial_grid(gamma_wr)
    
                    elif fill_gamma_method == (iwr + 1):
                        gamma_global[k] = _to_spatial_grid(gamma_wr)
    
                    # Assign coefficients according to cascades adaptive mask
                    # iwr+1 matches the index stored in var1_cascades_adaptive
                    if cad_loc:
                        mask_cascades_adaptive = self.__state.var1_cascades_adaptive == (iwr + 1)
                    else:  # "variable"
                        mask_cascades_adaptive = (
                            (self.__state.var1_cascades_adaptive == (iwr + 1))
                            & (k < iwr)
                            | (
                                (self.__state.var1_cascades_adaptive <= (iwr + 1))
                                & (self.__state.var1_cascades_adaptive != 0)
                                & (k == iwr)
                            )
                        )
                        
                    # Apply Mask with broadcasting
                    gamma_full = _to_spatial_grid(gamma_wr)
                    gamma[k][:, mask_cascades_adaptive] = gamma_full[:, mask_cascades_adaptive]
    
            # Assign fallback to unassigned pixels (mask == 0)
            zero_mask = gamma == 0 #np.isnan(gamma) #self.__state.var1_cascades_adaptive == 0
            gamma[zero_mask] = gamma_global[zero_mask]
    
        else:
            # Standard (Non-Adaptive) Logic
            if DASK_IMPORTED and self.__config.num_workers > 1:
                tasks = [
                    dask.delayed(self.__temporal_autocorrelation)(
                        cascades[k],
                        d=d_order,
                        domain=domain,
                        x_shape=(M, N),
                        mask=mask_threshold,
                        window=gamma_filter_method,
                        window_radius=ar_window_radius[k],
                    )
                    for k in range(n_cascade_levels)
                ]
                results = dask.compute(*tasks, num_workers=self.__config.num_workers)
                for k in range(n_cascade_levels):
                    gamma_wr = np.array(results[k])
                    gamma[k] = _to_spatial_grid(gamma_wr)
            else:
                for k in range(n_cascade_levels):
                    gamma_wr = np.array(
                        self.__temporal_autocorrelation(
                            cascades[k],
                            d=d_order,
                            domain=domain,
                            x_shape=(M, N),
                            mask=mask_threshold,
                            window=gamma_filter_method,
                            window_radius=ar_window_radius[k],
                        )
                    )
                    gamma[k] = _to_spatial_grid(gamma_wr)
    
        # 3. Post-Process Gamma (Factor & Clip)
        gamma *= self.__config.autocorrelation_coefficients_factor
        gamma[gamma >= 1] = 0.999999
    
        # Adjust lag-2 if AR(2)
        if ar_order == 2:
            for k in range(n_cascade_levels):
                gamma[k, 1] = autoregression.adjust_lag2_corrcoef2(
                    gamma[k, 0],
                    gamma[k, 1],
                )
    
        # 4. Compute Phi0 explicitly if needed (Noise Method & d=1)
        phi0s = []
        if update_phi0:
            gamma_phi0 = np.empty((n_cascade_levels, ar_order, M, N))
    
            for k in range(n_cascade_levels):
                gamma_phi0_ = np.array(
                    self.__temporal_autocorrelation(
                        cascades[k][1:],  # Original series shift
                        d=d_order,
                        domain=domain,
                        x_shape=(M, N),
                        mask=mask_threshold,
                        window=gamma_filter_method,
                        window_radius=np.inf,  # Always global for phi0 stability
                    )
                )
                gamma_phi0[k] = _to_spatial_grid(gamma_phi0_)
    
            # Adjust lag-2 for phi0 series
            if ar_order == 2:
                for k in range(n_cascade_levels):
                    gamma_phi0[k, 1] = autoregression.adjust_lag2_corrcoef2(
                        gamma_phi0[k, 0],
                        gamma_phi0[k, 1],
                    )
    
            # Calculate Phi0
            for k in range(n_cascade_levels):
                if ar_order == 1:
                    p1 = gamma_phi0[k][0, :]
                    phi0 = np.sqrt(np.maximum(1.0 - p1**2, 0))
                elif ar_order == 2:
                    g1 = gamma_phi0[k][0, :]
                    g2 = gamma_phi0[k][1, :]
                    p1 = (g1 * (1.0 - g2)) / (1.0 - g1**2)
                    p2 = (g2 - g1**2) / (1.0 - g1**2)
                    phi0 = np.sqrt(np.maximum(1.0 - p1 * g1 - p2 * g2, 0))
                phi0s.append(phi0)
                
        # 5. Estimate AR Parameters (Phi)
        phi = np.empty((n_cascade_levels, ar_order + d_order + 1, M, N))
        
        for k in range(n_cascade_levels):
            if ar_order > 2 or d_order > 1:
                phi[k, :] = autoregression.estimate_ar_params_yw_localized(
                    gamma[k], d=d_order
                )
            elif ar_order == 2:
                phi[k, :] = self.__estimate_ar2_params(
                    gamma[k], d=d_order
                )
            elif ar_order == 1:
                phi[k, :] = self.__estimate_ar1_params(
                    gamma[k], d=d_order
                )
    
        # Replace phi0 if computed
        if update_phi0:
            for k in range(n_cascade_levels):
                phi[k, -1] = phi0s[k]
        
        print(f"*** Summary ARI(p,d) for {var_name}")
        nowcast_utils.print_corrcoefs(np.nanmean(gamma, axis=(-2, -1)))
        nowcast_utils.print_ar_params(np.nanmean(phi, axis=(-2, -1)))
    
        return gamma, phi


    def __finalize_cascades_and_rngs(self):
        """
        Prepare final cascade structure and random generators.
        
        1. Trim cascades to AR order + differencing steps
        2. Reshape to ensemble format: list[member][level]
        3. Initialize var2 noise state (η₀ = 0)
        4. Setup RNGs for noise + motion perturbations
        """
        trim = self.__config.ar_order + self.__config.d_order
    
        # Reshape var1 cascades: list[ens][level] → used in nowcast loop
        self.__state.var1_cascades = [
            [self.__state.var1_cascades[k, -trim:].copy() 
             for k in range(self.__config.n_cascade_levels)]
            for _ in range(self.__config.n_ens_members)
        ]
        self.__state.var1_cascades_mask = self.__state.var1_cascades[0].copy()
        self.__state.var1_decomposed_mask = self.__state.var1_decomposed[0].copy()
        
        # Same for var2
        if self.__var2 is not None:
            self.__state.var2_cascades = [
                [self.__state.var2_cascades[k, -trim:].copy() 
                 for k in range(self.__config.n_cascade_levels)]
                for _ in range(self.__config.n_ens_members)
            ]
            self.__state.var2_cascades_mask = self.__state.var2_cascades[0].copy()
            self.__state.var2_decomposed_mask = self.__state.var2_decomposed[0].copy()
        else:
            self.__state.var2_cascades_mask = None
            self.__state.var2_decomposed_mask = None
            
        self.__state.last_eps = [None] * self.__config.n_ens_members
        self.__state.last_eps_var2 = [None] * self.__config.n_ens_members
        self.__initialize_random_generators()
    
    def __initialize_random_generators(self):
        """
        Create separate random number generators for each ensemble member.
        
        Three RNG types:
        1. var1 noise (main precipitation perturbations)
        2. var2 noise (VIL-correlated perturbations)
        3. motion perturbations (velocity field noise)
        
        Seeds updated sequentially for reproducibility.
        """
        is_var1_probabilistic = self.__params.is_var1_probabilistic
        is_var2_probabilistic = self.__params.is_var2_probabilistic
        is_vel_probabilistic = self.__config.velocity_perturbation_method is not None
        is_probabilistic = is_var1_probabilistic or is_var2_probabilistic or is_vel_probabilistic
        self.__params.is_vel_probabilistic = is_vel_probabilistic
        self.__params.is_probabilistic = is_probabilistic
        if is_probabilistic:
            # 1. Capture the master seed (ensure it's an integer)
            master_seed = self.__config.seed if self.__config.seed is not None else np.random.randint(0, 1e9)
            
            # 2. Define distinct starting points for each variable type
            s1 = master_seed + 0
            s2 = master_seed + 0 #rain always with same noise
            sm = master_seed + 1000000
            
            self.__state.random_generator_var1 = []
            self.__state.random_generator_var2 = []
            self.__state.random_generator_motion = []
            
            for _ in range(self.__config.n_ens_members):
                # Var1 noise RNG
                if is_var1_probabilistic:
                    rs1 = np.random.RandomState(s1)
                    self.__state.random_generator_var1.append(rs1)
                    s1 = rs1.randint(0, high=int(1e9))
                    
                # Var2 noise RNG
                if is_var2_probabilistic:
                    rs2 = np.random.RandomState(s2)
                    self.__state.random_generator_var2.append(rs2)
                    s2 = rs2.randint(0, high=int(1e9))
                    
                # Motion perturbation RNG
                if is_vel_probabilistic:
                    rsm = np.random.RandomState(sm)
                    self.__state.random_generator_motion.append(rsm)
                    sm = rsm.randint(0, high=int(1e9))
        else:
            self.__state.random_generator_var1 = None
            self.__state.random_generator_var2 = None
            self.__state.random_generator_motion = None

    # optimized version of timeseries.autoregression.estimate_ar_params_yw_localized
    # for an AR(1,1) modeL
    def __estimate_ar1_params(self, gamma, d):
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
            #Noise term
            phi.append(phi0)
            
        elif d == 1:
            # ARI(1,1) model (first differenced)
            phi1_d = 1 + phi1
            phi2_d = -phi1
            phi.append(phi1_d)
            phi.append(phi2_d)
            #Noise term
            phi.append(phi0)
        
        return np.array(phi)
    
    def __estimate_ar2_params(self, gamma, d):
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
            #Noise term
            phi.append(phi0)
        elif d == 1:
            # ARI(2,1) model (first differenced)
            phi1_d = 1 + phi1
            phi2_d = phi2 - phi1
            phi3_d = -phi2
            phi.append(phi1_d)
            phi.append(phi2_d)
            phi.append(phi3_d)
            # Noise term
            phi.append(phi0)
            
        return np.array(phi)
    
    def __temporal_autocorrelation(
        self,
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
                    cc = self.__moving_window_corrcoef(
                        x[-1, :], x[-(k + 2), :], window_radius, window=window, mask=mask
                    )
                    cc[~np.isfinite(cc)] = ccg
            else:
                cc = spectral.corrcoef(
                    x[-1, :, :], x[-(k + 2), :], x_shape, use_full_fft=use_full_fft
                )
            gamma.append(cc)
    
        return gamma
    
    
    def __moving_window_corrcoef(self, x, y, window_radius, window="gaussian", mask=None):
        if window not in ["gaussian", "uniform"]:
            raise ValueError(
                "unknown window type %s, the available options are 'gaussian' and 'uniform'"
                % window
            )
    
        if mask is None:
            mask = np.ones(x.shape)
        else:
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
                    "p_par": self.__config.velocity_perturbation_kwargs.get(
                        "p_par", self.__params.velocity_perturbation_parallel
                    ),
                    "p_perp": self.__config.velocity_perturbation_kwargs.get(
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


    def __initialize_variable_mask(self):
        """
        Initialize the variable 1 mask and handle different mask methods (sprog, incremental, obs).
        """
        # Local configuration aliases
        n_ens_members = self.__config.n_ens_members
        probmatching_method = self.__config.probmatching_method
        mask_method = self.__config.mask_method
        var1_threshold = self.__config.var1_threshold
        var1 = self.__var1
        var2 = self.__var2
        
        if probmatching_method == "mean":
            self.__params.variable1_mean = np.mean(
                var1[-1][var1[-1] >= var1_threshold]
            )
        
        if mask_method is not None:
            # Base Mask Calculation
            base_mask_var1 = (var1[-1] >= var1_threshold)
            
            if mask_method == "sprog":
                # Compute the wet area ratio and the variable 1 mask
                self.__params.wet_area_ratio = np.sum(base_mask_var1) / base_mask_var1.size
                self.__state.mask_var1 = [base_mask_var1.copy() for _ in range(n_ens_members)]
                
            elif "incremental" in mask_method:
                # Get mask parameters
                mask_kwargs = self.__config.mask_kwargs
                # Determine rim size (default 10)
                self.__params.mask_rim = mask_kwargs.get("mask_rim", 10)
                mask_f = mask_kwargs.get("mask_f", 1.0)
                
                # Initialize and expand the structuring element
                struct = generate_binary_structure(2, 1)
                n = mask_f * self.__config.timestep / self.__config.kmperpixel
                struct = iterate_structure(struct, int((n - 1) / 2.0))
                
                # Store for worker use
                self.__params.structuring_element = struct
                
                # Compute and apply the dilated mask
                dilated_mask = nowcast_utils.compute_dilated_mask(
                    base_mask_var1,
                    struct,
                    self.__params.mask_rim,
                )
                # Replicate for all members
                self.__state.mask_var1 = [dilated_mask.copy() for _ in range(n_ens_members)]
                
            elif mask_method == "obs":
                # Use the observed mask directly
                self.__state.mask_var1 = base_mask_var1.copy()
        else:
            # No mask / Full domain
            full_mask = np.ones_like(var1[-1], dtype=bool)
            self.__state.mask_var1 = full_mask
            
        print(f"{self.__config.var1_name} mask initialized successfully.")
        
        if var2 is not None:
            # Local aliases for Var2
            probmatching_method_var2 = self.__config.probmatching_method_var2
            mask_method_var2 = self.__config.mask_method_var2
            var2_threshold = self.__config.var2_threshold
            
            # Var 2 Mean
            if probmatching_method_var2 == "mean":
                self.__params.variable2_mean = np.mean(
                    var2[-1][var2[-1] >= var2_threshold]
                )
            
            # Var 2 Mask Logic
            if mask_method_var2 is not None:
                base_mask_var2 = (var2[-1] >= var2_threshold)
                if mask_method_var2 == "sprog":
                    self.__params.wet_area_ratio_var2 = np.sum(base_mask_var2) / base_mask_var2.size
                    self.__state.mask_var2 = [base_mask_var2.copy() for _ in range(n_ens_members)]
                    
                elif "incremental" in mask_method_var2:
                    mask_kwargs = self.__config.mask_kwargs
                    self.__params.mask_rim_var2 = mask_kwargs.get("mask_rim_var2", 10)
                    mask_f_var2 = mask_kwargs.get("mask_f_var2", 1.0)
                    
                    # Initialize and expand structure
                    struct_var2 = generate_binary_structure(2, 1)
                    n_var2 = mask_f_var2 * self.__config.timestep / self.__config.kmperpixel
                    struct_var2 = iterate_structure(struct_var2, int((n_var2 - 1) / 2.0))
                    
                    # Store for worker use
                    self.__params.structuring_element_var2 = struct_var2
                    
                    dilated_mask_var2 = nowcast_utils.compute_dilated_mask(
                        base_mask_var2,
                        struct_var2,
                        self.__params.mask_rim_var2,
                    )
                    self.__state.mask_var2 = [dilated_mask_var2.copy() for _ in range(n_ens_members)]
                    
                elif mask_method_var2 == "obs":
                    self.__state.mask_var2 = base_mask_var2.copy()
            else:
                full_mask_var2 = np.ones_like(var2[-1], dtype=bool)
                self.__state.mask_var2 = full_mask_var2
                
            print(f"{self.__config.var2_name} mask initialized successfully.")
            
            # Initialize Combined Mask
            self.__state.mask_vars = [None] * n_ens_members
            for j in range(n_ens_members):
                # Combine var1 and var2 masks (Intersection for initial state)
                self.__state.mask_vars[j] = np.logical_or(
                    self.__state.mask_var1[j],
                    self.__state.mask_var2[j]
                )
        else:
            # If no var2, mask_vars is just a clone of mask_var1
            self.__state.mask_vars = [m.copy() for m in self.__state.mask_var1]

        
    def __initialize_fft_objects(self):
        """
        Initialize FFT objects for each ensemble member.
        """
        self.__state.fft_objs = []
        for _ in range(self.__config.n_ens_members):
            fft_obj = utils.get_method(
                self.__config.fft_method, shape=self.__var1.shape[1:]
            )
            self.__state.fft_objs.append(fft_obj)
        print("FFT objects initialized successfully.")


    def __initialize_state(self):
        """
        Initialize the state dictionary used during the nowcast iteration.
        """
        return {
            "fft_objs": self.__state.fft_objs,
            # Masks
            "mask_var1": self.__state.mask_var1,
            "mask_var2": self.__state.mask_var2,
            "mask_vars": self.__state.mask_vars,
            # Random Generators
            "randgen_var1": self.__state.random_generator_var1,
            "randgen_var2": self.__state.random_generator_var2,
            # Var 1 Cascades & Decomp
            "var1_cascades": self.__state.var1_cascades,
            "var1_decomp": self.__state.var1_decomposed,
            "var1_cascades_adaptive": self.__state.var1_cascades_adaptive,
            # Var 2 Cascades & Decomp
            "var2_cascades": self.__state.var2_cascades,
            "var2_decomp": self.__state.var2_decomposed,
            # Var 1 and Var 2 Cascades & Decomp for deterministic masking as STEPS
            "var1_cascades_mask": self.__state.var1_cascades_mask,
            "var1_decomp_mask": self.__state.var1_decomposed_mask,
            "var2_cascades_mask": self.__state.var2_cascades_mask,
            "var2_decomp_mask": self.__state.var2_decomposed_mask,
            # History
            "last_eps": self.__state.last_eps,
            "last_eps_var2": self.__state.last_eps_var2,
            "last_var1": self.__state.last_var1,
            "last_var2": self.__state.last_var2,
            # Loop control
            "step": 0,
        }


    def __initialize_params(self, var1, var2=None):
        """
        Initialize the params dictionary used during the nowcast iteration.
        """
        return {
            # --- General Config ---
            "domain": self.__config.domain,
            "domain_mask": self.__params.domain_mask,
            "n_cascade_levels": self.__config.n_cascade_levels,
            "n_ens_members": self.__config.n_ens_members,
            "num_ensemble_workers": self.__params.num_ensemble_workers,
            
            # --- Variable Names & Inputs ---
            "var1_name": self.__config.var1_name,
            "var2_name": self.__config.var2_name,
            "var1": var1,
            "var2": var2,
            
            # --- Thresholds & Mins ---
            "var1_thr": self.__config.var1_threshold,
            "var2_thr": self.__config.var2_threshold,
            "var1_min": self.__params.var1_min,
            "var2_min": self.__params.var2_min,
            
            # --- Masking ---
            "mask_method": self.__config.mask_method,
            "mask_method_var2": self.__config.mask_method_var2,
            "mask_rim": self.__params.mask_rim,
            "mask_rim_var2": self.__params.mask_rim_var2,
            "struct": self.__params.structuring_element,
            "struct_var2": self.__params.structuring_element_var2,
            "war": self.__params.wet_area_ratio,
            "war_var2": self.__params.wet_area_ratio_var2,
            
            # --- Probability Matching ---
            "probmatching_method": self.__config.probmatching_method,
            "probmatching_method_var2": self.__config.probmatching_method_var2,
            "mu_0": self.__params.variable1_mean,
            "mu_0_var2": self.__params.variable2_mean,
            
            # --- Decomposition/Recomposition ---
            "decomp_method": self.__params.decomposition_method,
            "recomp_method": self.__params.recomposition_method,
            "filter": self.__params.bandpass_filter,
            "fft": self.__params.fft,
            
            # --- AR Model & Noise ---
            "noise_method": self.__config.noise_method,
            "noise_method_var2": self.__config.noise_method_var2,
            "generate_noise": self.__params.noise_generator,
            "generate_noise_var2": self.__params.noise_generator_var2,
            "pert_gen": self.__params.perturbation_generator,
            "pert_gen_var2": self.__params.perturbation_generator_var2,
            "vel_pert_method": self.__params.velocity_perturbation_method,
            "noise_levels": self.__config.noise_levels,
            "noise_levels_var2": self.__config.noise_levels_var2,
            "is_probabilistic": self.__params.is_probabilistic,
            "is_var1_probabilistic": self.__params.is_var1_probabilistic,
            "is_var2_probabilistic": self.__params.is_var2_probabilistic,
            "is_vel_probabilistic": self.__params.is_vel_probabilistic,
            
            # --- Coefficients (AR, Noise, AutoCorr) ---
            "phi": self.__params.ar_model_coefficients,
            "phi_var2": self.__params.ar_model_coefficients_var2,
            "phi0": self.__params.phi0,
            "phi0_var2": self.__params.phi0_var2,
            "noise_std_coeffs": self.__params.noise_std_coefficients,
            "noise_std_coeffs_var2": self.__params.noise_std_coefficients_var2,
            
            # --- VIL/Rain Conversion & Coupling ---
            "r_vil_conversion_method": self.__config.r_vil_conversion_method,
            "fill_global_regression": self.__config.fill_global_regression,
            "global_transition": self.__config.global_transition,
            "var1_var2_a": self.__params.var1_var2_a,
            "var1_var2_b": self.__params.var1_var2_b,
            "global_var1_var2_a": self.__params.global_var1_var2_a,
            "global_var1_var2_b": self.__params.global_var1_var2_b,
            "prob_conversion": self.__config.prob_conversion,
            
            "a_glar": self.__config.a_glar,
            "b_glar": self.__config.b_glar,
            "phi_glar": self.__config.phi_glar,
            "alpha": self.__config.alpha,
            "beta": self.__config.beta,
            "seed": self.__config.seed,
            
            "timestep": self.__config.timestep,
        }


    def __update_state(self, state, params):
        """
        Update the state during the nowcasting loop. This function handles the AR model iteration,
        noise generation, recomposition, and mask application for each ensemble member.
        """
        var1_forecast_out = [None] * params["n_ens_members"]
        var2_forecast_out = [None] * params["n_ens_members"]
        
        is_deterministic = params["noise_method"] is None and params["noise_method_var2"] is None
        # Update the deterministic AR(p) model if noise or sprog mask is used
        if is_deterministic or params["mask_method"] == "sprog":
            self.__update_deterministic_ar_model(state, params, is_var2=False)
        if params["var2"] is not None and (is_deterministic or params["mask_method_var2"] == "sprog"):
            self.__update_deterministic_ar_model(state, params, is_var2=True)
        
        # Worker function for each ensemble member
        def worker(j):
            #Create noise at the beginning, then just read it
            if params["is_var1_probabilistic"]:
                eps = self.__generate_and_decompose_noise(j, state, params, is_var2=False)
                state["last_eps"][j] = eps
            
            if params["is_var2_probabilistic"]:
                eps2 = self.__generate_and_decompose_noise(j, state, params, is_var2=True)
                state["last_eps_var2"][j] = eps2
            
            # 1. Update cascades/AR model for var1 (VIL)
            self.__apply_ar_model_to_cascades(j, state, params, is_var2=False)
            
            # 2. Recompose var1 (Deterministic Baseline)
            var1_forecast = self.__recompose_field(j, state, params, is_var2=False)
            var1_forecast = self.__apply_postprocessing(var1_forecast, j, state, params, is_var2=False)
            
            # 3. Handle var2 (Rain Rate)
            if params["var2"] is not None:
                # Update incremental mask as in STEPS and generate noise for the first time
                if params["mask_method_var2"] == "stepsincremental": #needed first just to create noise before adding to var2
                    # Compute mask for var2 according to STEPS masks
                    self.__apply_ar_model_to_cascades(j, state, params, is_var2=True)
                    var2_forecast_mask = self.__recompose_field(j, state, params, is_var2=True) #, for_mask=True
                
                # Convert VIL -> Rain (Deterministic)
                var2_forecast_det = self.__convert_var1_to_var2(var1_forecast, j, state, params)
                
                # Add noise to deterministic nowcast
                if params["noise_method_var2"] is not None:
                    # Decompose & Add Noise to Rain
                    var2_forecast = self.__add_noise_to_var(var2_forecast_det, j, state, params, is_var2=True)
                else:
                    var2_forecast = var2_forecast_det
                
                # Apply mask and postprocessing
                if params["mask_method_var2"] == "stepsincremental": #first apply but dont update and then update mask according to STEPS
                    var2_forecast = self.__apply_postprocessing(var2_forecast, j, state, params, is_var2=True, update_mask=False)
                    # Update mask for var2 according to STEPS masks
                    var2_forecast_mask = self.__apply_postprocessing(var2_forecast_mask, j, state, params, is_var2=True, update_mask=True)
                else:
                    var2_forecast = self.__apply_postprocessing(var2_forecast, j, state, params, is_var2=True, update_mask=True)
            else:
                var2_forecast = var1_forecast
            
            # Final domain masking
            var2_forecast[params["domain_mask"]] = np.nan
            
            # Store outputs
            var1_forecast_out[j] = var1_forecast
            var2_forecast_out[j] = var2_forecast
            
            # Update history for next timestep
            state["last_var1"][j] = var1_forecast_out[j]
            state["last_var2"][j] = var2_forecast_out[j]
            
        # Use Dask for parallel execution if available
        if DASK_IMPORTED and params["n_ens_members"] > 1 and params["num_ensemble_workers"] > 1:
            res = [dask.delayed(worker)(j) for j in range(params["n_ens_members"])]
            dask.compute(*res, num_workers=params["num_ensemble_workers"])
        else:
            for j in range(params["n_ens_members"]):
                worker(j)
        
        state["step"] += 1
        
        return np.stack(var2_forecast_out), state
    
    
    def __update_deterministic_ar_model(self, state, params, is_var2=False):
        """
        Update the deterministic AR(p) model for each cascade level if noise is disabled
        or if the sprog mask is used.
        """
        var = "var2" if is_var2 else "var1"
        var_cascades_mask = f"{var}_cascades_mask"
        var_decomp_mask = f"{var}_decomp_mask"
        phi = params[f"phi_{var}"] if is_var2 else params["phi"]
        n_cascade_levels = params["n_cascade_levels"]
        domain = params["domain"]
        recomp_method = params["recomp_method"]
        fft_obj = params["fft"]
        mask_state_var = f"mask_{var}"
        war = params[f"war_{var}"] if is_var2 else params["war"]
        mask_method = params[f"mask_method_{var}"] if is_var2 else params["mask_method"]
        
        for k in range(n_cascade_levels):
            state[var_cascades_mask][k] = autoregression.iterate_ar_model(
                state[var_cascades_mask][k],
                phi[k]
            )

        state[var_decomp_mask]["cascade_levels"] = [
            state[var_cascades_mask][k][-1] for k in range(n_cascade_levels)
        ]

        if domain == "spatial":
            state[var_decomp_mask]["cascade_levels"] = np.stack(
                state[var_decomp_mask]["cascade_levels"]
            )

        var_det_forecast = recomp_method(state[var_decomp_mask])

        if domain == "spectral":
            var_det_forecast = fft_obj.irfft2(var_det_forecast)

        if mask_method == "sprog":
            state[mask_state_var] = compute_percentile_mask(var_det_forecast, war)
            
    
    def __convert_var1_to_var2(self, var1_forecast, j, state, params):
        """
        Convert var1 (VIL) to var2 (Rain Rate).
        Methods:
          - "glar": AR(1) correction with optional stochastic sampling.
          - "loclinreg": Local linear regression (legacy).
        """
        method = params["r_vil_conversion_method"]
        var1_min = params["var1_min"]
        var2_thr = params["var2_thr"]
        var2_min = params["var2_min"]
        
        var2_forecast = np.full(var1_forecast.shape, var2_min)
        
        # 1. GLOBAL LINEAR AUTOREGRESSIVE (used in STAN)
        if method == "glar":
            a = params["a_glar"]
            b = params["b_glar"]
            phi = params["phi_glar"]
            prob_conversion = params["prob_conversion"]
            
            last_var1 = state["last_var1"][j]
            last_var2 = state["last_var2"][j]
            
            if np.isscalar(phi):
                # Shape (M, N)
                M, N = last_var1.shape[-2], last_var1.shape[-1]
                phi = np.full((M, N), float(phi))
            
            # Masks: wet in both VIL and rain at previous step
            mask_wetwet = (last_var1 > var1_min) & (last_var2 > var2_min)
            
            # Deterministic mean in dB
            var2_mean_dB      = b + a * var1_forecast
            var2_mean_prev_dB = b + a * last_var1
            
            res_prev = np.zeros_like(var2_forecast)
            res_prev[mask_wetwet] = (last_var2[mask_wetwet] - var2_mean_prev_dB[mask_wetwet])
                                                                               
            # AR(1) correction: β₀ + φ * r_{prev}
            res_corrected = np.zeros_like(var2_forecast)
            res_corrected[mask_wetwet] = phi[mask_wetwet] * res_prev[mask_wetwet] #+ beta0
            
            var2_forecast = var2_mean_dB + res_corrected
            
            # Stochastic occurrence: use logistic p_rain and correlated noise
            if prob_conversion:
                alpha = params["alpha"]
                beta = params["beta"]
                vil_activation_thr  = -16.87
                min_vil_trend       = 0
                min_neigh_wet_frac  = 0.1
                
                p_rain = 1 / (1 + np.exp(-(alpha + beta * var1_forecast)))
                recomp_method = params["recomp_method"]
                eps = state["last_eps_var2"][j]
                eps_field = recomp_method(eps)
                u = 0.5 + 0.5 * np.tanh(eps_field)
                rain_mask_base = (u < p_rain)
                
                vil_trend = var1_forecast - last_var1
                vil_trend_support = vil_trend > min_vil_trend
                prev_rain_mask = last_var2 > var2_min
                neigh_wet_frac = uniform_filter(prev_rain_mask.astype(float), size=3)
                neigh_support = neigh_wet_frac > min_neigh_wet_frac #original
                
                mask_var2 = state["mask_var2"][j].astype(bool)
                rain_mask = (
                    rain_mask_base &
                    (var1_forecast > vil_activation_thr) &
                    vil_trend_support &
                    neigh_support &
                    mask_var2
                )
                
                high_conf = mask_wetwet & mask_var2
                low_conf  = (~high_conf) & mask_var2
                mask_notwet = ~mask_wetwet
            
                var2_mean_dB_clamped = np.where(
                    var1_forecast > var1_min,
                    var2_mean_dB,
                    var2_min
                )
                
                var2_forecast = np.where(
                    low_conf & mask_notwet,
                    np.where(
                        rain_mask,
                        np.where(
                            var1_forecast < vil_activation_thr,
                            var2_thr,          # weak VIL → fixed minimum rain threshold
                            var2_mean_dB_clamped
                        ),
                        var2_min
                    ),
                    var2_forecast
                )
                
                var2_forecast[var2_forecast <= var2_min] = var2_min
                
        # 2. LOCALIZED LINEAR REGRESSION (modified from ANVIL)
        elif method == "loclinreg":
            var2_forecast_localized = params["var1_var2_a"] * 10**(var1_forecast/10) + params["var1_var2_b"]
            
            var2_forecast_localized[var2_forecast_localized < 10**(var2_thr/10)] = 0
            var2_forecast_localized = 10 * np.log10(var2_forecast_localized)
            
            var2_localized_nan_mask = np.isnan(var2_forecast_localized) | (var2_forecast_localized < var2_min)
            var2_forecast_localized[var2_localized_nan_mask] = var2_min
            
            if params["noise_method"] is not None or params["fill_global_regression"] or params["global_transition"]:
                var2_forecast_global = params["global_var1_var2_a"] * var1_forecast + params["global_var1_var2_b"]
                global_mask_condition = (var2_forecast_global < var2_thr) & state["mask_vars"][j].astype(bool)
                var2_forecast_global[global_mask_condition] = var2_min
                fill_mask = var2_localized_nan_mask & state["mask_vars"][j].astype(bool)
                var2_forecast[fill_mask] = var2_forecast_global[fill_mask]
            
            if params["global_transition"]:
                local_global_transition = self.__time_steps - 1
                if state['step'] <= local_global_transition:
                    local_weight = 1 - state['step'] / local_global_transition
                    global_weight = state['step'] / local_global_transition
                    mask_valid = ~var2_localized_nan_mask
                    var2_forecast[mask_valid] = (
                        var2_forecast_localized[mask_valid] * local_weight + 
                        var2_forecast_global[mask_valid] * global_weight
                    )
                else:
                    var2_forecast[~var2_localized_nan_mask] = var2_forecast_global[~var2_localized_nan_mask]
            else:
                var2_forecast[~var2_localized_nan_mask] = var2_forecast_localized[~var2_localized_nan_mask]
                
            var2_forecast[var2_forecast < var2_thr] = var2_min
            var2_forecast[np.isnan(var2_forecast)] = var2_min
        
        return var2_forecast
    
    
    def __apply_ar_model_to_cascades(self, j, state, params, is_var2=False):
        """
        Apply the ARI(p,d) model to the cascades for each ensemble member, including
        noise generation and normalization.
        """
        var = "var2" if is_var2 else "var1"
        var_cascades = f"{var}_cascades" 
        var_cascades_mask = f"{var}_cascades_mask"
        phi = params[f"phi_{var}"] if is_var2 else params["phi"]
        noise_std_coeffs = params[f"noise_std_coeffs_{var}"] if is_var2 else params["noise_std_coeffs"]
        n_cascade_levels = params["n_cascade_levels"]
        is_probabilistic = params["noise_method"] is not None or params["noise_method_var2"] is not None or params["vel_pert_method"] is not None
        noise_levels = None if is_var2 else params["noise_levels"]
        
        # Read noise if enabled
        eps = state[f"last_eps_{var}"][j] if is_var2 else state["last_eps"][j]
        
        # Iterate the ARI(p,d) model for each cascade level
        for k in range(n_cascade_levels):
            if eps is not None and (noise_levels is None or k < noise_levels):
                eps_ = eps["cascade_levels"][k]
                eps_ *= noise_std_coeffs[k]
            else:
                eps_ = None
            
            # Apply the AR(p) model with or without perturbations
            if is_probabilistic:
                state[var_cascades][j][k] = autoregression.iterate_ar_model(
                    state[var_cascades][j][k],
                    phi[k],
                    eps=eps_
                )
            else:
                # use the deterministic AR(p) model computed above if
                # perturbations are disabled
                state[var_cascades][j][k] = state[var_cascades_mask][k]
        
        eps = None
        eps_ = None
        
    
    def __add_noise_to_var(self, var_forecast, j, state, params, is_var2=False, noise=None):
        """
        Generate noise and add it to the cascade levels.
        """
        var = "var2" if is_var2 else "var1"
        var_thr = params[f"{var}_thr"]
        noise_std_coeffs = params["noise_std_coeffs_var2"] if is_var2 else params["noise_std_coeffs"]
        phi0_values = params["phi0_var2"] if is_var2 else params["phi0"]
        n_cascade_levels = params["n_cascade_levels"]
        noise_levels = params["noise_levels_var2"] if is_var2 else params["noise_levels"]
        domain = params["domain"]
        decomp_method = params["decomp_method"]
        recomp_method = params["recomp_method"]
        fft_obj = params["fft"]
        
        # 1. Decompose field
        mask_threshold = var_forecast >= var_thr
        var_cascades = decomp_method(
            var_forecast,
            params["filter"],
            mask=mask_threshold,
            fft_method=state["fft_objs"][j],
            input_domain=domain,
            output_domain=domain,
            **self.__config.decomp_kwargs,
        )
        
        # 2. Load correlated noise
        if noise is None:
            eps = state[f"last_eps_{var}"][j] if is_var2 else state["last_eps"][j]
        else:
            eps = decomp_method(
                noise,
                params["filter"],
                fft_method = state["fft_objs"][j],
                input_domain = domain,
                output_domain = domain,
                **self.__config.decomp_kwargs,
            )
        
        # 3. Add perturbations
        for k in range(n_cascade_levels):
            if noise_levels is None or k < noise_levels:
                eps_ = eps["cascade_levels"][k]
                eps_ *= noise_std_coeffs[k]
                
                phi0 = phi0_values[k]
                var_cascades["cascade_levels"][k] += phi0 * eps_
            
        eps = None
        eps_ = None
        
        var_forecast = recomp_method(var_cascades)
    
        if domain == "spectral":
            var_forecast = fft_obj.irfft2(var_forecast)
            
        return var_forecast
    
    
    def __generate_and_decompose_noise(self, j, state, params, is_var2=False):
        """
        Generate and decompose the noise field into cascades for a given ensemble member.
        If is_var2 is True, use the var2 noise generator and RNG, always based on var2.
        """
        if not is_var2:
            noise_generator = params["generate_noise"]
            perturbation_generator = params["pert_gen"]
            random_state = state["randgen_var1"][j]
        else:
            noise_generator = params["generate_noise_var2"]
            perturbation_generator = params["pert_gen_var2"]
            random_state = state["randgen_var2"][j]
        domain = params["domain"]
        
        eps = noise_generator(
            perturbation_generator,
            randstate = random_state,
            fft_method = state["fft_objs"][j],
            domain = domain,
        )
        
        eps = params["decomp_method"](
            eps,
            params["filter"],
            fft_method = state["fft_objs"][j],
            input_domain = domain,
            output_domain = domain,
            **self.__config.decomp_kwargs,
        )

        return eps


    def __recompose_field(self, j, state, params, is_var2=False):
        """
        Recompose var1/var2 from cascades.
        """
        var = "var2" if is_var2 else "var1"
        var_decomp = f"{var}_decomp"
        var_cascades = state[f"{var}_cascades"]
        n_cascade_levels = params["n_cascade_levels"]
        domain = params["domain"]
        recomp_method = params["recomp_method"]
        fft_obj = params["fft"]
        
        # Take last time slice
        state[var_decomp][j]["cascade_levels"] = [
            var_cascades[j][k][-1, :] for k in range(n_cascade_levels)
        ]
        
        # Stack & Recompose
        if domain == "spatial":
            state[var_decomp][j]["cascade_levels"] = np.stack(state[var_decomp][j]["cascade_levels"])
    
        var_forecast = recomp_method(state[var_decomp][j])
    
        if domain == "spectral":
            var_forecast = fft_obj[j].irfft2(var_forecast)
            
        return var_forecast
    
    
    def __apply_postprocessing(self, var_forecast, j, state, params, is_var2=False, update_mask=True):
        # Update mask if noise in var2 is None and apply mask for var1
        var = "var2" if is_var2 else "var1"
        mask_method = params[f"mask_method_{var}"] if is_var2 else params["mask_method"]
        
        #Apply mask before postprocessing
        var_forecast = self.__apply_var_mask(var_forecast, j, state, params, is_var2=is_var2)
        var_forecast = self.__probability_matching(var_forecast, j, state, params, is_var2=is_var2)
        var_forecast = self.__apply_var_mask(var_forecast, j, state, params, is_var2=is_var2)
        
        if mask_method is not None:
            if "incremental" in mask_method and update_mask:
                self.__update_mask(var_forecast, j, state, params, is_var2=is_var2)
        return var_forecast
    
    
    def __probability_matching(self, var_forecast, j, state, params, is_var2=False):
        """
        Apply probability matching (CDF or Mean).
        """
        var = "var2" if is_var2 else "var1"
        probmatching_method = params[f"probmatching_method_{var}"] if is_var2 else params["probmatching_method"]
        var_obs = params[var]
        var_thr = params[f"{var}_thr"]
        mu0 = params[f"mu_0_{var}"] if is_var2 else params["mu_0"]
        var_min = params[f"{var}_min"]
        
        if probmatching_method == "cdf":
            var_forecast = probmatching.nonparam_match_empirical_cdf(var_forecast, var_obs)
            var_forecast[var_forecast <= var_min] = var_min
            
        elif probmatching_method == "mean":
            mask_thr = var_forecast >= var_thr
            if np.sum(mask_thr) > 0:
                mu_fct = np.mean(var_forecast[mask_thr])
                var_forecast[mask_thr] = var_forecast[mask_thr] - mu_fct + mu0
        
        return var_forecast
    
    
    def __update_mask(self, var_forecast, j, state, params, is_var2=False):
        """
        Update mask states (incremental growth only) and apply.
        """
        var = "var2" if is_var2 else "var1"
        mask_method = params[f"mask_method_{var}"] if is_var2 else params["mask_method"]
        var_thr = params[f"{var}_thr"]
        struct = params[f"struct_{var}"] if is_var2 else params["struct"]
        mask_rim = params[f"mask_rim_{var}"] if is_var2 else params["mask_rim"]
        mask_state_var = f"mask_{var}"
        
        # Update & Apply Mask
        if "incremental" in mask_method:
            if mask_method == "obsincremental":
                prev_mask = state[mask_state_var][j].astype(bool) | (var_forecast >= var_thr)
            else:
                # Update incremental mask based on current forecast
                prev_mask = var_forecast >= var_thr
            state[mask_state_var][j] = nowcast_utils.compute_dilated_mask(
                prev_mask,
                struct,
                mask_rim,
            )
            
            
    def __apply_var_mask(self, var_forecast, j, state, params, is_var2=False):
        """
        Apply the variable mask (var1 or var2) to prevent new values from
        generating in areas where they were not observed.
        """
        # Determine variable-specific vars
        var = "var2" if is_var2 else "var1"
        var_min = params[f"{var}_min"]
        mask_method = params[f"mask_method_{var}"] if is_var2 else params["mask_method"]
        mask_var = state[f"mask_{var}"]
        
        if mask_method is not None:
            if "incremental" in mask_method:
                # Apply incremental mask (soft edge)
                var_forecast = var_min + (var_forecast - var_min) * mask_var[j]
                mask_ = var_forecast > var_min
            else:
                # Apply static/global mask
                mask_ = mask_var
            
            var_forecast[~mask_] = var_min
        
        return var_forecast

    
    def __var1_var2_regression(self, var1, b_fixed=True, a_fixed=False):
        """
        Performs a localized, threshold-aware linear regression between two variables (`var1` and `var2`)
        to estimate pixel-wise coefficients `a` and `b` such that:
        
            var2 ≈ a * var1 + b
        
        This method is adapted from the localized regression approach described in Section II.G of PCLH2020.
        It is designed for spatially adaptive modeling of the relationship between Vertically Integrated Liquid (VIL)
        and precipitation, using a moving window centered on each pixel.
        
        ### Core Workflow:
        - `var1` is the current VIL field in decibel (dB) scale; converted to linear scale.
        - `var2` is the previous timestep's precipitation field in dB; also converted to linear scale.
        - Thresholds for both variables are applied (in linear scale) to mask out low-signal regions.
        - Local statistics (mean, variance, covariance) are computed using either a Gaussian or uniform filter.
        - Regression coefficients are estimated using mean-centered formulas:
            a = (E[var1 * var2] - E[var1] * E[var2]) / (E[var1²] - E[var1]²)
            b = var2 - a * var1
          where E[·] denotes the local average over the window.
        - Regression is only applied where the local sample count exceeds a minimum threshold and variance is stable.
        - In regions where regression is unstable but data is valid, a fallback ratio `a = var2 / var1` is used with `b = 0`.
        
        ### Filter Options:
        - `"gaussian"`: Applies a Gaussian-weighted filter. Normalization is handled via local valid pixel count.
        - `"uniform"`: Applies a square uniform filter. Scaling is applied to recover raw sums.
        - Other methods are not supported and will raise a `ValueError`.
        
        ### Parameters:
        - var1 (np.ndarray): VIL input in dB scale.
        
        ### Returns:
        - a (np.ndarray): Estimated slope coefficient for each pixel.
        - b (np.ndarray): Estimated intercept coefficient for each pixel.
        
        ### Notes:
        - `var2` is retrieved from `self.__var2_last`, representing the previous timestep's precipitation.
        - Thresholds are defined in `self.__config.var1_threshold` and `self.__config.var2_threshold`.
        - The window size is determined by `self.__config.var1_var2_window_radius`.
        - Pixels failing stability checks are assigned default values: `a = 0`, `b = 0`.
        - This implementation avoids matrix inversion and uses direct mean-centered regression for numerical stability.
        """
        
        # Make a copy of var1 and convert from dB to linear scale
        var1 = var1.copy()
        var1 = 10**(var1 / 10)
        
        # Copy the previous timestep's var2 and convert from dB to linear scale
        var2 = self.__var2_last.copy()
        var2 = 10**(var2 / 10)
        
        # Convert configured thresholds from dB to linear scale
        var1_thr = 10**(self.__config.var1_threshold / 10)
        var2_thr = 10**(self.__config.var2_threshold / 10)
        
        # Create masks based on threshold exceedance
        reg_mask_var1 = var1 >= var1_thr
        reg_mask_var2 = var2 >= var2_thr
        
        # Combine masks to identify valid observation regions
        mask_obs = np.logical_and(reg_mask_var1, reg_mask_var2)
        
        # Set invalid regions to NaN for clarity
        var1[~mask_obs] = 0.0
        var2[~mask_obs] = 0.0
        
        # Choose filtering method and window size from config
        method = self.__config.regression_filter_method
        
        # Compute local sample count using mask
        window_size_nt = 2 * self.__config.var1_var2_window_radius + 1
        nt = uniform_filter(mask_obs.astype(float), window_size_nt, mode='constant') * window_size_nt**2 #rescale to get valid pixels
        
        # Define minimum valid pixels required for stable regression
        min_valid_pixels = 0.1 * window_size_nt**2
        
        stable_mask = nt >= min_valid_pixels
        
        # Select convolution filter and scaling method
        if method == "gaussian":
            convol_filter = gaussian_filter
            window_size = self.__config.var1_var2_window_radius
            n = convol_filter(mask_obs.astype(float), window_size, mode='constant')
            scale = np.where(stable_mask, 1.0 / n, np.nan) # Normalize Gaussian-weighted sums by local valid pixel count
        elif method == "uniform":
            convol_filter = uniform_filter
            window_size = 2 * self.__config.var1_var2_window_radius + 1
            n = convol_filter(mask_obs.astype(float), window_size, mode='constant') * window_size**2 #rescale to get valid pixels
            scale = np.where(stable_mask, window_size**2 / n, np.nan)  # Uniform filter returns mean over full window; rescale to account for valid pixels
        else:
            raise ValueError(f"Unknown regression_filter_method: {method}")
        
        # Compute local means and cross-products using selected filter
        sx = convol_filter(var1, window_size, mode="constant") * scale
        sx2 = convol_filter(var1 * var1, window_size, mode="constant") * scale
        sxy = convol_filter(var1 * var2, window_size, mode="constant") * scale
        sy = convol_filter(var2, window_size, mode="constant") * scale
        
        # Compute regression slope and intercept terms
        numerator = sxy - sx * sy
        denominator = sx2 - sx * sx
        a = np.full_like(var1, np.nan)
        b = np.full_like(var1, np.nan)
        
        # Mask for valid regression regions
        mask = (np.abs(denominator) > 1e-8) & (stable_mask)
        
        # Compute regression coefficients where valid
        a[mask] = numerator[mask] / denominator[mask]
        if b_fixed:
            b[mask] = var2[mask] - a[mask] * var1[mask]
        else:
            b[mask] = sy[mask] - a[mask] * sx[mask]
        
        # Fallback: use direct ratio where regression is unstable but data is valid
        extra_mask = ~mask & mask_obs | (a <= 0)
        if a_fixed:
            a[extra_mask] = var2[extra_mask] / var1[extra_mask]
        else:
            a[extra_mask] = sy[extra_mask] / sx[extra_mask]
        b[extra_mask] = 0.0
        
        return a, b
    
    
    def __get_global_var1_var2_regression(self, method="max"):
        # Make a copy of var1
        var1 = self.__var1.copy()
        var2 = self.__var2.copy()
        if var2.ndim == 2:
            var1 = var1[-1, :]
        else:
            var1 = var1[-len(var2), :]
        
        # Get configured thresholds
        var1_thr = self.__config.var1_threshold
        var2_thr = self.__config.var2_threshold
        
        # Create masks based on threshold exceedance
        reg_mask_var1 = var1 >= var1_thr
        reg_mask_var2 = var2 >= var2_thr
        
        # Combine masks to identify valid observation regions
        mask_obs = np.logical_and(reg_mask_var1, reg_mask_var2)
        
        if np.sum(mask_obs) > 10000: #Compute global values from last observation if there are enough pixels
            # Set invalid regions to NaN for clarity
            var1_mask = var1[mask_obs]
            var2_mask = var2[mask_obs]
            
            # Linear regression on all R-VIL data
            if method == "all":
                ma, mb = np.polyfit(var1_mask, var2_mask, 1)
                print(f"Obs fit: a={ma:.8f}, b={mb:.8f}")
            
            elif method == "max":
                # Get maximum repetition of R-VIL relation and linear regression
                hist_xy = np.histogram2d(var1_mask, var2_mask, bins=100)
                max_hist_ix = np.argmax(hist_xy[0], axis=0)
                max_hist_x = np.array([hist_xy[1][ix] for ix in max_hist_ix])
                max_hist_y = np.array([(hist_xy[2][iy] + hist_xy[2][iy+1])/2 for iy in range(100)])
                ma, mb = np.polyfit(max_hist_x, max_hist_y, 1)
                print(f"Max fit: a={ma:.8f}, b={mb:.8f}")
            
            elif method == "core":
                H, xedges, yedges = np.histogram2d(var1_mask, var2_mask, bins=100)
                Xc = 0.5 * (xedges[:-1] + xedges[1:])
                Yc = 0.5 * (yedges[:-1] + yedges[1:])
                Xg, Yg = np.meshgrid(Xc, Yc, indexing='ij')
                
                mask_mean = H > 10
                x_mean = Xg[mask_mean].flatten()
                y_mean = Yg[mask_mean].flatten()
                counts_mean = H[mask_mean].flatten()
                
                w = counts_mean
                ma, mb = np.polyfit(x_mean, y_mean, deg=1, w=np.sqrt(w))
                print(f"Robust fit: a={ma:.8f}, b={mb:.8f}")
            
        else: #Use predefined values estimated from from past events
            ma, mb = self.__config.a_glar, self.__config.b_glar
        
        return ma, mb
    
    
    def __get_phi_var1_var2_regression(self):
        """
        Robust pixel-wise AR(1) phi estimation from temporal var1 (VIL).
        """
        var1 = self.__var1.copy()
        var1_thr = self.__config.var1_threshold
        
        # AR(1) pairs
        var1_t   = var1[1:]
        var1_tm1 = var1[:-1]
        
        # Combine masks to identify valid observation regions
        valid = ~np.isnan(var1_t) & ~np.isnan(var1_tm1)
        
        num = np.nansum(var1_t * var1_tm1 * valid, axis=0)
        den = np.nansum(var1_tm1 * var1_tm1 * valid, axis=0)
        mask = den > 0
        phi = np.full_like(num, np.nan)
        phi[mask] = num[mask] / den[mask]
        phi = np.clip(phi, -0.999, 0.999)
        
        # Mask low-VIL pixels
        var1_last = var1[-1]
        mask_nan = var1_last < var1_thr
        phi[mask_nan] = np.nan
        
        # 6) Fill NaNs with global mean
        mean_phi = np.nanmean(phi)
        phi = np.where(np.isnan(phi), mean_phi, phi)
        
        return phi
    
    
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
        the inputs like var1, velocity, time_steps, or config.
        """
        # Re-initialize the state and parameters
        self.__state = StepsNowcasterState()
        self.__params = StepsNowcasterParams()

        # Reset time measurement variables
        self.__start_time_init = None
        self.__init_time = None
        self.__mainloop_time = None
        

# Wrapper function to preserve backward compatibility
@deprecate_args({"R": "var1", "V": "velocity", "R_thr": "var1_thr"}, "1.8.0")
def forecast(
    var1,
    velocity,
    timesteps,
    n_ens_members=24,
    n_cascade_levels=6,
    var2=None,
    var1_thr=None,
    var2_thr=None,
    var1_name="var1",
    var2_name="var2",
    ar_order=2,
    d_order=0,
    ar_window_radius=None,
    var1_var2_window_radius=3,
    gamma_adj_factor=1,
    kmperpixel=None,
    timestep=None,
    extrap_method="semilagrangian",
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    gamma_filter_method="gaussian",
    regression_filter_method="gaussian",
    noise_method="nonparametric",
    noise_stddev_adj=None,
    noise_levels=None,
    noise_levels_var2=None,
    prob_conversion=False,
    noise_method_var2=None,
    r_vil_conversion_method="loclinreg",
    noise_stddev_adj_var2=None,
    motion_field_general=None,
    vel_pert_method="bps",
    fill_global_regression=False,
    conditional=False,
    conditional_var2=False,
    adaptive_localization=False,
    fill_autocorrelation_coefficients_method="first",
    constant_adaptive_localization=True,
    noise_with_var2=False,
    phi_with_var2=False,
    probmatching_method="cdf",
    probmatching_method_var2="cdf",
    mask_method="incremental",
    mask_method_var2="incremental",
    global_transition=False,
    seed=None,
    num_workers=1,
    fft_method="numpy",
    domain="spatial",
    extrap_kwargs=None,
    decomp_kwargs=None,
    filter_kwargs=None,
    noise_kwargs=None,
    noise_kwargs_var2=None,
    vel_pert_kwargs=None,
    mask_kwargs=None,
    measure_time=False,
    callback=None,
    return_output=True,
):
    """
    Generate a nowcast ensemble using the STAN (STEPS and Autoregressive
    nowcasting using the VIL) method.

    This method combines the stochastic downscaling of STEPS with the vertical
    integrated liquid (VIL) modeling of ANVIL. It uses VIL as the primary
    prognostic variable for large-scale structure and autoregressive evolution,
    while coupling it with rainfall intensity for small-scale stochastic texture.

    Parameters
    ----------
    var1: array-like
        Array of shape (ar_order+d_order+1,m,n) containing the input variable 1 fields
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
    var2: array_like, optional
        Array of shape (m,n) containing the most recently observed
        field. If set to None, no var2(var1) conversion is done and the outputs
        are in the same units as the inputs.
    var1_thr: float, optional
        Specifies the threshold value for minimum observable value of variable 1.
        Required if mask_method is not None or conditional is True.
    var2_thr: float, optional
        Specifies the threshold value for minimum observable value of variable 2.
        Required if var2 is not None.
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
        Name of the noise generator to use for perturbating the variable 1
        field. See the documentation of pysteps.noise.interface. If set to None,
        no noise is generated.
    noise_stddev_adj: {'auto','fixed',None}, optional
        Optional adjustment for the standard deviations of the noise fields added
        to each cascade level. This is done to compensate incorrect std. dev.
        estimates of cascade levels due to presence of no-rain areas. 'auto'=use
        the method implemented in pysteps.noise.utils.compute_noise_stddev_adjs.
        'fixed'= use the formula given in :cite:`BPS2006` (eq. 6), None=disable
        noise std. dev adjustment.
    noise_levels: int, optional
        The cascade level threshold (0-based) for noise injection.
        Levels < noise_levels receive noise for var1 (VIL).
        Levels >= noise_levels receive noise for var2 (Rain).
        If None, standard behavior applies (noise on all levels if enabled).
    noise_method_var2: {'parametric','nonparametric','ssft','nested',None}, optional
        Name of the noise generator to use for perturbating the variable 2
        field. Defaults to 'nonparametric'.
    r_vil_conversion_method: {'glar', 'loclinreg', None}, optional
        Method used to convert VIL (var1) to Rain Rate (var2).
    noise_stddev_adj_var2: {'auto','fixed',None}, optional
        Optional adjustment for the standard deviations of the noise fields added
        to each cascade level for variable 2.
    vel_pert_method: {'bps',None}, optional
        Name of the noise generator to use for perturbing the advection field. See
        the documentation of pysteps.noise.interface. If set to None, the advection
        field is not perturbed.
    fill_global_regression: bool, optional
        If True, fills unobserved areas in the regression with global statistics.
    conditional: bool, optional
        If set to True, compute the statistics of the variable 1 field
        conditionally by excluding pixels where the values are below the
        threshold var1_thr.
    conditional_var2: bool, optional
        If set to True, compute the statistics of the variable 2 field
        conditionally by excluding pixels where the values are below the
        threshold var2_thr.
    adaptive_localization: bool, optional
        If True, applies adaptive localization to the AR parameter estimation.
    fill_autocorrelation_coefficients_method: str or int, optional
        Method to fill missing autocorrelation coefficients. Defaults to "first".
    constant_adaptive_localization: bool, optional
        If True, keeps the adaptive localization parameters constant.
    probmatching_method: {'cdf','mean',None}, optional
        Method for matching the statistics of the forecast field with those of
        the most recently observed one. 'cdf'=map the forecast CDF to the observed
        one, 'mean'=adjust only the conditional mean value of the forecast field
        in variable 1 areas, None=no matching applied. Using 'mean' requires
        that var1_thr and mask_method are not None.
    probmatching_method_var2: {'cdf','mean',None}, optional
        Method for matching the statistics of the variable 2 forecast field.
    mask_method: {'obs','sprog','incremental',None}, optional
        The method to use for masking no variable 1 areas in the forecast
        field. The masked pixels are set to the minimum value of the observations.
        'obs' = apply var1_thr to the most recently observed variable 1
        field, 'sprog' = use the smoothed forecast field from S-PROG,
        where the ARI(p,d) model has been applied, 'incremental' = iteratively
        buffer the mask with a certain rate (currently it is 1 km/min),
        None=no masking.
    mask_method_var2: {'obs','sprog','incremental',None}, optional
        The method to use for masking no variable 2 areas in the forecast field.
    global_transition: bool, optional
        If True, applies a global transition model.
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
        classical STEPS model). If "spectral", the ARI(2,d) models and stochastic
        perturbations are applied directly in the spectral domain to reduce
        memory footprint and improve performance :cite:`PCH2019b`.
    extrap_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the extrapolation
        method. See the documentation of pysteps.extrapolation.
    decomp_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the decomposition
        method.
    filter_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the filter method.
        See the documentation of pysteps.cascade.bandpass_filters.py.
    noise_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the initializer of
        the noise generator. See the documentation of pysteps.noise.fftgenerators.
    noise_kwargs_var2: dict, optional
        Optional dictionary containing keyword arguments for the variable 2
        noise generator.
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
        of the input variable 1 fields, respectively. This can be used, for
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
        variable 1 fields for each ensemble member. Otherwise, a None value
        is returned. The time series starts from t0+timestep, where timestep is
        taken from the input variable 1 fields. If measure_time is True, the
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
        ar_window_radius=ar_window_radius,
        var1_var2_window_radius=var1_var2_window_radius,
        var1_threshold=var1_thr,
        var2_threshold=var2_thr,
        var1_name=var1_name,
        var2_name=var2_name,
        kmperpixel=kmperpixel,
        timestep=timestep,
        extrapolation_method=extrap_method,
        decomposition_method=decomp_method,
        bandpass_filter_method=bandpass_filter_method,
        gamma_filter_method=gamma_filter_method,
        regression_filter_method=regression_filter_method,
        noise_method=noise_method,
        noise_stddev_adj=noise_stddev_adj,
        noise_levels=noise_levels,
        noise_levels_var2=noise_levels_var2,
        prob_conversion=prob_conversion,
        noise_method_var2=noise_method_var2,
        noise_stddev_adj_var2=noise_stddev_adj_var2,
        ar_order=ar_order,
        d_order=d_order,
        autocorrelation_coefficients_factor=gamma_adj_factor,
        motion_field_general=motion_field_general,
        velocity_perturbation_method=vel_pert_method,
        fill_global_regression=fill_global_regression,
        r_vil_conversion_method=r_vil_conversion_method,
        conditional=conditional,
        conditional_var2=conditional_var2,
        adaptive_localization=adaptive_localization,
        fill_autocorrelation_coefficients_method=fill_autocorrelation_coefficients_method,
        constant_adaptive_localization=constant_adaptive_localization,
        noise_with_var2=noise_with_var2,
        phi_with_var2=phi_with_var2,
        probmatching_method=probmatching_method,
        probmatching_method_var2=probmatching_method_var2,
        mask_method=mask_method,
        mask_method_var2=mask_method_var2,
        global_transition=global_transition,
        seed=seed,
        num_workers=num_workers,
        fft_method=fft_method,
        domain=domain,
        extrapolation_kwargs=extrap_kwargs,
        decomp_kwargs=decomp_kwargs,
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
        var1, velocity, timesteps, steps_config=nowcaster_config, var2=var2
    )
    forecast_steps_nowcast = nowcaster.compute_forecast()
    nowcaster.reset_states_and_params()
    # Call the appropriate methods within the class
    return forecast_steps_nowcast



