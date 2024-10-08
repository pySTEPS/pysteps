"""
pysteps.nowcasts.steps
======================

Implementation of the STEPS stochastic nowcasting method as described in
:cite:`Seed2003`, :cite:`BPS2006` and :cite:`SPN2013`.

.. autosummary::
    :toctree: ../generated/

    forecast
"""

import numpy as np
from scipy.ndimage import generate_binary_structure, iterate_structure
import time

from pysteps import cascade
from pysteps import extrapolation
from pysteps import noise
from pysteps import utils
from pysteps.decorators import deprecate_args
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.postprocessing import probmatching
from pysteps.timeseries import autoregression, correlation
from pysteps.nowcasts.utils import compute_percentile_mask, nowcast_main_loop

try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


class StepsNowcaster:
    def __init__(self, precip, velocity, timesteps, **kwargs):
        # Store inputs and optional parameters
        self.precip = precip
        self.velocity = velocity
        self.timesteps = timesteps
        self.n_ens_members = kwargs.get("n_ens_members", 24)
        self.n_cascade_levels = kwargs.get("n_cascade_levels", 6)
        self.precip_thr = kwargs.get("precip_thr", None)
        self.kmperpixel = kwargs.get("kmperpixel", None)
        self.timestep = kwargs.get("timestep", None)
        self.extrap_method = kwargs.get("extrap_method", "semilagrangian")
        self.decomp_method = kwargs.get("decomp_method", "fft")
        self.bandpass_filter_method = kwargs.get("bandpass_filter_method", "gaussian")
        self.noise_method = kwargs.get("noise_method", "nonparametric")
        self.noise_stddev_adj = kwargs.get("noise_stddev_adj", None)
        self.ar_order = kwargs.get("ar_order", 2)
        self.vel_pert_method = kwargs.get("vel_pert_method", "bps")
        self.conditional = kwargs.get("conditional", False)
        self.probmatching_method = kwargs.get("probmatching_method", "cdf")
        self.mask_method = kwargs.get("mask_method", "incremental")
        self.seed = kwargs.get("seed", None)
        self.num_workers = kwargs.get("num_workers", 1)
        self.fft_method = kwargs.get("fft_method", "numpy")
        self.domain = kwargs.get("domain", "spatial")
        self.extrap_kwargs = kwargs.get("extrap_kwargs", None)
        self.filter_kwargs = kwargs.get("filter_kwargs", None)
        self.noise_kwargs = kwargs.get("noise_kwargs", None)
        self.vel_pert_kwargs = kwargs.get("vel_pert_kwargs", None)
        self.mask_kwargs = kwargs.get("mask_kwargs", None)
        self.measure_time = kwargs.get("measure_time", False)
        self.callback = kwargs.get("callback", None)
        self.return_output = kwargs.get("return_output", True)

        # Additional variables for internal state management
        self.fft = None
        self.bp_filter = None
        self.extrapolator_method = None
        self.domain_mask = None
        self.precip_cascades = None
        self.gamma = None
        self.phi = None
        self.pert_gen = None
        self.noise_std_coeffs = None
        self.randgen_prec = None
        self.randgen_motion = None
        self.velocity_perturbators = None
        self.precip_forecast = None
        self.mask_prec = None
        self.mask_thr = None

        # Initialize number of ensemble workers
        self.num_ensemble_workers = min(self.n_ens_members, self.num_workers)

    def compute_forecast(self):
        """
        Main loop for nowcast ensemble generation. This handles extrapolation,
        noise application, autoregressive modeling, and recomposition of cascades.
        """
        self._check_inputs()
        self._print_forecast_info()
        # Measure time for initialization
        if self.measure_time:
            start_time_init = time.time()

        self._initialize_nowcast_components()
        # Slice the precipitation field to only use the last ar_order + 1 fields
        self.precip = self.precip[-(self.ar_order + 1) :, :, :].copy()
        # Measure and print initialization time
        if self.measure_time:
            self._measure_time("Initialization", start_time_init)

        self._perform_extrapolation()

        self._apply_noise_and_ar_model()

        # Main forecasting loop for each timestep
        for t in range(len(self.timesteps)):
            # Measure time for each timestep
            if self.measure_time:
                start_time_loop = time.time()

            # Apply noise and autoregressive model
            self._apply_noise_and_ar_model()

            # Recompose the cascades into forecast fields
            self._recompose_cascades()

            # Optionally apply a mask if required
            if self.mask_method:
                self._apply_mask()

            # Measure and print time taken for each timestep
            if self.measure_time:
                self._measure_time(f"Timestep {t}", start_time_loop)

        print("Forecasting complete.")

    def _check_inputs(self):
        """
        Validate the inputs to ensure consistency and correct shapes.
        """

        if self.precip.ndim != 3:
            raise ValueError("precip must be a three-dimensional array")
        if self.precip.shape[0] < self.ar_order + 1:
            raise ValueError(
                f"precip.shape[0] must be at least ar_order+1, "
                f"but found {self.precip.shape[0]}"
            )
        if self.velocity.ndim != 3:
            raise ValueError("velocity must be a three-dimensional array")
        if self.precip.shape[1:3] != self.velocity.shape[1:3]:
            raise ValueError(
                f"Dimension mismatch between precip and velocity: "
                f"shape(precip)={self.precip.shape}, shape(velocity)={self.velocity.shape}"
            )
        if (
            isinstance(self.timesteps, list)
            and not sorted(self.timesteps) == self.timesteps
        ):
            raise ValueError("timesteps must be in ascending order")
        if np.any(~np.isfinite(self.velocity)):
            raise ValueError("velocity contains non-finite values")
        if self.mask_method not in ["obs", "sprog", "incremental", None]:
            raise ValueError(
                f"Unknown mask method '{self.mask_method}'. "
                "Must be 'obs', 'sprog', 'incremental', or None."
            )
        if self.precip_thr is None:
            if self.conditional:
                raise ValueError("conditional=True but precip_thr is not specified.")
            if self.mask_method is not None:
                raise ValueError("mask_method is set but precip_thr is not specified.")
            if self.probmatching_method == "mean":
                raise ValueError(
                    "probmatching_method='mean' but precip_thr is not specified."
                )
            if self.noise_method is not None and self.noise_stddev_adj == "auto":
                raise ValueError(
                    "noise_stddev_adj='auto' but precip_thr is not specified."
                )
        if self.noise_stddev_adj not in ["auto", "fixed", None]:
            raise ValueError(
                f"Unknown noise_stddev_adj method '{self.noise_stddev_adj}'. "
                "Must be 'auto', 'fixed', or None."
            )
        if self.kmperpixel is None:
            if self.vel_pert_method is not None:
                raise ValueError("vel_pert_method is set but kmperpixel=None")
            if self.mask_method == "incremental":
                raise ValueError("mask_method='incremental' but kmperpixel=None")
        if self.timestep is None:
            if self.vel_pert_method is not None:
                raise ValueError("vel_pert_method is set but timestep=None")
            if self.mask_method == "incremental":
                raise ValueError("mask_method='incremental' but timestep=None")

        # Handle None values for various kwargs
        if self.extrap_kwargs is None:
            self.extrap_kwargs = {}
        if self.filter_kwargs is None:
            self.filter_kwargs = {}
        if self.noise_kwargs is None:
            self.noise_kwargs = {}
        if self.vel_pert_kwargs is None:
            self.vel_pert_kwargs = {}
        if self.mask_kwargs is None:
            self.mask_kwargs = {}

        print("Inputs validated and initialized successfully.")

    def _print_forecast_info(self):
        """
        Print information about the forecast setup, including inputs, methods, and parameters.
        """
        print("Computing STEPS nowcast")
        print("-----------------------")
        print("")

        print("Inputs")
        print("------")
        print(f"input dimensions: {self.precip.shape[1]}x{self.precip.shape[2]}")
        if self.kmperpixel is not None:
            print(f"km/pixel:         {self.kmperpixel}")
        if self.timestep is not None:
            print(f"time step:        {self.timestep} minutes")
        print("")

        print("Methods")
        print("-------")
        print(f"extrapolation:          {self.extrap_method}")
        print(f"bandpass filter:        {self.bandpass_filter_method}")
        print(f"decomposition:          {self.decomp_method}")
        print(f"noise generator:        {self.noise_method}")
        print(
            "noise adjustment:       {}".format(
                ("yes" if self.noise_stddev_adj else "no")
            )
        )
        print(f"velocity perturbator:   {self.vel_pert_method}")
        print(
            "conditional statistics: {}".format(("yes" if self.conditional else "no"))
        )
        print(f"precip. mask method:    {self.mask_method}")
        print(f"probability matching:   {self.probmatching_method}")
        print(f"FFT method:             {self.fft_method}")
        print(f"domain:                 {self.domain}")
        print("")

        print("Parameters")
        print("----------")
        if isinstance(self.timesteps, int):
            print(f"number of time steps:     {self.timesteps}")
        else:
            print(f"time steps:               {self.timesteps}")
        print(f"ensemble size:            {self.n_ens_members}")
        print(f"parallel threads:         {self.num_workers}")
        print(f"number of cascade levels: {self.n_cascade_levels}")
        print(f"order of the AR(p) model: {self.ar_order}")

        if self.vel_pert_method == "bps":
            vp_par = self.vel_pert_kwargs.get(
                "p_par", noise.motion.get_default_params_bps_par()
            )
            vp_perp = self.vel_pert_kwargs.get(
                "p_perp", noise.motion.get_default_params_bps_perp()
            )
            print(
                f"velocity perturbations, parallel:      {vp_par[0]},{vp_par[1]},{vp_par[2]}"
            )
            print(
                f"velocity perturbations, perpendicular: {vp_perp[0]},{vp_perp[1]},{vp_perp[2]}"
            )

        if self.precip_thr is not None:
            print(f"precip. intensity threshold: {self.precip_thr}")

    def _initialize_nowcast_components(self):
        """
        Initialize the FFT, bandpass filters, decomposition methods, and extrapolation method.
        """
        M, N = self.precip.shape[1:]  # Extract the spatial dimensions (height, width)

        # Initialize FFT method
        self.fft = utils.get_method(
            self.fft_method, shape=(M, N), n_threads=self.num_workers
        )

        # Initialize the band-pass filter for the cascade decomposition
        filter_method = cascade.get_method(self.bandpass_filter_method)
        self.bp_filter = filter_method(
            (M, N), self.n_cascade_levels, **(self.filter_kwargs or {})
        )

        # Get the decomposition method (e.g., FFT)
        self.decomp_method, self.recomp_method = cascade.get_method(self.decomp_method)

        # Get the extrapolation method (e.g., semilagrangian)
        self.extrapolator_method = extrapolation.get_method(self.extrap_method)

        # Generate the mesh grid for spatial coordinates
        x_values, y_values = np.meshgrid(np.arange(N), np.arange(M))
        self.xy_coords = np.stack([x_values, y_values])

        # Determine the domain mask from non-finite values in the precipitation data
        self.domain_mask = np.logical_or.reduce(
            [~np.isfinite(self.precip[i, :]) for i in range(self.precip.shape[0])]
        )

        print("Nowcast components initialized successfully.")

    def _perform_extrapolation(self):
        """
        Extrapolate (advect) precipitation fields based on the velocity field to align
        them in time. This prepares the precipitation fields for autoregressive modeling.
        """
        # Determine the precipitation threshold mask if conditional is set
        if self.conditional:
            self.mask_thr = np.logical_and.reduce(
                [
                    self.precip[i, :, :] >= self.precip_thr
                    for i in range(self.precip.shape[0])
                ]
            )
        else:
            self.mask_thr = None

        extrap_kwargs = self.extrap_kwargs.copy()
        extrap_kwargs["xy_coords"] = self.xy_coords
        extrap_kwargs["allow_nonfinite_values"] = (
            True if np.any(~np.isfinite(self.precip)) else False
        )

        res = []

        def _extrapolate_single_field(precip, i):
            # Extrapolate a single precipitation field using the velocity field
            return self.extrapolator_method(
                precip[i, :, :],
                self.velocity,
                self.ar_order - i,
                "min",
                **extrap_kwargs,
            )[-1]

        for i in range(self.ar_order):
            if (
                not DASK_IMPORTED
            ):  # If Dask is not available, perform sequential extrapolation
                self.precip[i, :, :] = _extrapolate_single_field(self.precip, i)
            else:
                # If Dask is available, accumulate delayed computations for parallel execution
                res.append(dask.delayed(_extrapolate_single_field)(self.precip, i))

        # If Dask is available, perform the parallel computation
        if DASK_IMPORTED and res:
            num_workers_ = min(self.num_ensemble_workers, len(res))
            self.precip = np.stack(
                list(dask.compute(*res, num_workers=num_workers_))
                + [self.precip[-1, :, :]]
            )

        print("Extrapolation complete and precipitation fields aligned.")

    def _apply_noise_and_ar_model(self):
        """
        Apply noise and autoregressive (AR) models to precipitation cascades.
        This method applies the AR model to the decomposed precipitation cascades
        and adds noise perturbations if necessary.
        """
        # Make a copy of the precipitation data and replace non-finite values
        self.precip = self.precip.copy()
        for i in range(self.precip.shape[0]):
            # Replace non-finite values with the minimum finite value of the precipitation field
            self.precip[i, ~np.isfinite(self.precip[i, :])] = np.nanmin(
                self.precip[i, :]
            )

        # Initialize the noise generator if the noise_method is provided
        if self.noise_method is not None:
            np.random.seed(self.seed)  # Set the random seed for reproducibility
            init_noise, generate_noise = noise.get_method(
                self.noise_method
            )  # Get noise methods

            # Initialize the perturbation generator for the precipitation field
            self.pert_gen = init_noise(
                self.precip, fft_method=self.fft, **self.noise_kwargs
            )

            # Handle noise standard deviation adjustments if necessary
            if self.noise_stddev_adj == "auto":
                print("Computing noise adjustment coefficients... ", end="", flush=True)
                if self.measure_time:
                    starttime = time.time()

                # Compute noise adjustment coefficients
                self.noise_std_coeffs = noise.utils.compute_noise_stddev_adjs(
                    self.precip[-1, :, :],
                    self.precip_thr,
                    np.min(self.precip),
                    self.bp_filter,
                    self.decomp_method,
                    self.pert_gen,
                    generate_noise,
                    20,
                    conditional=self.conditional,
                    num_workers=self.num_workers,
                    seed=self.seed,
                )

                # Measure and print time taken
                if self.measure_time:
                    self._measure_time(
                        "Noise adjustment coefficient computation", starttime
                    )
                else:
                    print("done.")

            elif self.noise_stddev_adj == "fixed":
                # Set fixed noise adjustment coefficients
                func = lambda k: 1.0 / (0.75 + 0.09 * k)
                self.noise_std_coeffs = [
                    func(k) for k in range(1, self.n_cascade_levels + 1)
                ]

            else:
                # Default to no adjustment
                self.noise_std_coeffs = np.ones(self.n_cascade_levels)

            if self.noise_stddev_adj is not None:
                # Print noise std deviation coefficients if adjustments were made
                print(f"noise std. dev. coeffs:   {str(self.noise_std_coeffs)}")

        else:
            # No noise, so set perturbation generator and noise_std_coeffs to None
            self.pert_gen = None
            self.noise_std_coeffs = np.ones(
                self.n_cascade_levels
            )  # Keep default as 1.0 to avoid breaking AR model
        # TODO: The following parts of the method are not yet fully checked compared to the original

        # Decompose the input precipitation fields
        precip_decomp = []
        for i in range(self.ar_order + 1):
            precip_ = self.decomp_method(
                self.precip[i, :, :],
                self.bp_filter,
                mask=self.mask_thr,
                fft_method=self.fft,
                output_domain=self.domain,
                normalize=True,
                compute_stats=True,
                compact_output=True,
            )
            precip_decomp.append(precip_)

        # Normalize the cascades and rearrange them into a 4D array
        self.precip_cascades = nowcast_utils.stack_cascades(
            precip_decomp, self.n_cascade_levels
        )
        precip_decomp = precip_decomp[-1]
        precip_decomp = [precip_decomp.copy() for _ in range(self.n_ens_members)]

        # Compute temporal autocorrelation coefficients for each cascade level
        self.gamma = np.empty((self.n_cascade_levels, self.ar_order))
        for i in range(self.n_cascade_levels):
            self.gamma[i, :] = correlation.temporal_autocorrelation(
                self.precip_cascades[i], mask=self.mask_thr
            )

        nowcast_utils.print_corrcoefs(self.gamma)

        # Adjust the lag-2 correlation coefficient if AR(2) model is used
        if self.ar_order == 2:
            for i in range(self.n_cascade_levels):
                self.gamma[i, 1] = autoregression.adjust_lag2_corrcoef2(
                    self.gamma[i, 0], self.gamma[i, 1]
                )

        # Estimate the parameters of the AR model using autocorrelation coefficients
        self.phi = np.empty((self.n_cascade_levels, self.ar_order + 1))
        for i in range(self.n_cascade_levels):
            self.phi[i, :] = autoregression.estimate_ar_params_yw(self.gamma[i, :])

        nowcast_utils.print_ar_params(self.phi)

        # Discard all except the last ar_order cascades for AR model
        self.precip_cascades = [
            self.precip_cascades[i][-self.ar_order :]
            for i in range(self.n_cascade_levels)
        ]

        # Stack the cascades into a list containing all ensemble members
        self.precip_cascades = [
            [self.precip_cascades[j].copy() for j in range(self.n_cascade_levels)]
            for _ in range(self.n_ens_members)
        ]

        # Initialize random generators if noise_method is provided
        if self.noise_method is not None:
            self.randgen_prec = []
            for _ in range(self.n_ens_members):
                rs = np.random.RandomState(self.seed)
                self.randgen_prec.append(rs)
                self.seed = rs.randint(0, high=int(1e9))
        else:
            self.randgen_prec = None

        print("AR model and noise applied to precipitation cascades.")

    def _recompose_cascades(self):
        """
        Recompose the cascades into the final precipitation forecast fields.
        """
        # Logic to recompose cascades back into forecast fields
        pass

    def _apply_mask(self):
        """
        Apply the precipitation mask (if applicable) to the forecast fields.
        """
        # Logic for applying the mask based on the mask method
        pass

    def _update_state(self):
        """
        Update the internal state of the nowcasting system after each loop iteration.
        """
        # Logic to handle updates to internal state (e.g., for velocity, precipitation, etc.)
        pass

    def _measure_time(self, label, start_time):
        """
        Measure and print the time taken for a specific part of the process.

        Parameters:
        - label: A description of the part of the process being measured.
        - start_time: The timestamp when the process started (from time.time()).
        """
        if self.measure_time:
            elapsed_time = time.time() - start_time
            print(f"{label} took {elapsed_time:.2f} seconds.")


# Wrapper function to preserve backward compatibility
def forecast(
    precip,
    velocity,
    timesteps,
    n_ens_members=24,
    n_cascade_levels=6,
    precip_thr=None,
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
    # Create an instance of the new class with all the provided arguments
    nowcaster = StepsNowcaster(
        precip,
        velocity,
        timesteps,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        precip_thr=precip_thr,
        kmperpixel=kmperpixel,
        timestep=timestep,
        extrap_method=extrap_method,
        decomp_method=decomp_method,
        bandpass_filter_method=bandpass_filter_method,
        noise_method=noise_method,
        noise_stddev_adj=noise_stddev_adj,
        ar_order=ar_order,
        vel_pert_method=vel_pert_method,
        conditional=conditional,
        probmatching_method=probmatching_method,
        mask_method=mask_method,
        seed=seed,
        num_workers=num_workers,
        fft_method=fft_method,
        domain=domain,
        extrap_kwargs=extrap_kwargs,
        filter_kwargs=filter_kwargs,
        noise_kwargs=noise_kwargs,
        vel_pert_kwargs=vel_pert_kwargs,
        mask_kwargs=mask_kwargs,
        measure_time=measure_time,
        callback=callback,
        return_output=return_output,
    )

    # Call the appropriate methods within the class
    return nowcaster.compute_forecast()


@deprecate_args({"R": "precip", "V": "velocity", "R_thr": "precip_thr"}, "1.8.0")
def forecast(
    precip,
    velocity,
    timesteps,
    n_ens_members=24,
    n_cascade_levels=6,
    precip_thr=None,
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

    # compute the cascade decompositions of the input precipitation fields
    precip_decomp = []
    for i in range(ar_order + 1):
        precip_ = decomp_method(
            precip[i, :, :],
            bp_filter,
            mask=mask_thr,
            fft_method=fft,
            output_domain=domain,
            normalize=True,
            compute_stats=True,
            compact_output=True,
        )
        precip_decomp.append(precip_)

    # normalize the cascades and rearrange them into a four-dimensional array
    # of shape (n_cascade_levels,ar_order+1,m,n) for the autoregressive model
    precip_cascades = nowcast_utils.stack_cascades(precip_decomp, n_cascade_levels)

    precip_decomp = precip_decomp[-1]
    precip_decomp = [precip_decomp.copy() for _ in range(n_ens_members)]

    # compute lag-l temporal autocorrelation coefficients for each cascade level
    gamma = np.empty((n_cascade_levels, ar_order))
    for i in range(n_cascade_levels):
        gamma[i, :] = correlation.temporal_autocorrelation(
            precip_cascades[i], mask=mask_thr
        )

    nowcast_utils.print_corrcoefs(gamma)

    if ar_order == 2:
        # adjust the lag-2 correlation coefficient to ensure that the AR(p)
        # process is stationary
        for i in range(n_cascade_levels):
            gamma[i, 1] = autoregression.adjust_lag2_corrcoef2(gamma[i, 0], gamma[i, 1])

    # estimate the parameters of the AR(p) model from the autocorrelation
    # coefficients
    phi = np.empty((n_cascade_levels, ar_order + 1))
    for i in range(n_cascade_levels):
        phi[i, :] = autoregression.estimate_ar_params_yw(gamma[i, :])

    nowcast_utils.print_ar_params(phi)

    # discard all except the p-1 last cascades because they are not needed for
    # the AR(p) model
    precip_cascades = [precip_cascades[i][-ar_order:] for i in range(n_cascade_levels)]

    # stack the cascades into a list containing all ensemble members
    precip_cascades = [
        [precip_cascades[j].copy() for j in range(n_cascade_levels)]
        for _ in range(n_ens_members)
    ]

    # initialize the random generators
    if noise_method is not None:
        randgen_prec = []
        randgen_motion = []
        for _ in range(n_ens_members):
            rs = np.random.RandomState(seed)
            randgen_prec.append(rs)
            seed = rs.randint(0, high=1e9)
            rs = np.random.RandomState(seed)
            randgen_motion.append(rs)
            seed = rs.randint(0, high=1e9)
    else:
        randgen_prec = None

    if vel_pert_method is not None:
        init_vel_noise, generate_vel_noise = noise.get_method(vel_pert_method)

        # initialize the perturbation generators for the motion field
        velocity_perturbators = []
        for j in range(n_ens_members):
            kwargs = {
                "randstate": randgen_motion[j],
                "p_par": vp_par,
                "p_perp": vp_perp,
            }
            vp = init_vel_noise(velocity, 1.0 / kmperpixel, timestep, **kwargs)
            velocity_perturbators.append(
                lambda t, vp=vp: generate_vel_noise(vp, t * timestep)
            )
    else:
        velocity_perturbators = None

    precip_forecast = [[] for _ in range(n_ens_members)]

    if probmatching_method == "mean":
        mu_0 = np.mean(precip[-1, :, :][precip[-1, :, :] >= precip_thr])
    else:
        mu_0 = None

    precip_m = None
    precip_m_d = None
    war = None
    struct = None
    mask_rim = None

    if mask_method is not None:
        mask_prec = precip[-1, :, :] >= precip_thr

        if mask_method == "sprog":
            # compute the wet area ratio and the precipitation mask
            war = 1.0 * np.sum(mask_prec) / (precip.shape[1] * precip.shape[2])
            precip_m = [precip_cascades[0][i].copy() for i in range(n_cascade_levels)]
            precip_m_d = precip_decomp[0].copy()
        elif mask_method == "incremental":
            # get mask parameters
            mask_rim = mask_kwargs.get("mask_rim", 10)
            mask_f = mask_kwargs.get("mask_f", 1.0)
            # initialize the structuring element
            struct = generate_binary_structure(2, 1)
            # iterate it to expand it nxn
            n = mask_f * timestep / kmperpixel
            struct = iterate_structure(struct, int((n - 1) / 2.0))
            # initialize precip mask for each member
            mask_prec = nowcast_utils.compute_dilated_mask(mask_prec, struct, mask_rim)
            mask_prec = [mask_prec.copy() for _ in range(n_ens_members)]
    else:
        mask_prec = None

    if noise_method is None and precip_m is None:
        precip_m = [precip_cascades[0][i].copy() for i in range(n_cascade_levels)]

    fft_objs = []
    for _ in range(n_ens_members):
        fft_objs.append(utils.get_method(fft_method, shape=precip.shape[1:]))

    if measure_time:
        init_time = time.time() - starttime_init

    precip = precip[-1, :, :]

    print("Starting nowcast computation.")

    # the nowcast iteration for each ensemble member
    state = {
        "fft_objs": fft_objs,
        "mask_prec": mask_prec,
        "precip_cascades": precip_cascades,
        "precip_decomp": precip_decomp,
        "precip_m": precip_m,
        "precip_m_d": precip_m_d,
        "randgen_prec": randgen_prec,
    }
    params = {
        "decomp_method": decomp_method,
        "domain": domain,
        "domain_mask": domain_mask,
        "filter": bp_filter,
        "fft": fft,
        "generate_noise": generate_noise,
        "mask_method": mask_method,
        "mask_rim": mask_rim,
        "mu_0": mu_0,
        "n_cascade_levels": n_cascade_levels,
        "n_ens_members": n_ens_members,
        "noise_method": noise_method,
        "noise_std_coeffs": noise_std_coeffs,
        "num_ensemble_workers": num_ensemble_workers,
        "phi": phi,
        "pert_gen": pert_gen,
        "probmatching_method": probmatching_method,
        "precip": precip,
        "precip_thr": precip_thr,
        "recomp_method": recomp_method,
        "struct": struct,
        "war": war,
    }

    precip_forecast = nowcast_main_loop(
        precip,
        velocity,
        state,
        timesteps,
        extrap_method,
        _update,
        extrap_kwargs=extrap_kwargs,
        velocity_pert_gen=velocity_perturbators,
        params=params,
        ensemble=True,
        num_ensemble_members=n_ens_members,
        callback=callback,
        return_output=return_output,
        num_workers=num_ensemble_workers,
        measure_time=measure_time,
    )
    if measure_time:
        precip_forecast, mainloop_time = precip_forecast

    if return_output:
        precip_forecast = np.stack(
            [np.stack(precip_forecast[j]) for j in range(n_ens_members)]
        )
        if measure_time:
            return precip_forecast, init_time, mainloop_time
        else:
            return precip_forecast
    else:
        return None


def _update(state, params):
    precip_forecast_out = [None] * params["n_ens_members"]

    if params["noise_method"] is None or params["mask_method"] == "sprog":
        for i in range(params["n_cascade_levels"]):
            # use a separate AR(p) model for the non-perturbed forecast,
            # from which the mask is obtained
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

    def worker(j):
        if params["noise_method"] is not None:
            # generate noise field
            eps = params["generate_noise"](
                params["pert_gen"],
                randstate=state["randgen_prec"][j],
                fft_method=state["fft_objs"][j],
                domain=params["domain"],
            )

            # decompose the noise field into a cascade
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
        else:
            eps = None

        # iterate the AR(p) model for each cascade level
        for i in range(params["n_cascade_levels"]):
            # normalize the noise cascade
            if eps is not None:
                eps_ = eps["cascade_levels"][i]
                eps_ *= params["noise_std_coeffs"][i]
            else:
                eps_ = None
            # apply AR(p) process to cascade level
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

        # compute the recomposed precipitation field(s) from the cascades
        # obtained from the AR(p) model(s)
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

        if params["mask_method"] is not None:
            # apply the precipitation mask to prevent generation of new
            # precipitation into areas where it was not originally
            # observed
            precip_forecast_min = precip_forecast.min()
            if params["mask_method"] == "incremental":
                precip_forecast = (
                    precip_forecast_min
                    + (precip_forecast - precip_forecast_min) * state["mask_prec"][j]
                )
                mask_prec_ = precip_forecast > precip_forecast_min
            else:
                mask_prec_ = state["mask_prec"]

            # set to min value outside mask
            precip_forecast[~mask_prec_] = precip_forecast_min

        if params["probmatching_method"] == "cdf":
            # adjust the CDF of the forecast to match the most recently
            # observed precipitation field
            precip_forecast = probmatching.nonparam_match_empirical_cdf(
                precip_forecast, params["precip"]
            )
        elif params["probmatching_method"] == "mean":
            mask = precip_forecast >= params["precip_thr"]
            mu_fct = np.mean(precip_forecast[mask])
            precip_forecast[mask] = precip_forecast[mask] - mu_fct + params["mu_0"]

        if params["mask_method"] == "incremental":
            state["mask_prec"][j] = nowcast_utils.compute_dilated_mask(
                precip_forecast >= params["precip_thr"],
                params["struct"],
                params["mask_rim"],
            )

        precip_forecast[params["domain_mask"]] = np.nan

        precip_forecast_out[j] = precip_forecast

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
