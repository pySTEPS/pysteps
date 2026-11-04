Parametric ensemble nowcasting
==============================

``pysteps.nowcasts.parametric.forecast`` takes three consecutive rain-rate
observations and returns an ensemble in mm/h. Transformation, motion estimation,
cascade decomposition, statistical estimation, and forecast evolution happen
inside the call. No intermediate files or database are needed.

For observations stored in a v2 ``dataset`` with three time slices, ordered
oldest to newest::

    from pysteps import nowcasts

    forecast = nowcasts.get_method("parametric")
    ensemble, diagnostics = forecast(
        dataset,
        timesteps=12,
        n_ens_members=20,
        seed=42,
        return_diagnostics=True,
    )
    rain = ensemble[ensemble.attrs["precip_var"]]

The precipitation variable must have dimensions ``(time, y, x)``, in any order,
and its name must be stored in ``dataset.attrs["precip_var"]``. Its ``units``
attribute must be ``mm/h`` and ``transform`` must be absent or ``None``.
Observations must be floating point, finite, and nonnegative. Missing pixels
are rejected, including when the latest observation is dry.

The spatial coordinates must be regularly spaced with square pixels, ``units``
of ``m`` or ``km``, and signed ``stepsize`` attributes matching the coordinates.
Descending y coordinates are supported without reversing the data. The three
observation timestamps must be increasing, regularly spaced datetime64 values.
``time.attrs["stepsize"]`` must give their interval in seconds. Pixel size and
cadence are derived from these coordinates. Optional ``kmperpixel`` (km) and
``timestep`` (minutes) overrides must agree with the dataset. In particular,
450-second observations retain their physical interval of 7.5 minutes.

The result is an xarray dataset whose precipitation has dimensions
``(ens_number, time, y, x)``, with ensemble members numbered from one. It
preserves spatial coordinates and projection metadata and adds a scalar
``forecast_reference_time`` equal to the newest observation time. Forecast
valid times start one observation interval later. Outputs are thresholded
linear rain rates in mm/h, with the occurrence threshold and zero dry value
recorded in their attributes. Untransformed output omits the ``transform``
attribute, allowing native xarray NetCDF serialization. The numerical engine
loads arrays eagerly.

Fixed motion can be supplied in ``velocity_x`` and ``velocity_y`` dataset
variables, each with dimensions ``(y, x)`` in any order. Their values are x/y
pixel displacements per observation interval. Velocity units can be omitted,
as in the upstream motion methods, or recorded as ``pixels/timestep``.
Both components must be present together and finite. Otherwise dense
Lucas-Kanade motion is estimated from the latest two transformed observations.
Statistical parameters and motion remain fixed throughout each forecast.

See :ref:`sphx_glr_auto_examples_plot_parametric_nowcast.py` for a runnable
synthetic dataset example. For the complete Swiss radar workflow, including
clipping, aggregation, plotting, and a lossless NetCDF round trip, see
:ref:`sphx_glr_auto_examples_plot_parametric_nowcast_swiss.py`. The common
dataset model is described in :py:mod:`pysteps.io.importers`.

The Swiss example also writes ``parametric_nowcast_swiss_member0.gif`` into
``rcparams.outputs["path_outputs"]`` (the current directory by default). It
plays the three QPE observations followed by the six forecasts of member index
zero, which has ``ens_number=1``. All frames share a map and colour scale, and
titles identify the data source and valid time. QPE retains measured rain
rates, while forecasts apply the configured occurrence threshold. The frames
show the actual five-minute fields, with no temporal interpolation.

NetCDF export
--------------

The returned dataset can be exported directly with xarray::

    import xarray as xr

    precip_var = ensemble.attrs["precip_var"]
    ensemble.to_netcdf(
        "forecast.nc",
        engine="netcdf4",
        encoding={precip_var: {"zlib": True, "complevel": 1}},
    )
    with xr.open_dataset("forecast.nc", engine="netcdf4") as reopened:
        xr.testing.assert_identical(ensemble, reopened.load())

This uses lossless compression and retains float64 rainfall, member identifiers,
valid times, forecast reference time, and available geographic metadata.
Projection metadata are preserved as supplied; this does not add missing CF
projection mappings. The older ``pysteps.io.exporters`` interface still expects
arrays and a separate metadata dictionary.

Callbacks continue to receive one NumPy array per lead. Native NetCDF export
here writes the assembled dataset; incremental writing through the older
exporter remains a separate interface.

Reference settings and regional overrides
-----------------------------------------

These defaults preserve the evaluated ``nowcast_v2`` engine configuration from
``psteps_param_paper/config/db/config.json`` and the WRNZ acceptance configuration,
combined with the identified numerical implementation and generation safeguards.
They are a configurable regional reference, not universal calibration constants.
The contribution record in ``doc/proposals/parametric_engine.md`` identifies
source revisions and validation limits.

Top-level settings:

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Meaning
   * - ``n_cascade_levels``
     - 6
     - Normalized Gaussian cascades
   * - ``rain_threshold``
     - 1.0
     - Occurrence threshold, mm/h
   * - ``scale_break``
     - 20.0
     - Fixed spectral break, km
   * - ``zerovalue``
     - threshold dBR - 0.1
     - Dry transformed value
   * - ``taper_sigma``
     - 5.0
     - Boundary-mask Gaussian width, pixels
   * - ``min_rain_fraction``
     - 0.025
     - Post-AR suppression, strictly below

``estimation_kwargs`` accepts these overrides (unknown keys raise):

=========================== ======================== ==============================
Key                         Default                  Meaning
=========================== ======================== ==============================
``lower_truncation``        1.25                     Latent Normal fit cutoff, dBR
``temporal_exponent``       0.81                     Timescale versus wavelength
``alpha``                   2.72                     Pre-adjustment lag-pair power
``reference_levels``        ``(0,)``                 Levels used for reference time
``reference_time``          None                     Optional time prior, minutes
``moment_distance_limit``   0.5                      Latent/ordinary selection
``beta_1_bounds``           (-3.9, just below -1.25) Large-scale slope bounds
``beta_2_min``              -4.0                     Small-scale lower bound
``minimum_beta_gap``        0.1                      beta_2 <= beta_1 - gap
``correlation_time_bounds`` (60.0, 720.0)            Reference-time bounds, minutes
``low_rain_fraction``       0.05                     Regional fallback below this
``fallback_beta_1``         -1.45                    Low-coverage slope replacement
``fallback_beta_2``         -3.253958088759979       Regional median replacement
``fallback_reference_time`` 330.61752959848894       Regional median, minutes
=========================== ======================== ==============================

Ordinary wet-pixel moments and fitted parent-Normal moments are returned
separately in diagnostics. The latent pair is selected only if finite, with
positive scale, and within the configured normalized Euclidean distance of
the ordinary pair. Otherwise the ordinary pair is used. Rain fraction and
marginal moments are not clipped to a climatology. Below the low-coverage
threshold, both slopes and reference time are replaced together before joint
bounds are applied. These regional fallback medians were saved in the source
``config/low_war_fallback.json`` from the Auckland archive; they can be replaced.

The default reference time uses the largest cascade only, as in the evaluated
engine. The standalone estimator defaults to three levels, reflecting the later
selected paper analysis. To investigate that configuration, explicitly set
``reference_levels=(0, 1, 2)`` and ``temporal_exponent=0.7718612877`` together.
That later configuration has component support but does not inherit the forecast
validation of the evaluated default. No automatic fallback drops failed levels.
Timescales use the one-sided discrete AR(2) ACF sum including lag zero; they are
persistence summaries rather than forecast-skill horizons. Nominal wavelengths
come from the actual Gaussian filter. Scale-specific times at or below one
timestep resolve to white noise, with the limitation recorded in diagnostics.

``noise_kwargs`` accepts ``spectral_bias=-0.5``, ``low_war_threshold=0.1``,
``max_attempts=5``, ``distance_threshold=1.0``,
``beta_variances=(0.067738, 0.076405)``, ``nbins=500``, and ``eps=0.001``.
The last two also control the post-AR marginal mapping. See
:doc:`calibrated_generator` for calibration and finite-grid limitations.

``motion_kwargs`` overrides dense Lucas-Kanade settings. Mapping values such
as ``lk_kwargs`` or ``fd_kwargs`` update the corresponding reference submapping.
The reference has window (31,31), three pyramid levels, minimum eigenvalue
3e-5, Shi-Tomasi corners (400, quality 0.025, separation 5), IDW interpolation
(8 neighbors, power 1.25), outlier settings 3.5 and 20, and declustering scale 8.
It is estimated from two images, once, with no velocity perturbations.

Dry and unavailable estimates
-----------------------------

If the newest field has no rain at the occurrence threshold, the ensemble is
zero, including when earlier observations were wet. Motion/noise generation is
unnecessary and the diagnostic status is ``dry_latest_observation``. Subthreshold
rain counts as dry for this method. Earlier constant fields give zero normalized
anomalies rather than a division by zero.

Positive rainfall requires at least two distinct wet-pixel values to estimate
its marginal. Failure raises a clear error. If spectral or temporal estimates
are unavailable outside the configured low-coverage fallback, the engine raises
instead of inventing estimates or reporting dry output. For example, partly dry
histories can lack a persistence estimate; users with an investigated temporal
prior can supply ``estimation_kwargs={"reference_time": 200.0}``. The raw failed
estimate remains visible, and bounds still apply to the supplied prior.

The post-AR fraction cutoff is separate: a valid positive target below 0.025
produces dry output by the reference policy. Set ``min_rain_fraction=0.0`` to
disable that suppression. The operational source's forecast-eligibility gate
at 0.03 is outside this computational engine. Missing data is always an error.

State evolution and boundaries
------------------------------

Each member starts from the newest and previous observed cascade states.
One- and two-step semi-Lagrangian backward trajectories are cached once. The
two-step trajectory retains two unit integration steps. Histories are sampled
with linear interpolation and zero normalized anomalies outside the domain.
Normalized stochastic innovations are then added everywhere, including inflow.
There is no second output-advection pass or periodic wrap of rainfall.

After the reference global normalization across cascade levels and space,
recomposition uses the innovation cascade's moments. Histogram-based marginal
mapping is applied to the recomposed field. The engine returns its inverse dBR
transform, then multiplies the transformed field by the smoothed one-step
domain mask and decomposes it again to form the next state. This nonlinear
feedback differs from native STEPS, which keeps its AR state separate from
output probability matching. Raw calibrated fields retain a rainfall marginal;
their cascade decomposition supplies normalized innovations.

All calculations and results are float64. They precede the source workflow's
NetCDF packing and storage. Outputs have no regional 200 mm/h cap; finite
generated values are required. No input arrays or settings are modified.

Randomness, diagnostics, and incremental output
-----------------------------------------------

A local NumPy SeedSequence spawns independent member streams. With the same
seed and settings, the leading members agree when ensemble size increases.
Global NumPy random state and supplied SeedSequence objects are unchanged.
When seed is omitted, diagnostics record the entropy needed to repeat the run.
Numerical implementations, calibration retry decisions, and method settings
are part of reproducibility.

``return_diagnostics=True`` returns ``(ensemble, diagnostics)`` with effective
settings, seed information, motion source/field, raw marginal and spectral
estimates, observed correlations/times, selected bounded parameters, safeguards,
AR coefficients, and calibration diagnostics indexed ``[member][lead]``.

A callback receives one complete ``(member, y, x)`` array in mm/h per lead.
Its copy can be modified without affecting state or retained output. Set
``return_output=False`` to return None while streaming; this retains cascade
histories and current fields rather than the full forecast horizon. Diagnostics
can still be requested. Callback exceptions propagate to the caller.

This first engine supports integer lead counts at the observation cadence,
fixed motion and parameters, complete grids, and AR(2). Irregular output times,
motion perturbations, and lead-dependent parameter blending remain extensions.
