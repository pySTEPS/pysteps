# Scope proposal: parametric ensemble nowcasting

Status: agreed development scope and broad public-interface outline for the
`parametric-nowcasting` branch; proposed for upstream discussion. This document
does not imply upstream approval. The forecast API described below is now
implemented in this branch.

Date: 14 October 2026.

## Objective and endpoint

Contribute a self-contained parametric ensemble nowcasting method to pySTEPS,
derived from the scientific components of `pysteps_param`.

The public endpoint is:

**Three consecutive rainfall fields → internal preparation and parameter
estimation → a nowcast ensemble.**

An ordinary caller supplies rainfall observations and forecast settings. The
engine performs transformation, motion estimation, cascade decomposition,
statistical parameter estimation, temporal modelling, and ensemble generation.
The caller does not need to prepare or persist intermediate artifacts.

The contribution includes the scientific components needed to reach this
endpoint, with those components also available for independent use. Database
services and operational orchestration remain in projects that depend on
pySTEPS. This boundary follows the direction discussed in
[pySTEPS issue #472](https://github.com/pySTEPS/pysteps/issues/472).

## Public forecast contract

### Observations

- Accept a floating-point NumPy `ndarray` with shape `(3, y, x)` containing rain
  rates in `mm/h`.
- Order the fields from oldest to newest, at a regular observation interval.
- Require consistent grid geometry and units across all three fields.
- Treat the newest observation as the forecast reference time.
- Use the three fields for AR(2) initialization and lag-correlation estimation.

The first implementation targets complete, finite input grids. Invalid or
missing observations produce a clear error rather than being silently converted
to dry rainfall. A confirmed dry input history is a valid physical case and must
have defined dry-output behaviour. Sparse rainfall and partially dry histories
require explicit, documented estimation safeguards.

### Forecast settings

The caller specifies the number of forecast timesteps, ensemble size, observation
timestep, and pixel size. The initial method advances at the observation timestep;
arbitrary output intervals and irregular lead times are subsequent extensions.
Units must be explicit and consistent with the surrounding pySTEPS interfaces.

The method supplies documented defaults for rainfall thresholds, cascade
settings, statistical fitting, temporal scaling, and generator calibration.
Users can override these settings for their region and data. Random-number
control is part of the public interface.

Motion is estimated internally by default. An optional externally supplied
velocity field, with shape `(2, y, x)` in pixels per observation timestep,
supports controlled comparisons and applications with an existing motion
estimate. Motion remains fixed through the initial forecast and is shared across
members.

### Results

Return rainfall in `mm/h` as a floating-point NumPy `ndarray` with shape
`(member, lead_time, y, x)`. The first forecast is one timestep after the newest
observation; the input observation is not included as forecast lead zero.

The default return value is the ensemble. Optional diagnostics provide the
estimated parameters, effective settings, motion estimate, fitting and fallback
status, and relevant generation diagnostics. An incremental-output callback can
follow the existing pySTEPS forecast conventions and remain independent of any
storage backend.

The precise function signature, keyword names, and diagnostic container are
implementation decisions to agree with upstream maintainers. These decisions
must preserve the three-field entry point and internal estimation.

NumPy arrays are the computational input and output throughout the engine.
Timestep, pixel size, and method settings are supplied separately; the forecast
does not require labelled coordinates or an xarray object. Diagnostics can use
ordinary Python containers and NumPy arrays. Applications may wrap the resulting
ensemble in xarray for labelled coordinates, analysis, or storage.

## Scientific method included in the contribution

The engine assembles the following stages:

1. Validate the rainfall history and forecast settings, and resolve the reference
   defaults together with any user overrides.
2. Apply the configured rainfall transformation, retaining threshold and dry-value
   metadata. The reference parametric method operates in dBR.
3. Estimate motion from the input history, unless the caller supplies velocity.
4. Decompose the observations into cascades and prepare the history needed by the
   forecast recursion and Lagrangian correlation estimation.
5. Estimate rainfall occurrence, ordinary wet-pixel moments, latent marginal
   parameters, and spectral slopes at the configured scale break.
6. Estimate cascade correlations and construct the reference persistence and
   temporal scaling model using the method supported by the paper experiments.
7. Generate calibrated stochastic fields and decompose them into normalized
   cascade innovations.
8. Advance each ensemble member using advection, the cascade AR model, and those
   innovations; recompose and apply the reference marginal adjustment.
9. Prepare the adjusted field's cascade state for the next forecast step,
   including the reference boundary taper, and return rainfall in physical units.

The initial forecast holds the estimated statistical parameter set fixed through
lead time. Each member evolves independently from the same observed initial
history, with its own reproducible random stream.

The port must retain the evaluated state-evolution method. In the existing
implementation, the marginal-adjusted field is tapered and decomposed again to
form the next state. This feedback is scientifically consequential and belongs
inside the engine. Adding only a noise option to native STEPS would not reproduce
the complete method.

Generated stochastic rainfall fields retain their rainfall marginal; they are
not themselves guaranteed to have global mean zero and variance one. Cascade
decomposition supplies normalized innovations. Marginal adjustment and realised
spectral matching must be described with their finite-grid approximations and
diagnostics.

## Reference defaults and scientific evidence

The default configuration represents the method evaluated in the paper workflow
and WRNZ experiments. It provides a usable reference implementation and can be
modified by users who have investigated their region and data.

Keep the following distinction explicit:

| Estimated from the three observations | Configurable method settings and defaults |
| --- | --- |
| Rainfall occurrence and marginal parameters | Rain threshold, dry value, and marginal fitting cutoff |
| Spectral parameters | Scale break, spectral fitting conventions, and calibration corrections |
| Motion and observed cascade correlations | Motion-estimator settings, cascade definition, and temporal scaling settings |
| Reference persistence estimate | Estimation safeguards, constraints, and any regional fallback values |

Ordinary retained wet-pixel moments and fitted latent Normal parameters are
different quantities and must remain distinguishable. Rain occurrence and
latent-distribution fitting may use different thresholds. Temporal correlation
times must retain an explicit definition and timestep convention; they are not
forecast-skill horizons.

Numerical reference defaults will be transcribed from the identified evaluated
configuration during extraction. They must not be assembled from unrelated test
configurations or silently replaced with settings from another model revision.
Record the effective values and overrides in diagnostics. Regional empirical
fallbacks must be identified as such and be configurable.

The evidence sources are:

| Project | Role and starting points |
| --- | --- |
| `pysteps_param` | Numerical implementation, component tests, and method notes; especially `rain_stats_compute.py`, `stochastic_generator.py`, `shared_utils.py`, and `cli/nowcast.py` under `src/pysteps_param/` |
| `psteps_param_paper` | Paper methods, calibration investigations, reproducible figures, and scientific comparisons; start with `README.md`, `docs/`, and `workflows/` |
| `pysteps_param_wrnz_cases` | Regional experiments and forecast-output evaluation; start with `experiments/calibrated_field_generator_acceptance/` and `experiments/nowcast_v2_acceptance/` |

The paper repository is locally named `psteps_param_paper` and was referred to as
`pysteps_param_paper` in the scope discussion.

As of this proposal, the author reports that the manuscript is close to submission
to the *Quarterly Journal of the Royal Meteorological Society*. It is not yet a
published reference. The author also reports that subsequent output evaluation
following a minor bug finding supports continued use of the reference defaults.
The scope therefore does not require retuning those defaults before porting.

During extraction, associate the reference configuration with exact source
revisions, experiment manifests, and relevant evaluation artifacts. Record the
bug and the implementation revision covered by the subsequent evaluation in that
evidence record. This proposal records the author's assessment; it does not claim
an independent rerun of that evaluation. Historical experiment results must retain
their original implementation and configuration identities.

## Reusable components and integration

The forecast function provides the normal user entry point. Its components remain
independently callable for scientific investigation and testing.

| Component | Integration direction |
| --- | --- |
| Spectral estimation and prescribed-break fitting | Extend or reuse `pysteps.utils.spectral`, preserving documented numerical conventions |
| Marginal statistics and lower-truncated Normal fitting | Provide array-based statistical utilities used by the engine |
| Correlation times and temporal scaling | Integrate with `pysteps.timeseries` and reuse cascade-filter frequencies |
| Calibrated stochastic field generation | Add a distinct method under `pysteps.noise` with explicit target parameters and diagnostics |
| Marginal adjustment | Reuse or extend `pysteps.postprocessing.probmatching` where conventions agree |
| Forecast preparation and state evolution | Add a parametric method under `pysteps.nowcasts` and register it through the usual method interface |
| Scientific parameter representation | Use a small container separating estimated values, effective settings, and diagnostics from operational identities |

These module placements are proposals, not a requirement to create a new package
hierarchy. Reuse existing primitives when doing so preserves the evaluated
method. Isolate kernels from database imports and avoid requiring pandas or
xarray solely to estimate parameters or generate an ensemble.

The current `StepsParameters` object combines scientific values, database
metadata, and operational QA. Extract the scientific representation and the
method safeguards needed by the forecast. Storage identities and operational
audit persistence stay with the application.

## Dependencies

The planned scientific core can use NumPy, SciPy, and existing pySTEPS primitives.
The source implementation's package dependencies do not all become dependencies
of the upstream method: extract the required numerical operations together with
their array-based interfaces.

| Dependency | Expected role and current repository support |
| --- | --- |
| NumPy | Arrays, FFT operations, and random streams; already a core pySTEPS dependency |
| SciPy | Distribution functions, interpolation, optimization, and advection sampling; already a core pySTEPS dependency |
| OpenCV | Used through the existing Lucas–Kanade motion method; already listed in the pySTEPS requirements and Conda environments, with availability checked by the motion implementation |
| pyproj and netCDF4 | Relevant to separate geometry and I/O work; already listed in the pySTEPS requirements and Conda environments |

xarray is used for result packaging in the source estimators and I/O. Return
NumPy arrays and ordinary parameter containers from the extracted estimators.
pandas and statsmodels support parameter-table processing and NWP smoothing in
the source workflow; they are not required by the initial fixed-parameter
nowcast engine. MongoDB dependencies remain with the operational application.

No new required package is currently anticipated for the scoped engine.
Additional dependencies or higher minimum versions may be introduced if a
specific implementation need emerges. Document the reason, distinguish required
from optional capabilities, and keep the appropriate package metadata,
requirements files, Conda environments, documentation, and CI configuration
consistent. Verify the engine using the supported pySTEPS environment rather
than relying on packages present only in the source project's environment.

## Supporting contributions and scope boundaries

A transformation object may be useful as a wrapper around existing pySTEPS
transformation functions. The endpoint only requires consistent transformation
and metadata handling; a new public object hierarchy is optional.

CF NetCDF projection handling and interoperability improvements can be separate
I/O contributions. Existing pySTEPS NetCDF support should be extended rather than
duplicated. A validated domain-geometry adapter and example JSON configuration
may also be useful. Neither feature is a prerequisite for passing arrays to the
forecast engine. Geometry adapters must respect pySTEPS raster-boundary and
pixel-centre conventions.

The initial contribution excludes:

- MongoDB, GridFS, field catalogues, and persistent forecast-cycle records.
- Database configuration loading and database-oriented command-line workflows.
- Archive selection, storm catalogues, scheduling, operational reporting, and
  application-specific viewers.
- Radar/NWP parameter blending and lead-dependent statistical parameter inputs.
- Velocity perturbations and externally supplied forecast-state initialization.
- A new operational framework or mandatory database, NetCDF, or JSON input path.

Parameter blending is a subsequent scientific extension once the standalone
forecast engine is established. External projects can retain complete operational
workflows while using the upstream engine for computation.

## Validation and completion criteria

The contribution is complete when a caller can pass three observations and
forecast settings to the public method and obtain a reproducible ensemble without
precomputing parameters or accessing a database.

Validation should establish:

- Component correctness for spectra, marginal fitting, temporal correlations,
  advection, and stochastic generation, with independent references where useful.
- Correct input ordering, output shape, units, and forecast lead-time convention.
- Explicit behaviour for dry and sparse histories, invalid input, and failed or
  constrained parameter estimates.
- Reproducibility with a fixed seed, independent member state and random streams,
  and consistent results when using incremental output.
- Numerical parity with the identified reference implementation using matching
  inputs, defaults, random streams, and state evolution. Document justified
  differences rather than assuming exact equality across numerical backends.
- End-to-end preservation of the evaluated rainfall distributions, spectra, and
  temporal behaviour, using suitable retained cases from the existing evidence.
- Compatibility with the pySTEPS dependency and supported-Python conventions,
  together with the required upstream test checks.

The existing scientific evidence supports the reference defaults. Port validation
establishes that the new implementation preserves that method; it does not impose
a requirement to discover a new calibration or demonstrate superiority in every
region. Material numerical changes require proportionate evaluation.

## Delivery and documentation

Step 2, the reusable parameter estimators and temporal-model utilities, is
implemented in this branch. See the [extraction and validation record](parametric_estimators.md)
for the public helpers, numerical conventions, source provenance, and checks.
Step 3, the calibrated generator and shared marginal mapping, is also implemented;
see its [extraction and validation record](calibrated_generator.md). The complete
forecast engine and executable example are now implemented; see the
[engine extraction and validation record](parametric_engine.md) for the
public call, reference defaults, source comparisons, limitations, and a
separate assessment of the non-parametric inflow extension.

Use this Markdown proposal as the scope and design record for upstream discussion.
Link it from issue #472 or a subsequent contribution discussion when ready.

Deliver the work in reviewable stages:

1. Identify the evaluated reference revision and configuration, and agree the
   public interface and diagnostic conventions.
2. Extract the reusable estimators and temporal-model utilities.
3. Integrate calibrated stochastic generation and its diagnostics.
4. Assemble the three-field forecast engine, preserving reference state evolution
   and adding reproducible random-number control.
5. Complete parity checks, an executable forecast example, and user documentation.

The published pySTEPS documentation currently uses reStructuredText. During
implementation, add API docstrings and reference entries in that format, a gallery
example starting from three rainfall fields, documentation of each reference
default and its override, and manuscript or publication citations as available.
The example should run without the source projects' database infrastructure.

Keep detailed experiment artifacts with their research projects and provide a
concise, revision-specific validation summary in the upstream contribution.
Update this scope document if implementation changes the agreed endpoint or
scientific behaviour.
