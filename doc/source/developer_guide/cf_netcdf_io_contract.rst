.. _cf-netcdf-io-contract:

CF NetCDF dataset contract
==========================

Status: stage 1 development proposal, 5 November 2026. The choices below are
concrete defaults for this contribution, ready for review; they do not claim
upstream approval. Shared metadata and numerical helpers are implemented in
:ref:`cf-netcdf-io-stage2`. File I/O and callback adapters are implemented in
:ref:`cf-netcdf-io-stage34`; new storage profiles and full acceptance below
remain the target of :ref:`cf-netcdf-io-work-plan`.

Target and scope
----------------

New exports target CF-1.8, with NETCDF4 storage and the classic numeric types
permitted by that CF version. This is a compatibility target, not a claim that
1.8 is the latest CF release. Pin the standard-name vocabulary used by the
acceptance run; the stage 1 examples use table version 95.

The first implementation supports regular rectilinear projected grids with
one horizontal CRS per dataset, deterministic fields and ensembles, and the
four physical products below. Different physical variables may share a file.
Geographic-only, curvilinear, unstructured, vertical-volume, multiple-CRS and
multi-initialization forecast datasets require explicit extensions; report an
unsupported input clearly without calling a valid external CF file invalid.
Reading never implicitly reprojects or interpolates a field.

Quantities and units
--------------------

.. list-table:: Physical variables
   :header-rows: 1
   :widths: 22 23 35 20

   * - Quantity
     - Suggested variable name
     - CF standard_name
     - Internal / file units
   * - Instantaneous rate
     - precip_intensity
     - lwe_precipitation_rate
     - mm/h / mm h-1
   * - Interval mean rate
     - precip_mean_rate
     - lwe_precipitation_rate
     - mm/h / mm h-1
   * - Accumulated depth
     - precip_accum
     - lwe_thickness_of_precipitation_amount
     - mm / mm
   * - Reflectivity
     - reflectivity
     - equivalent_reflectivity_factor
     - dBZ / dBZ

Names are defaults, not the means of identifying an external quantity. Preserve
unambiguous source variable names; interpret supported CF names, units and
cell methods together.

Interval and cumulative depths are both rainfall depths measured in mm, but
cover different accumulation periods. An interval depth is rainfall accumulated
between the start and end of a particular interval. A cumulative depth is
**rainfall accumulated since a fixed starting time**: its start bound stays
fixed while its end bound advances. This is a common representation of NWP
forecast rainfall, often accumulated from forecast initialization. A suggested
variable name is precip_cumulative.

For example, interval depths of 1, 2 and 3 mm over three successive five-minute
periods correspond to cumulative depths of 1, 3 and 6 mm from the same starting
time. Recover interval depths by subtracting consecutive cumulative totals
that share that origin; account for any accumulation reset explicitly.

Normalize supported unit spellings and scales with a bounded conversion
registry. At minimum accept mm/h, mm h-1 and mm hr-1 for rate, and m s-1 with
the numerical factor 3,600,000 to mm/h. Depth supports mm, cm and m, with
factors 1, 10 and 1,000 to mm. Unknown or dimensionally incompatible units
raise a descriptive error. Mass fluxes require a density assumption and are
not silently treated as depth rates. Linear Z is not labelled dBZ.

Physical values, including threshold and zerovalue when supplied, change
numerically with unit normalization. No optional Pint or UDUNITS runtime
dependency is required for this initial explicit registry.

Variables, dimensions and selection
-----------------------------------

The active field is selected by an explicit variable argument, then a valid
precip_var attribute, then a unique supported physical candidate. Ambiguity is
an error; auxiliary quality flags are not precipitation candidates.

The returned xarray Dataset retains the supported physical variables on the
shared grid and their linked metadata. Its precip_var attribute identifies
the active variable. Normalize that variable's dimensions to (time, y, x) or
(ens_number, time, y, x), retaining singleton time and member dimensions.
Do not invent observation times for undated 2D inputs; require an explicit
time override to normalize them to this contract.

Variables may share a time axis only if their timestamps and interval support
are compatible. Different periods use distinct CF time axes and bounds, as
illustrated by multiple_variables in the reference fixtures. Normalize the
active axis to time; preserve other axes under distinct names, resolving a
name collision with a stable _aux suffix and updating all references. Export
and import must not drop inactive variables merely to activate another one.
Computational routines operate on the selected field. Forecast construction
creates a forecast of that field; it does not imply that the other observed
products have also been forecast.

Preserve ensemble labels, including member zero and nonconsecutive labels.
The file coordinate has standard_name=realization, units=1 and no fill value.
Use int32 for the initial CF-1.8 output profile; check range before casting.

Time and temporal support
-------------------------

Canonical times are UTC numpy datetime64 values, with microsecond resolution
supported. Export float64 seconds relative to a nearby explicit origin with
calendar=proleptic_gregorian, checking that the required precision survives.
Do not strip a timezone without first converting the instant to UTC. Accept
standard/Gregorian modern dates on import when equivalent; non-Gregorian
calendars and leap-second representations need an explicit extension.

Point values have cell_methods="time: point". Intervals use a bounds variable
of shape (time, nv), nv=2, linked by time:bounds. Variable-specific time-axis
names replace time in the corresponding cell_methods string. Interval times
are labelled by their end, with bounds containing start and end. On importing
a midpoint-labelled interval, preserve its source time as an auxiliary
coordinate and normalize the computational valid time to the interval end.

Mean rates use time: mean; accumulated depths use time: sum. Bounds are
required for interval products. Durations are positive. A zero-length initial
cumulative total is allowed only at its origin with zero valid amounts; it
cannot be divided by its duration. Shared bounds do not make an explicitly
point-valued variable an interval statistic.

Sampling cadence is derived from time differences. Keep time.stepsize as a
seconds-valued compatibility cache only for regular axes with known spacing.
Do not infer it from a singleton or from accumulation duration. Irregular
selections such as leads 1, 2 and 4 are valid I/O; a nowcast algorithm may
separately require regular input spacing.

Bounds are authoritative. A scalar accutime in minutes may be supplied as a
compatibility cache only when all intervals have that duration. Inconsistent
caches raise an error. Preserve variable durations without forcing a scalar.

For a depth H over duration dt seconds, the corresponding interval mean is
H * 3600 / dt in mm/h, with the same bounds and time: mean. The inverse
conversion uses the same interval. An instantaneous rate cannot yield a depth
without an explicitly requested integration rule and support interval.
Aggregation combines nonoverlapping intervals and updates bounds, timestamps,
duration caches and metadata. Cumulative totals are differenced before
interval aggregation; never sum running totals or overlapping windows as
though they were disjoint depths. Missing coverage remains missing by default.

Keep forecast_reference_time as a scalar coordinate with its CF standard name.
It is independent of the file's numeric time origin and accumulation start.
Construct forecast bounds from the requested product, rather than copying
observation bounds or using gaps between selected leads as durations.

Reflectivity and transformations
--------------------------------

dBZ is already a logarithmic physical representation. Its correct
interpretation does not require transform="dB". Negative and zero dBZ can be
valid detected values, and zero dBZ is not a universal no-rain marker.

For Z-R conversion use Z=10**(dBZ/10) followed by R=(Z/a)**(1/b). Record the
actual coefficients and their source when a conversion is performed. Reading
or writing reflectivity does not implicitly apply a Z-R relation. Existing
conversion defaults may be retained for compatibility, but their use must be
recorded rather than represented as source-supplied coefficients.

Optional statistical transforms, including log rain, Box-Cox and normal
scores, are separate from the physical quantity. Strict physical export
rejects any selected transformed field until it has explicitly been returned to
physical values; no automatic inverse transform is hidden in writing.
Treat an absent transform and transform=None as the untransformed state.
Legacy dBZ plus transform="dB" is recognized once as the built-in dBZ
representation, not as an additional transform to apply twice.

Geometry and CRS
----------------

x and y contain pixel centres, with float64 coordinates and units m after
normalization. x is increasing; retain either increasing or decreasing y
without changing its relation to the array. For a regular grid derive outer
edges using half a cell. A singleton axis needs explicit bounds or spacing.
Use a numerical spacing tolerance rather than exact floating-point equality.

Normalize coordinate and grid-mapping linear units consistently. The scalar
CRS variable has grid_mapping_name, complete applicable CF attributes and
crs_wkt from pyproj. Preserve the datum/ellipsoid, sphere radius, false offsets
and coordinate-system units. Conflicting WKT, CF parameters or coordinate
metadata are errors, not occasions to assume WGS84. A legacy projection
attribute may be retained as a validated cache, not an independent authority.

Projected CF-1.8 data with a valid grid mapping need not store redundant
two-dimensional lat/lon arrays. Default to this compact representation; offer
an explicit option to include geographic auxiliary coordinates. If included,
compute them in the projection's geographic CRS and link them correctly.
Coordinates and CRS are stored once, without time or member duplication.

Missing values and metadata
---------------------------

Canonical missing physical values are NaN. Decode _FillValue and missing_value
in their packed domain before applying scale and offset, exactly once. Reject
infinities. Finite negative physical rain rates and depths are invalid; finite
negative dBZ is not. Dry, trace, no-detection and missing are distinct states.
Preserve a supplied no-detection flag or declared finite floor as source
metadata; do not infer it from a field minimum. Auxiliary flags use
ancillary_variables, flag_values/flag_masks and flag_meanings as appropriate.

Threshold and zerovalue are optional I/O metadata. Reading a scientifically
identified CF field must not require inventing model-specific rain/no-rain
parameters. A separate model-readiness check can require explicit values for
the selected algorithm; coordinate that distinction with the dataset validator.

Preserve relevant serializable source metadata, quality variables, institution,
source, references and processing history. Required scientific identity comes
from CF metadata rather than opaque JSON. Reserved attributes, conflicting
metadata, unsupported object-valued attributes and dangling links produce
clear errors. Omit optional None-valued attributes deliberately; do not turn
them into the string "None". Record supplied overrides in processing history.

Public interfaces and compatibility
-----------------------------------

The proposed dataset-first interfaces are::

    import_netcdf_cf(filename, *, variable=None, group=None,
                     dtype="double", load=True, legacy=False, metadata=None)
    export_netcdf_cf(dataset, filename, *, variables=None, encoding=None,
                     include_latlon=False, overwrite=False)

Register netcdf_cf as an importer through the existing I/O interface. Import
returns a Dataset, never an array/metadata tuple. It raises descriptive errors
by default. Eager loading is the default so the file is closed before return;
load=False returns a lazy Dataset owned and closed by the caller. Preserve
source dtype/encoding metadata separately from the requested floating dtype.
The optional group selects one explicitly named NetCDF group; no recursive
group merge is performed. metadata supplies missing supported metadata; it
cannot silently contradict an existing scientific definition.

For export, variables=None writes all supported physical variables and their
linked coordinates and ancillary data. Explicit variable subsets retain only
their relevant dependencies and must leave no stale active-variable selector
or metadata links.

Keep import_netcdf_pysteps as a compatibility wrapper with its documented
onerror and dtype/fillna behaviour. A finite fillna view must retain a validity
mask so it cannot later be exported as observed dry rainfall. New core imports
keep NaNs. Keep existing exporter initialization, write and close entry points
as adapters over the shared schema, including None, timestep and member write
modes. Existing explicit datatype, fill, scale, offset and compression options
retain their meanings. Correct the packing formula documentation.

Strict input interprets supported CF semantics. legacy=True permits documented
pySTEPS variable-name and projection fallbacks. It still requires a supplied
accumulation interval or explicit duration/end-time convention when that
information is absent; never derive an interval from cadence alone. Likewise,
do not fabricate a CRS, observation time, threshold, dry value or forecast
origin. Readable but scientifically incomplete legacy files receive a
descriptive request for the missing override. Tuple-returning v1 APIs are
outside this v2 contribution.

The shared schema must drive both bulk and incremental writes. Preserve the
existing callback workflow; use region writes through netCDF4 for incremental
NetCDF output rather than treating xarray mode="a" as an append API. Validate
the schema before creating a file and each batch before writing. Close handles
on failure; identify incomplete output and never present it as a complete
forecast. Bulk output protects existing files unless overwrite is explicit.

Storage and dependencies
------------------------

The new bulk API preserves the input floating-point precision by default,
using zlib level 4 with shuffle and no quantization. Coordinate values remain
float64. Initial candidate chunks isolate a member and time step and tile
space up to 256 by 256 cells; stage 5 measurements may revise this documented
candidate before it becomes a production default. The legacy API retains its
explicit defaults during adaptation.

Packing is an opt-in per-variable encoding. Signed int16 rain-rate storage at
0.1 mm/h uses scale_factor=0.1, add_offset=0.0 and _FillValue=-32768. The
nonnegative decoded range is 0 to 3276.7 mm/h. Apply nearest rounding with ties
to even; validate physical values, rounded integer range and fill collisions
before casting. Do not wrap or silently clip. Precision for depth and dBZ is
selected in their own units, not inherited from a rate profile. Recompute
encoding after a quantity/unit conversion instead of blindly copying an
unrelated source packing scale. The error bound is half a quantization step
plus an explicit floating-point tolerance.

Use xarray, numpy, netCDF4 and pyproj already present in the v2 environment.
Declare actual runtime dependencies in package metadata; CRS.cs_to_cf requires
pyproj >=3.0. Xarray >=2025.1.2 is required for explicit microsecond decoding
through its public CFDatetimeCoder. No regridding dependency is introduced. The independent checker
and documentation tools are development dependencies only. Verify compatible
lower dependency versions during implementation rather than assuming the
stage 1 environment is a supported minimum.

Evidence and integration points
-------------------------------

See :ref:`cf-netcdf-io-acceptance` for the executable reference cases and planned
acceptance matrix, and :ref:`cf-netcdf-io-baseline` for baseline evidence.
Coordinate the shared validation entry point with draft PR 594, CRS handling
with PR 596, and the existing xarray/exporter prototype. Do not introduce a
second competing dataset validator or copy the old projection parser.

References
----------

* `CF-1.8 <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html>`_
* `Standard names v95 <https://cfconventions.org/Data/cf-standard-names/95/build/cf-standard-name-table.html>`_
* `pyproj CF conversion <https://pyproj4.github.io/pyproj/stable/build_crs_cf.html>`_
* `Xarray NetCDF I/O <https://docs.xarray.dev/en/stable/user-guide/io.html>`_
