.. _cf-netcdf-io-acceptance:

CF NetCDF acceptance matrix
===========================

Status: local implementation acceptance completed through stage 6.
Reference fixtures below remain independent of production I/O. Combined
production, checker and independent-reader evidence is recorded in
:ref:`cf-netcdf-io-stage67`; checker success alone is not a scientific oracle.

Reference cases
---------------

The editable fixtures live in pysteps/tests/cf_netcdf_reference.py. They use
hand-specified arrays and CF metadata with xarray serialization, without
calling pySTEPS I/O, conversions or xarray_helpers. Generate the files with::

    python -m pysteps.tests.cf_netcdf_reference --output /tmp/cf-reference
    python -m pytest -q pysteps/tests/test_cf_netcdf_reference.py

Generated NetCDF binaries are temporary artifacts. inventory.json records file
sizes and SHA-256 hashes for that generation; binary hashes are not a portable
cross-version expectation. Scientific expectations reside in source and tests.

.. list-table:: Implemented reference fixtures
   :header-rows: 1
   :widths: 28 42 30

   * - Case ID
     - Scientific expectation
     - Verified I/O expectation
   * - instantaneous_rate
     - Point rates; dry zero, trace values and NaN remain distinct.
     - Correct CF identity, units and mask; no invented duration.
   * - mean_rate
     - 450-second sampling with 600-second, overlapping support.
     - Preserve bounds; do not substitute cadence or sum overlapping windows.
   * - interval_depth
     - Durations 450, 450 and 900 seconds; depths 1.5, 1.5 and 3 mm.
     - Derive 12 mm/h interval means; retain variable durations.
   * - cumulative_depth
     - Fixed origin, totals 1.5, 3 and 6 mm at sparse forecast leads.
     - Difference cumulative amounts; keep origin distinct from duration.
   * - reflectivity
     - Detected -10 and 0 dBZ, 20 dBZ, a flagged floor, and missing data.
     - No required transform marker; preserve detection flag and NaN.
   * - ensemble_rate
     - Belgian LCC grid, ascending y, member identifiers 0 and 7.
     - Keep actual member labels, dimension order and CRS.
   * - multiple_variables
     - Point rate and interval depth have different time axes and bounds.
     - Explicit active selection; preserve all fields and update axis links.
   * - unit_alias_rate
     - Rate values in m s-1 equal the mm/h fixture after scaling by 3,600,000.
     - Scale numerically; normalize units, thresholds and applicable encoding.
   * - all_missing
     - Every rate pixel is missing, with complete coordinates.
     - No inference of dry values, thresholds or zerovalue from minima.
   * - polar_sphere
     - Stereographic CRS uses a sphere of radius 6,371,000 m.
     - Preserve radius; do not reconstruct a default WGS84 ellipsoid.
   * - albers
     - Albers CRS retains its parallels, ellipsoid and origin.
     - Use shared CRS conversion rather than a projection-specific parser.
   * - projected_without_latlon
     - NZTM coordinates and full CRS without redundant geographic arrays.
     - Valid compact CF-1.8 output with recoverable geolocation.

Tests also check raw NetCDF coordinate types, no coordinate fill values, scalar
CRS storage, exact time semantics, numerical values and missing masks.
No member or time axis may be squeezed away during import.

Implementation matrix
---------------------

The following dimensions are implemented using focused parameterizations,
not a full Cartesian product. Coverage locations are listed below.

.. list-table:: Acceptance dimensions
   :header-rows: 1
   :widths: 20 48 32

   * - Dimension
     - Required cases
     - Acceptance rule
   * - Selection
     - Unique variable; explicit selection; valid/invalid selector; ambiguity.
     - Never choose the first candidate silently or drop inactive data.
   * - Time
     - Singleton; fractional minute; subsecond; cadence not duration; sparse
       leads; midpoint-labelled intervals; separate temporal axes.
     - Preserve instants and support; require cadence only where needed.
   * - Accumulation
     - Interval; overlapping rolling window; cumulative; reset; initial zero.
     - No summing overlap or running totals; detect undefined conversions.
   * - Conversion
     - Unit aliases and scales; depth/mean-rate inverse; explicit point
       integration; dBZ with and without legacy transform metadata.
     - Correct physical values, bounds, labels, thresholds and provenance.
   * - Geometry
     - EPSG:2193, EPSG:3812, polar sphere and Albers; metre/kilometre units;
       both y orientations; singleton axis; optional geographic coordinates.
     - Equivalent CRS, unchanged physical cell centres and array alignment.
   * - Metadata errors
     - Conflicting WKT/CF; invalid units; dangling links; wrong bounds;
       inconsistent accutime; missing CRS or observation time.
     - Clear failure without fabricated metadata or partially valid output.
   * - Missing/detection
     - Dry; trace; missing pixels/whole field; missing_value and _FillValue;
       valid negative dBZ; explicit no-detection flags; infinities.
     - Preserve states; mask before scaling; reject unsupported infinities.
   * - Encoding
     - float32/float64; int16 profiles; both signs for dBZ; near-half steps;
       minimum/maximum codes; fill collisions; overflow; unit conversion.
     - No double scaling, clipping or wrapping; declared error bound holds.
   * - Write modes
     - Bulk; timestep; member; close/failure; overwrite; large arrays.
     - Same shared schema; explicit incomplete state; no repeated full rewrite.
   * - Ownership
     - Eager and lazy import; success/error close; explicit group.
     - Eager files close; lazy ownership documented; no leaked handles.
   * - Legacy
     - Old pySTEPS variable names, projection and encoding options; missing
       scientific metadata; explicit overrides; dtype/fillna compatibility.
     - Follow the documented compatibility policy; log supplied overrides.
   * - Independent checks
     - External CF fixture to importer; exporter to independent reader and CF
       checker; complete nowcast-to-file-to-import example.
     - Agreement in scientific meaning, masks and numeric tolerances.

Validation protocol
-------------------

Run an independent checker with an explicit CF version and vocabulary. The
stage 1 run uses compliance-checker 6.1.0, CF-1.8, strict criteria, standard
name table 95, with no skipped checks::

    CF_STANDARD_NAME_TABLE=/path/to/cf-standard-name-table-v95.xml \
      compliance-checker -t cf:1.8 -c strict -f json_new \
      -o /tmp/cf-checks.json /tmp/cf-reference/*.nc

Keep checker versions, vocabulary hash, per-file results and fixture hashes
with each acceptance run. Checker success complements the numerical oracles;
it does not prove correct accumulation interpretation or forecast skill.
Stage 6 adds independent Iris interpretation of production exports; see
:ref:`cf-netcdf-io-stage67`. Stage 1 used xarray and raw netCDF4 inspections only.

Coverage locations
------------------

All test paths below are relative to ``pysteps/tests``. Historical run results
and commands are retained in the stage 2, 3/4 and 5 reports; new independent
reader and example results are retained in the stage 6/7 report.

.. list-table:: Implementation evidence
   :header-rows: 1
   :widths: 22 78

   * - Matrix dimension
     - Focused tests and executable evidence
   * - Selection and metadata errors
     - ``test_dataset_metadata.py``: second-axis selection, contradictory
       metadata and unsupported definitions; ``test_io_netcdf_cf.py``:
       variable subsets, missing overrides and dangling-link rejection.
   * - Time and accumulation
     - ``test_dataset_metadata.py``: midpoint labels, fractional/day duration,
       sparse leads, cumulative differencing/reset rejection and initial zero,
       overlapping means, partial coverage and shared axes.
       ``test_io_netcdf_cf.py``: midpoint/subsecond, singleton coordinates and
       time-bound serialization. Independent cumulative/mean/depth fixtures.
   * - Conversion
     - ``test_dataset_metadata.py``: unit scaling, duration-aware inverse,
       explicit point integration, reflectivity floors and legacy markers;
       ``test_utils_conversion.py`` and ``test_utils_dimension.py`` regression.
   * - Geometry
     - ``test_dataset_metadata.py``: kilometre offsets, sphere, singleton,
       geolocation and alignment; ``test_io_netcdf_cf.py``: NZTM/UTM/Albers,
       optional lat/lon; Iris geolocation checks on both y orientations.
   * - Missing/detection and encoding
     - ``test_io_netcdf_storage.py``: mask-before-scale, float/int16, offsets,
       every usable int16 code, ties/range/fill collisions, ancillary flags,
       all-dry/all-missing, separate quantity precision and normalized units.
       ``test_io_netcdf_cf.py``: independent packed inputs and invalid values.
   * - Write modes and ownership
     - ``test_io_netcdf_cf.py``: bulk/timestep/member schema, incomplete close,
       failure cleanup, concurrent publication, overwrite, groups, lazy/eager
       ownership; ``test_io_netcdf_storage.py``: masked callback arrays and
       packing/filter controls. Multi-megabyte storage benchmark workloads.
   * - Legacy
     - ``test_io_netcdf_cf.py``, ``test_io_nowcast_importers.py`` and
       ``test_exporters.py``: old names, explicit repairs, dtype/fillna,
       error modes and callback adapters.
   * - Independent checks and integration
     - ``cf_netcdf_stage34_acceptance.py`` and
       ``cf_netcdf_storage_acceptance.py``: independently constructed inputs,
       source-based numerical oracles and archive nowcast pipeline.
       ``cf_netcdf_independent_acceptance.py``: Iris interpretation.
       User guide examples: all quantities and ensemble, float and packed.
