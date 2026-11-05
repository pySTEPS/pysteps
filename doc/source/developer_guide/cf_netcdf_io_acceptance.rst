.. _cf-netcdf-io-acceptance:

CF NetCDF acceptance matrix
===========================

Status: stage 1 reference semantics and future implementation checks.
Passing the fixture checks or validating these independently defined files
does not establish compliance of the production importer or exporter.

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
     - Future I/O acceptance
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

The following rows are required implementation checks, not claims of existing
coverage. Add focused parameterizations rather than a full Cartesian product.

.. list-table:: Planned acceptance dimensions
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
Stage 6 must additionally validate the implemented exporter and an independent
CF-aware reader; stage 1 uses xarray and raw netCDF4 inspections only.
