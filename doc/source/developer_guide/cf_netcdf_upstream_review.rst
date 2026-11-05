.. _cf-netcdf-upstream-review:

CF NetCDF contribution review draft
===================================

Suggested title: Preserve physical rainfall semantics in shared CF NetCDF I/O

Proposed description
--------------------

Rainfall rate, interval depth and cumulative forecast totals need distinct
support metadata even when they use the same grid. Existing I/O paths can
lose that distinction, projection details or missing pixels when units and
integer packing are handled separately. This contribution adds shared dataset
normalization and physical CF-1.8 import/export, including explicit bounds,
full CF/WKT projection metadata and opt-in int16 packing.

For example, a cumulative NWP total retains its fixed starting time instead
of being interpreted as a sequence of independent interval depths. A 0.1 mm/h
int16 profile preserves missing cells as reserved raw codes, preserves observed
dry zero and rejects overflow or fill collisions before publication. Existing
forecast callback APIs use the same schema with incremental region writes.

The contribution includes a user guide, self-contained examples for every
quantity and an ensemble, compatibility/release notes, independent fixtures,
CF checker reports, Iris interoperability evidence and measured storage
tradeoffs. It addresses the scope discussed in issue #5; it does not claim to
support every valid CF geometry or calendar.

Suggested review order
-----------------------

1. Dataset contract and shared metadata helpers: quantities, units, bounds,
   cumulative semantics, CRS, conversion and temporal aggregation changes.
2. Core importer/exporter and registry integration: physical validation,
   source ownership, legacy metadata overrides and atomic publication.
3. Callback adapter and storage: shared schema, direct region writes,
   compression/chunk options, int16 quantization and raw missing-code handling.
4. Acceptance fixtures, failure tests, external-reader evidence, benchmarks
   and user documentation.

Validation for the description
-------------------------------

* 481 scoped regression tests passed, with known MRMS/ANVIL warnings; 80
  storage cases also passed with minimum xarray 2025.1.2.
* Independent Iris 3.16.2 checks passed for 62 floating/packed files, including
  all supported products, both y orientations, multiple variables, ensembles,
  projection examples and a radar-to-nowcast-to-file round trip.
* All 12 user example files passed unmodified strict CF-1.8 validation. The
  earlier 50 production exports pass with the documented Oblique Mercator
  checker-schema correction; unmodified results are retained as well.
* On the two measured workloads, int16/zlib4 files were 72.2% and 63.7%
  smaller than float32/zlib4. Timings use a warm OS cache and are not universal
  performance claims.

Review considerations
----------------------

The shared validator and numerical helper changes are part of the scientific
contract, so review them alongside the file APIs. The branch is based on
upstream revision ``f1a3af117a7085bef206a444edaa745b2431a9e7``. Reconcile any
subsequent validator or importer changes before merging; this local packet
does not assert the current state of other upstream proposals. Runtime
requirements become xarray >= 2025.1.2 and pyproj >= 3.0, with netCDF4 required
only for NetCDF use and dask for lazy import.

Explicit compatibility changes and migration examples are in
:ref:`cf-netcdf-user-guide`. The complete upstream suite and documentation site
build have not been run. Independent datum-shift validation, geographic-only,
curvilinear, unstructured, volume radar, multiple-CRS and multiple-initialization
products remain outside this implementation. No forecast-skill claim is made.

This is a local submission draft. No commits, push, pull request or messages to
maintainers were made as part of stages 6 and 7.
