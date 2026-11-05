.. _cf-netcdf-io-work-plan:

CF NetCDF I/O work plan
=======================

Status: stages 1 through 7 completed in the contribution worktree, 5 November 2026.
Stage 1 deliverables are recorded in :ref:`cf-netcdf-io-contract`,
:ref:`cf-netcdf-io-acceptance` and :ref:`cf-netcdf-io-baseline`. The shared
metadata implementation and validation evidence are recorded in
:ref:`cf-netcdf-io-stage2`. The file APIs and callback adapters are recorded in
:ref:`cf-netcdf-io-stage34`. Storage encodings and measurements are recorded in
:ref:`cf-netcdf-io-stage5`. Independent acceptance and user/upstream documentation are recorded in
:ref:`cf-netcdf-io-stage67`. Remote submission is a separate step.
The agreed objective and acceptance criteria are in :ref:`cf-netcdf-io-design`.

Each stage includes tests and documentation for its own behaviour. The final
acceptance stage brings those checks together with independent readers,
validation tools, and representative files.

1. Establish the baseline and dataset contract
----------------------------------------------

Record the upstream revision, environment, existing I/O and conversion test
results, and small reproductions of the relevant failures. Review related
upstream work to identify integration points before proposing API changes.

Specify and review:

* Supported quantities: instantaneous rate, interval mean rate, accumulated
  depth, and reflectivity, including CF names and canonical internal units.
* Variable selection, dimensions, coordinate layout, and ensemble identifiers.
  Decide whether and how multiple physical variables share a file, including
  variables with different temporal support.
* Valid time, interval bounds, forecast reference time, sampling cadence, and
  the representation of interval and cumulative forecast accumulations.
* Physical representations, optional modelling transformations, thresholds,
  no-detection values, missing values, and conversion parameters.
* Coordinate units, pixel centres and edges, CRS representation, and the
  initial set of supported grid types.
* Target CF version, dependency requirements, public API, resource ownership,
  bulk and incremental writing, and compatibility with existing files and APIs.
* Default storage behaviour and explicit options for precision, packing,
  compression, and chunking.

Deliverable: a documented contract and acceptance matrix, with small reference
datasets for each supported quantity and temporal meaning. Open choices above
become explicit decisions before the corresponding implementation begins.

2. Implement shared metadata handling and validation
----------------------------------------------------

Build the common interpretation and validation used by both I/O directions.
Normalize supported equivalent units with numerical scaling, construct valid
CF metadata, and use pyproj to preserve the CRS and coordinate-system units.
Check grid geometry, metadata consistency, and interval bounds explicitly.

Update the affected conversion, aggregation, and forecast-output helpers to
respect the contract. In particular, preserve temporal support through
depth/rate conversion, update it after aggregation, distinguish dBZ from a
generic modelling transformation, and construct appropriate forecast bounds.

Deliverable: shared helpers and regression tests demonstrating correct
quantities, coordinates, times, missing-data semantics, and transformation
handling. A depth-to-mean-rate conversion must use the actual accumulation
duration rather than the sampling cadence.

3. Implement the CF NetCDF importer
-----------------------------------

Read the supported CF representations into the agreed xarray dataset contract.
Decode packing, missing values, times, interval bounds, ensemble identifiers,
and CRS metadata; identify the requested variable and normalize its units.
Derive interval durations from bounds and check any legacy duration metadata
against them. Preserve agreed provenance and pySTEPS metadata.

Provide explicit handling for ambiguous or unsupported inputs and the agreed
legacy-file policy. Implement the documented dtype and missing-value options,
and make lazy loading and file ownership clear.

Deliverable: import tests using independently constructed CF files, including
equivalent unit spellings, dBZ without a pySTEPS transformation marker, and
accumulations defined by bounds without an accutime attribute.

4. Implement the CF NetCDF exporter
-----------------------------------

Validate the dataset before writing and serialize the shared contract,
including physical units, cell methods, time bounds, forecast reference time,
ensemble identifiers, full CRS information, and relevant metadata. Define how
modelling transformations are handled so the written values agree with their
physical labels. Keep internal attributes serializable and CF links valid.

Implement the agreed public API and write modes, including incremental
behaviour where supported. Preserve the metadata contract across write modes
and avoid repeated complete-file rewrites for streaming ensembles.

Deliverable: floating-point reference exports for each supported quantity,
with independent checks of dimensions, coordinates, masks, metadata links,
and physical values. Run the selected CF validator on these reference files.

5. Add and assess storage encodings
-----------------------------------

Implement optional packing profiles, including signed 16-bit storage at
0.1 mm/h for rain rate. Specify precision separately for depth and
reflectivity. Reserve a fill value, check the rounded packed range and fill
collisions before casting, and keep missing pixels distinct from dry or
no-detection values. Document the treatment of other non-finite values.

Add explicit compression and chunking options appropriate to the supported
read and write patterns. Check raw on-disk values as well as decoded values
to detect double scaling and incorrect fill handling.

Deliverable: encoding tests covering quantization bounds, traces, dry fields,
all-missing fields, valid negative dBZ, packing boundaries, and overflow.
Compare file size, write time, and representative read time for floating-point
and packed files using the same inputs and a recorded environment. Select
defaults from measured tradeoffs without imposing machine-specific CI timing
thresholds.

6. Complete independent acceptance and pipeline integration
-----------------------------------------------------------

Exercise the acceptance matrix across quantities, grids, temporal support,
ensembles, and encodings. Include fractional accumulation periods, cadence
different from duration, irregularly selected forecast leads, cumulative
accumulations, ascending and descending y coordinates, and missing pixels.

Validate exported files with a versioned independent CF checker and inspect
them through an independent CF-aware reader. Verify both imports from
independent files and exports consumed outside pySTEPS. Reimport exported
files and compare semantics and numerical values with explicit tolerances.

Run a supported input-to-nowcast-to-NetCDF-to-import workflow to check that
the I/O contract integrates with pySTEPS. Record fixture provenance, commands,
tool versions, checker results, and any intentionally unsupported cases.

Deliverable: reproducible acceptance examples, relevant regression-suite
results, independent validation evidence, and documented scope limits.

7. Prepare user documentation and the upstream contribution
-----------------------------------------------------------

Document the dataset contract, API, unit normalization, accumulation
conventions, dBZ handling, storage profiles, supported inputs, and compatibility
policy. Provide runnable examples for each quantity and an ensemble forecast,
plus release notes for changes in behaviour.

Deliverable: reviewable changes organized around the contract and shared
helpers, importer/exporter integration, and storage options. Include the
acceptance evidence and performance measurements needed for upstream review.

Execution order
---------------

Complete and review stage 1 first. Stage 2 provides the common foundation for
stages 3 and 4; complete a reliable floating-point import/export path before
adding packing. Develop independent acceptance fixtures and checker coverage
alongside these stages, then complete the combined acceptance in stage 6 and
the documentation and contribution preparation in stage 7.
