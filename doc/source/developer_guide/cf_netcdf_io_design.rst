.. _cf-netcdf-io-design:

CF NetCDF I/O design
====================

Status: agreed objective and acceptance criteria, recorded 5 November 2026.
Stage 1 records the contract and reference evidence; stage 2 implements shared
metadata and the affected numerical helpers. Stages 3 and 4 implement the
physical file importer/exporter and legacy callback adapters. Stage 5 implements
explicit storage encodings and records their measurements. Stages 6 and 7
complete local acceptance and contribution preparation. Implemented scope
and evidence are recorded in :ref:`cf-netcdf-io-stage2`,
:ref:`cf-netcdf-io-stage34`, :ref:`cf-netcdf-io-stage5` and
:ref:`cf-netcdf-io-stage67`.

Objective
---------

Implement a clean, CF-conformant NetCDF importer and exporter for pySTEPS v2
that preserves the meaning of rainfall and reflectivity fields, their spatial
and temporal coordinates, and associated metadata, with efficient storage
options.

The importer and exporter will share one documented dataset contract. Changes
to the internal xarray data model belong within this contribution where they
are needed to support that contract, particularly interval bounds and the
distinction between physical quantities and optional modelling transformations.

Requirements
------------

Physical meaning
    Distinguish instantaneous rain rate and interval mean rain rate (mm/h),
    accumulated rainfall depth (mm), and radar reflectivity (dBZ). Use
    appropriate CF standard names and units. Preserve the distinction between
    a physical quantity, its representation, and any optional modelling
    transformation. Unit normalization must account for numerical scaling when
    equivalent units differ.

Temporal meaning
    Preserve valid times, accumulation start and end times, and forecast
    reference times. Sampling cadence and accumulation duration are distinct.
    An interval mean rate derived from a depth must retain the interval it
    represents. Converting an instantaneous rate into a depth requires an
    explicit integration assumption. Temporal conversions and aggregation must
    update the associated metadata consistently.

Spatial meaning
    Preserve grid coordinates, projection, datum or ellipsoid, and coordinate
    units. CRS metadata must describe the stored coordinates consistently.

Data integrity
    Preserve missing pixels, dry or no-detection values, ensemble identifiers,
    and relevant pySTEPS metadata through an export/import round trip. Missing
    data must remain distinguishable from dry data. Preserve conversion
    parameters, such as Z-R coefficients, where they describe the data or its
    processing.

Efficient storage
    Support compression, appropriate chunking, and optional integer packing
    with explicit precision and range checks. Packing choices are specific to
    the quantity and its units; a scale factor of 0.1 has different meanings
    for mm/h, mm, and dBZ. Packing must preserve missing values and detect
    overflow. Storage encoding is separate from the physical values used in
    memory and from rainfall classification thresholds.

Acceptance criteria
-------------------

* Exported files pass an independent CF validator for the declared CF version.
* An independent CF-aware reader can interpret the physical quantity, spatial
  reference, and temporal support of each supported product.
* Importing an exported file restores the agreed pySTEPS dataset contract,
  including coordinates, interval bounds, forecast reference time, ensemble
  identifiers, missing-data masks, and relevant metadata.
* Numerical differences after a round trip are bounded by the selected
  packing precision, with appropriate floating-point tolerance. A 0.1-unit
  nearest-rounding encoding has an error bound of approximately 0.05 units.
* Tests cover the supported quantities, time semantics, unit normalization,
  transformations, CRS representations, and storage encodings, including
  missing data and packing limits.

Development path
----------------

First specify the shared dataset contract, including variable and coordinate
layout, CF names and units, temporal bounds and cell methods, transformation
handling, metadata preservation, and supported inputs. Select the target CF
version and define compatibility with existing pySTEPS files and APIs.

Then implement the importer and exporter against that contract, with explicit
storage options and executable acceptance examples. Use the agreed contract
and verified examples as the basis for user and API documentation.

.. toctree::
   :maxdepth: 1

   cf_netcdf_io_work_plan
   cf_netcdf_io_contract
   cf_netcdf_io_acceptance
   cf_netcdf_io_baseline
   cf_netcdf_io_stage2
   cf_netcdf_io_stage34
   cf_netcdf_io_stage5
   cf_netcdf_io_stage67
   cf_netcdf_upstream_review

References
----------

* `NetCDF exporter issue #5 <https://github.com/pySTEPS/pysteps/issues/5>`_
* `CF conventions <https://cfconventions.org/>`_
