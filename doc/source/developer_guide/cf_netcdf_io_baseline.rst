.. _cf-netcdf-io-baseline:

CF NetCDF stage 1 baseline
==========================

Recorded 5 November 2026 in the cf-netcdf-io worktree. The source baseline is
upstream main f1a3af117a7085bef206a444edaa745b2431a9e7, dated 3 November
2026. No production importer, exporter or conversion code was changed during
stage 1. Reference fixtures and diagnostic helpers are development artifacts.

Environment and existing tests
------------------------------

The dedicated Conda environment uses Python 3.13.18, numpy 2.5.4, xarray
2026.9.0, netCDF4 1.7.4, pyproj 3.8.0 and pytest 9.2.0. These are observed
versions, not a proposed minimum dependency set. The installed source resolves
to this contribution worktree.

The existing scoped baseline passed: 73 tests, with 72 warnings about imported
grid pixel sizes disagreeing with coordinate extents. These tests use the
existing local sample archives. The independent reference cases are synthetic
and do not need an archive or database. Reproduce the scoped baseline with::

    python -m pytest -q \
      pysteps/tests/test_utils_conversion.py \
      pysteps/tests/test_utils_transformation.py \
      pysteps/tests/test_utils_dimension.py \
      pysteps/tests/test_exporters.py \
      pysteps/tests/test_io_nowcast_importers.py

Numerical and metadata probes
-----------------------------

Run the diagnostic separately from regression tests::

    python -m pysteps.tests.cf_netcdf_baseline --output /tmp/cf-baseline.json

The recorded observations are in :download:`cf_netcdf_stage1_baseline.json`.
The diagnostic records intended and observed behaviour; it does not lock the
current bugs into tests or claim they have been fixed.

* Aggregating two five-minute 1 mm depths retains accutime=5 and yields an
  incorrect mean rate of 24 mm/h rather than 12 mm/h.
* 20 dBZ converts to 0.237137 mm/h without a transform marker, but to the
  expected 0.648420 mm/h when the legacy dB marker is present.
* transform=None is rejected by the conversion routine and cannot be directly
  serialized by xarray from the current helper's output.
* The valid unit spelling mm h-1 fails the current unit-conversion dispatch.
* An accumulation with CF bounds but no accutime fails conversion.
* Forecast output drops time bounds while leaving a dangling bounds link.
* The current helper fails for an EPSG:2193 CRS.
* Packed int16 export turns the tested missing input pixel into zero rainfall.
* Current import ignores the requested single dtype and finite fillna value.

Independent reference validation
--------------------------------

All 18 reference tests passed. The final combined run passed all 91 scoped
tests, with the same 72 pre-existing grid-spacing warnings.

The 12 source-defined reference cases and their semantic tests are described
in :ref:`cf-netcdf-io-acceptance`. They are serialized independently of the
production exporter. Check the reference tests with::

    python -m pytest -q pysteps/tests/test_cf_netcdf_reference.py

The validation toolchain is isolated from the Conda environment: an ephemeral
venv with system-site-packages uses compliance-checker 6.1.0, cf-units 3.3.1
and docutils 0.22.4. The vocabulary is standard-name table 95. Run the checker
command in the acceptance matrix with strict criteria and no skipped checks.
All 12 files passed strict checking, with every scored point satisfied and
no skipped checks. The retained machine-readable summary, vocabulary hash
and generated file inventory are in :download:`cf_netcdf_stage1_validation.json`.

Reference development exposed two useful implementation requirements: CF-1.8
does not permit the int64 time/member encodings initially tried, and pyproj
3.8.0 to_cf omits latitude_of_projection_origin for the polar stereographic
variant B fixture. The reference explicitly includes its known 90-degree
origin. Shared production metadata handling must complete and validate the
required CF attributes rather than treating pyproj output as certification.

The reference checks use xarray and raw netCDF4 reads plus an independent CF
checker. A second independent CF-aware high-level reader, production exporter
compliance, full pipeline integration and performance measurements remain
stage 6/5 acceptance work. No forecast skill is evaluated here.

Upstream integration snapshot
-----------------------------

Live GitHub metadata was read on 5 November 2026. No upstream messages or
changes were posted.

* `Issue 5 <https://github.com/pySTEPS/pysteps/issues/5>`_ remains open.
* `PR 594 <https://github.com/pySTEPS/pysteps/pull/594>`_ is an open draft
  dataset validator, updated 4 November. Integrate the shared contract there
  rather than adding a competing interpretation of a valid dataset.
* `PR 596 <https://github.com/pySTEPS/pysteps/pull/596>`_ is open, updated
  4 November, and uses rioxarray/pyproj for the FMI GeoTIFF import path.
  Share CRS handling where appropriate; no regridding dependency is needed.
* The xarray/exporter branch ends at ad929731b54cc1c6bb5b1532402e5814255cd53e
  from 25 July 2024. It is an existing prototype, not a current complete
  implementation to transplant. Preserve the public callback workflow while
  replacing duplicated schema construction.

Stage 1 disposition
-------------------

The concrete development proposal is :ref:`cf-netcdf-io-contract`.
The contract, reference cases, independent fixture checks, acceptance matrix,
baseline diagnostics and upstream integration points complete the stage 1
deliverables. The proposal remains subject to review; stages 2 through 7 and
their production acceptance criteria are not claimed complete.
