.. _cf-netcdf-io-stage67:

CF NetCDF stages 6 and 7
========================

Completed in the contribution worktree on 5 November 2026. Stage 6 reconciles
the implementation matrix and adds a separate CF-aware reader. Stage 7 adds
:ref:`cf-netcdf-user-guide`, runnable examples, compatibility/release notes and
:ref:`cf-netcdf-upstream-review`. No remote publication or upstream approval is
implied.

Independent acceptance
----------------------

Iris 3.16.2 independently loaded 62 files: the 26 floating production exports
from stages 3/4, 24 packed exports from stage 5, and 12 newly executed user
examples. Every physical field was checked for identity, units, shape, exact
missing masks, decoded values, dimension association, time points/bounds,
forecast reference time, realization labels, cell methods and projection.
Iris projection parameters were converted through Cartopy and compared to the
declared CRS at grid corners within 1e-7 degrees, without a datum shift.
Values agreed with independently decoded netCDF4 data within 1e-10 physical
units; earlier hand-defined source oracles separately establish the packing
error bounds. Iris does not call pySTEPS or xarray in this check.

Iris records a handled warning for fixed-origin cumulative bounds: these
cannot form its strictly monotonic DimCoord bounds, so it retains time as an
AuxCoord on the same data dimension. The checks verify all common-origin
bounds rather than requiring a particular reader coordinate class. Iris's
current default datum behaviour is selected explicitly: ellipsoid/projection
geometry is checked, but datum transformations are not independently tested.
Enabling its future datum support produced an unrecognized long-form NZGD2000
datum in Cartopy/PROJ. Warnings and load-problem details are retained, not hidden.

The 12 new example files also pass unmodified compliance-checker 6.1.0,
CF-1.8 strict, standard-name table 95. Earlier results are retained: the 26
floating and 24 packed files pass with the documented checker schema correction;
the unmodified checker rejects one Swiss Oblique Mercator file in each set
because it expects ``azimuth`` instead of CF's ``azimuth_of_central_line``.
Both unmodified and corrected reports remain in the stage 3/4 and 5 evidence.
No output metadata or skipped checks were used to work around that defect.

Reproduction
------------

Generate production files using the existing acceptance modules (the archive
workflow requires the configured MCH sample data)::

    python -m pysteps.tests.cf_netcdf_stage34_acceptance --output /tmp/cf-float --with-archive
    python -m pysteps.tests.cf_netcdf_storage_acceptance --output /tmp/cf-packed --with-archive
    python doc/source/user_guide/cf_netcdf_examples.py --output /tmp/cf-examples

Install ``scitools-iris==3.16.2`` in a separate acceptance environment, then::

    python -m pysteps.tests.cf_netcdf_independent_acceptance \
      --output /tmp/iris-validation.json \
      /tmp/cf-float /tmp/cf-packed /tmp/cf-examples

Run the independent checker as specified in :ref:`cf-netcdf-io-acceptance`.
The retained evidence is :download:`cf_netcdf_stage67_validation.json`, including
file hashes, reader/checker versions, per-field results, warnings and code
hashes. Earlier reports retain archive input hashes, pipeline commands and
numerical oracles. Generated NetCDF binaries remain temporary artifacts.

Regression and pipeline scope
------------------------------

The unchanged production implementation retains the stage 5 result of 481
scoped regression tests, including a STEPS callback, plus 80 storage tests
against minimum xarray 2025.1.2. Those passing runs were not repeated for
these documentation and independent-reader additions. Current source hashes
were checked against their retained evidence. The new script executes all 12
user round trips, and the independent reader checks all 62 exported files.

The existing pipeline reads three MCH radar frames, converts to physical
rates, aggregates to 2 km, applies an explicit dBR transform, estimates
Lucas-Kanade motion, extrapolates selected leads 1, 2 and 4, inverts the model
transform, exports and reimports. Both floating and packed versions are now
read by Iris too. It validates I/O integration, not forecast skill.

Storage benchmarks are reused from :ref:`cf-netcdf-io-stage5`; no timing
threshold is introduced. Remaining scope limits are the explicitly unsupported
grids/products, untested datum shifts, the complete upstream regression suite,
and a full Sphinx/site build. Documentation syntax, local references, downloads
and navigation were checked separately. Remote submission and maintainer
integration decisions remain outside these completed local stages.
