.. _cf-netcdf-io-stage34:

CF NetCDF stages 3 and 4
========================

Implemented on ``cf-netcdf-io`` from the committed stage 2 baseline
``beb4eb06b9032f8cf2ca19fdd2af55dba2f7cae3``. The physical file importer and
exporter, their registration, and the existing forecast callback adapters are
implemented. New packing profiles and performance measurements remain stage 5;
the full acceptance matrix and upstream contribution remain stages 6 and 7.

Public interfaces
-----------------

The new functions are available through ``pysteps.io`` and registered as
``netcdf_cf`` with ``io.get_method``. The exporter is a dataset-first bulk
function; the existing ``netcdf`` exporter remains an array callback initializer.

Typical use::

    from pysteps import io

    dataset = io.import_netcdf_cf("input.nc", variable="precip_accum")
    # Physical values, canonical units, explicit support and an active selector.
    path = io.export_netcdf_cf(dataset, "output.nc")

The signatures are::

    import_netcdf_cf(filename, *, variable=None, group=None,
                     dtype="double", load=True, legacy=False, metadata=None)
    export_netcdf_cf(dataset, filename, *, variables=None, encoding=None,
                     include_latlon=False, overwrite=False)

Import
------

The importer decodes CF packing and missing codes once with xarray/netCDF4,
then normalizes through the shared dataset validator. Declared valid ranges
are applied in the decoded affine domain before unit normalization, including
negative source gains. Their original raw bounds are retained as provenance.
The raw ``_FillValue`` and ``missing_value`` codes are masked before scaling.
Tests construct packed source arrays independently of the production writer.

The active variable follows explicit selection, a valid ``precip_var``, or a
unique supported CF candidate. All supported physical fields remain present,
including fields on different time axes. Input names are retained. Units and
numerical values are normalized together; rates use internal ``mm/h``, depths
``mm``, and reflectivity ``dBZ``. Reading dBZ does not apply a Z-R relation or
invent a detection floor, threshold or dry value.

The requested dtype is float32/single or float64/double, applied to the physical
fields. Source dtype, gain, offset and fill metadata are retained in
``dataset.encoding['pysteps_source_encoding']`` and selected prefixed provenance
attributes; source packing is not silently reused on export. Ensemble labels,
interval bounds, forecast origin, midpoint source times, quality fields and
relevant serializable metadata are retained.

Eager import is the default and closes the file before returning. Lazy import
requires optional ``dask[array]`` and transfers file ownership to the returned
Dataset::

    with io.import_netcdf_cf("input.nc", load=False) as dataset:
        subset = dataset[dataset.attrs["precip_var"]].isel(time=slice(0, 2)).load()

Lazy import validates metadata, coordinates and support without loading the
rainfall arrays. Value checks, including a zero cumulative initial value, are
deferred until explicit validation or export. Metadata bounds can be read to
interpret support. ``group`` selects one named group; no recursive merge is
performed. Incomplete pySTEPS output is rejected.

Scalar CF valid time and realization coordinates are promoted to singleton
axes. An undated 2D field still needs an explicit time override. Unsupported
calendars, ambiguous variables, conflicting definitions, dangling links,
invalid physical values and missing temporal support produce errors. CF links
are checked before coordinate decoding can discard a dangling reference.

Overrides and legacy files
--------------------------

``metadata`` supplies missing definitions through three mappings: ``attrs``
for global attributes, ``variables`` for variable or coordinate attributes,
and ``coords`` for new coordinate values. It cannot silently replace a
conflicting source definition. For example, a file with scientifically
identified fields and metric coordinates but missing CRS/time can use::

    import numpy as np

    dataset = io.import_netcdf_cf(
        "undated.nc",
        metadata={
            "attrs": {"projection": "EPSG:2193"},
            "coords": {"time": np.datetime64("2023-01-01T00:00:00")},
        },
    )

An explicitly supplied CRS can complete missing grid-mapping metadata; supplied
WKT is checked against the original CF parameters. A missing earth model does
not silently acquire WGS84. Overrides are recorded in history. Time and member
overrides must agree with any existing coordinates.

``legacy=True`` permits the documented pySTEPS names and projection caches.
An interval still needs bounds or a supplied ``accutime`` under the explicit
legacy end-time convention. Cadence is never substituted for accumulation
duration. Use ``metadata['variables']`` to supply missing units, CF identity,
cell methods or duration for the named field. Do not relabel a logarithmic
modelling field as a physical field; explicitly invert the transformation first.

``import_netcdf_pysteps`` delegates to the new importer with legacy mode enabled
by default. Its requested dtype and ``fillna`` now work. The historical
``onerror='warn'`` policy emits a RuntimeWarning and returns ``(None, None)``;
``onerror='raise'`` propagates the error. Unsupported integer dtypes and
unrepresentable fill values fail clearly.

A finite-fill compatibility view retains linked validity flags. Export restores
NaNs using those flags before validation, including when ``fillna=0`` was used.
Missing pixels therefore cannot become observed dry pixels on re-export.
Keep that validity mask with the view. For physical analysis, prefer the core
importer's NaNs.

Export
------

The exporter validates physical values and metadata before writing any batches.
It uses the shared schema, preserves floating precision by default, writes
rates as ``mm h-1``, and uses float64 projected coordinates/time and int32
realization labels. Inactive physical fields are serialized in CF dimension
order. CRS metadata and applicable coordinates are stored once. Generic
modelling transforms must be explicitly inverted; no automatic inverse or Z-R
conversion is hidden in writing.

By default all supported physical variables and retained auxiliary data are
written. ``variables`` selects a string or list of physical variable names;
the subset retains required bounds, CRS, ancillary fields and applicable
coordinates, and receives a valid active selector. No inactive quantity is
silently overwritten or dropped when all quantities are requested.

Projected files omit redundant latitude/longitude arrays by default.
``include_latlon=True`` retains validated geographic coordinates or computes
missing coordinates in the projection's geographic CRS. Both y orientations,
NZTM2000, UTM zone 60S, Australian national Albers, the BoM radar Albers
parameters, Belgian LCC and a polar sphere are covered.

Default physical storage is floating point with zlib level 4, shuffle and
chunks isolating a time/member and tiling space up to 256 by 256 cells.
``encoding`` is a per-variable mapping for floating dtype, ``_FillValue``,
``zlib``, ``complevel``, ``shuffle``, ``chunksizes`` and ``fletcher32``.
Coordinate precision cannot be reduced below the declared profile. Fill
collisions, overflow, flag-code truncation and invalid chunks fail. New bulk
integer packing/scale-offset profiles are deliberately rejected until stage 5.
Source encodings do not become output defaults.

Writing uses a temporary file beside the destination, followed by atomic
publication. Existing files are protected unless ``overwrite=True``; a failed
replacement leaves the existing file intact. A concurrent publication cannot
bypass overwrite protection. Failed writes close their handles and remove the
temporary file. Interrupted processes can leave an explicitly named hidden
``.incomplete-*`` artifact; they do not publish it as a completed forecast.

Existing forecast callbacks
---------------------------

``initialize_forecast_exporter_netcdf``, ``export_forecast_dataset`` and
``close_forecast_files`` use the same schema as the bulk exporter. Timestep
and member batches write directly into netCDF4 regions. The implementation
does not repeatedly rewrite the complete file. All dimensions, metadata and
coordinate values are fixed at initialization; complete publication occurs
only after all declared batches are supplied and the writer is closed.

The declared ``n_ens_members`` now also determines completeness for member
writes. Invalid shapes, negative physical rain, overflow, extra batches or an
incomplete close raise an error and discard the unpublished temporary output.
Existing explicit datatype, fill, gain, offset and compression options retain
their meanings, with range/fill checks. Packing means
``stored = (physical - offset) / scale_factor``; decoding performs the inverse.
These compatibility options are distinct from the new stage 5 packing profiles.

The legacy metadata dictionary describes outer grid edges and row orientation.
For interval rates/depths supply the known ``cell_methods`` and ``accutime``;
``timestep`` describes forecast sampling. Sparse lead gaps do not define product
duration. Point-valued rate/dBZ output does not acquire accumulation bounds.
Optional ``ensemble_ids`` retain labels such as 0 and 7. ``overwrite=True``
is needed to replace an existing target. The STEPS callback regression uses
physical values and explicitly supplies its interval meaning and duration.

Archive integration
-------------------

``read_timeseries`` can stack eagerly imported singleton CF frames. It checks
file times against archive dates and preserves their supplied interval bounds,
rather than replacing authoritative valid times. Missing fields remain NaN;
padding interval products requires one known fixed duration. Grids, quantities,
forecast origin and scalar modelling parameters must agree. Variable-duration
products with missing frames and varying scalar modelling parameters need an
explicit policy. Files with multiple valid times or incompatible time axes
should be read directly with ``import_netcdf_cf``. Lazy ownership is supported
by the direct importer, not the archive stacking helper.

Acceptance evidence
-------------------

The retained :download:`cf_netcdf_stage34_validation.json` records the source
baseline/hashes, environment, test results, fixture inventory and independent
checker scores. The final scoped regression run passes 399 tests, including
87 new file-I/O tests, with 17 existing MRMS grid-spacing/ANVIL dry-field warnings.
Two additional STEPS dry/callback tests pass. The 87 new tests also pass against
xarray 2025.1.2, installed only in a temporary directory; that version emits two
NumPy NaT deprecation warnings with the current NumPy. The conda environment
was not changed. The earlier minimum-version run found and fixed loss of
attributes when padding missing CF frames.

Xarray >=2025.1.2 is now declared because its public
``CFDatetimeCoder(time_unit='us')`` supports the contract's precision without
forcing nanosecond datetime ranges. See the
`xarray 2025.1.2 release notes
<https://docs.xarray.dev/en/v2025.01.2/whats-new.html>`_. Other lower dependency
versions were not independently validated; acceptance used the recorded
netCDF4/pyproj versions. The independent CF toolchain remains optional.

Run the new file tests and the existing regression set documented in stage 2::

    python -m pytest -q pysteps/tests/test_io_netcdf_cf.py
    python -m pytest -q pysteps/tests/test_importer_decorator.py
    python -m pytest -q pysteps/tests/test_nowcasts_steps.py \
      -k 'callback or default_steps_norain'

Generate the complete file acceptance set with::

    python -m pysteps.tests.cf_netcdf_stage34_acceptance \
      --output /tmp/cf-io --with-archive

Without ``--with-archive``, no local sample archive is needed. The full run checks
12 independently constructed input files, 23 production bulk exports and the
three legacy write modes. Physical values, missing masks, time/member/grid
coordinates and equivalent CRS survive both production import and independent
xarray/raw netCDF4 inspection. The archived MCH case includes three input frames,
2 km aggregation, explicit modelling transformation, LK motion and sparse-lead
extrapolation, followed by physical NetCDF export and import. Source-file hashes
and the actual assertions are recorded. This is an I/O integration check, not a
forecast-skill assessment.

Check the production exports with::

    CF_STANDARD_NAME_TABLE=/path/to/v95.xml compliance-checker \
      -t cf:1.8 -c strict -f json_new -o /tmp/cf-io-checks.json \
      /tmp/cf-io/*.nc

All 26 files satisfy every scored point after the single documented
Oblique Mercator schema correction from :ref:`cf-netcdf-io-stage2`. The
unmodified checker passes 25 and reports its known ``azimuth`` false positive
for the Swiss radar case. Both results are retained; no checks are skipped
and output metadata is not altered for the checker.

These checks cover floating file I/O and compatibility modes. They do not
complete the full upstream suite, a second independent high-level CF reader,
performance assessment, stage 5 encoding profiles or upstream coordination.
Documentation syntax, links and referenced artifacts are checked separately;
this is not a full Sphinx build.
