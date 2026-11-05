.. _cf-netcdf-io-stage2:

CF NetCDF stage 2 implementation
================================

Recorded 5 November 2026 on ``cf-netcdf-io``, based on upstream
``f1a3af117a7085bef206a444edaa745b2431a9e7``. Stage 2 implements shared physical
metadata handling and the affected numerical helpers. The new file importer,
file exporter and packing profiles remain stages 3, 4 and 5 respectively.

Implemented behaviour
---------------------

``pysteps.dataset_metadata.normalize_dataset(dataset, variable=None)`` returns
a validated copy of an already decoded xarray dataset. The convenience wrapper
``pysteps.xarray_helpers.validate_dataset`` calls the same implementation.
Neither opens files, decodes packing nor changes a modelling transformation
into physical values. Select the active quantity explicitly, through a valid
``precip_var``, or through a unique physical variable; ambiguous input fails.
Other physical variables and their separate time axes are retained.

The shared helpers provide:

* Supported unit aliases with numerical scaling, valid CF standard names,
  canonical internal units ``mm/h``, ``mm`` and ``dBZ``, and removal of optional
  ``None`` attributes. Unsupported units and nonserializable metadata fail.
  Generic transformed fields must be explicitly returned to physical values.
* Explicit temporal support, authoritative bounds, end-of-interval labels,
  preserved source midpoint labels, and separate cadence (seconds) and
  accumulation duration (minutes) caches. Conflicting caches fail. Relabelling
  a shared axis must not move point observations; such inputs need separate
  axes. Times use decoded UTC datetime64 values, with microsecond precision.
* Full pyproj CRS metadata and WKT, including spherical and ellipsoidal earth
  models, normalization of coordinates and linear projection parameters to
  metres, polar origins, and checks against supplied geolocation and bounds.
  The physical profile requires a regular projected grid; existing geographic
  importer adapters remain available outside that profile.
* Physical-value checks, optional I/O-level thresholds, explicit model-readiness
  checks for the active field, scalar forecast reference time, and preserved
  integer ensemble identifiers. Time/member encodings use CF-1.8-compatible
  float64/int32 types rather than int64 defaults.

Use ``require_model=True`` only when checking the selected field for modelling;
this additionally requires finite scalar ``threshold`` and ``zerovalue``.
``validate_values=False`` defers ordinary precipitation/geolocation value scans;
it does not promise a wholly lazy importer. Lazy file ownership and the initial
zero cumulative-value check need explicit handling in stage 3.

Conversions and aggregation
---------------------------

Depth-to-rate conversion divides by the actual interval duration, keeps its
bounds, and labels the result as an interval mean. Rate-to-depth conversion
uses mean-rate bounds, or requires an explicit constant-rate integration rule
and duration for a point sample. Cadence alone is not an accumulation period.
For example::

    from pysteps.dataset_metadata import normalize_dataset
    from pysteps.utils.conversion import to_rainrate, to_raindepth
    from pysteps.utils.dimension import aggregate_fields_time

    depths = normalize_dataset(decoded_depths, variable="precip_accum")
    mean_rates = to_rainrate(depths)
    totals_15min = aggregate_fields_time(depths, 15)
    point_depths = to_raindepth(
        decoded_point_rates, integration="constant", duration_seconds=600
    )

The integration example declares the sample constant over the preceding
600 seconds; this assumption is recorded in history. It is not an integration
of an otherwise unknown time-varying rate.

Rainfall accumulated since a fixed starting time is supported through
common-origin bounds. Temporal aggregation differences these cumulative totals
before summing intervals, rejects decreasing totals/resets, and permits an
initial zero at the origin without dividing it by zero. Mean rates are weighted
by their actual interval duration. Source intervals must form complete,
contiguous, nonoverlapping windows; splitting intervals or silently discarding
a partial trailing window is rejected.

Missing values remain missing by default. Explicit ``ignore_nan=True`` records
a linked coverage fraction, while an entirely missing window still yields
missing rainfall. A partially observed depth is the observed amount, not an
estimate of the complete interval. Scalar thresholds that become time-dependent
after a conversion are represented by linked parameter fields. Temporal
aggregation rejects ancillary fields lacking a supported reduction policy.
Spatial aggregation recomputes geolocation at the new grid centres.

Physical dBZ is handled as a logarithmic reflectivity representation, including
legacy files carrying a redundant ``transform='dB'`` marker. It is not inverted
as logarithmic rainfall. Z-R coefficients and their source are recorded; finite
negative dBZ and positive rain traces remain valid. A declared detection flag
or floor distinguishes no detection from weak echoes. Finite dBZ output for
dry zero rain requires an explicit threshold/floor policy. Applying Z-R to an
interval mean does not recover instantaneous reflectivity; that historical
proxy is only available through explicit compatibility mode.

``legacy=True`` permits documented old duration/name/CRS conventions. For a
dated interval without bounds, an explicit ``accutime`` is interpreted as an
interval ending at valid time, and bounds/history are constructed. Old undated
array conversion fixtures exercise this explicit compatibility path. The
production archive adapter also records its known end-time convention.
BoM fractional and multi-day accumulation durations use total elapsed seconds.

Forecast metadata and compatibility
-----------------------------------

``convert_output_to_xarray_dataset`` labels already computed output. For an
interval product, its duration is independent of gaps between requested leads;
variable source durations require ``interval_seconds``. For a point product,
no interval is invented. ``accumulation='cumulative'`` declares precomputed
depths accumulated since forecast initialization; it does not integrate values.
Input cadence can be replaced by an explicit ``timestep_seconds`` only when
consistent with known input sampling. Ensemble identifiers are retained or
supplied explicitly. Observational quality/detection fields are not relabelled
as forecast values.

Legacy exporter signatures and callback modes are retained. Their duplicated
physical-name and CRS construction now uses the shared helpers. This does not
complete the new CF exporter or repair its existing packing implementation.
The legacy writer rejects unit scaling that it cannot perform, rather than
labelling unscaled values with canonical units.
``xarray`` and ``pyproj>=3.0`` are declared runtime dependencies; acceptance
used the versions below, without a lower-version compatibility matrix.

Validation and reproduction
---------------------------

The scoped regression run passed 306 tests, including 73 new shared-metadata
checks, with one pre-existing autoregression warning on the ANVIL dry fixture.
The additional importer-decorator and STEPS dry/callback run passed 8 tests.
Its 16 warnings concern existing MRMS pixel-spacing metadata. The callback test writes and reimports a two-member STEPS forecast
using the existing NetCDF interface. These are scoped tests, not a full-suite
or forecast-skill claim.

Reproduce the principal regression set with::

    python -m pytest -q \
      pysteps/tests/test_dataset_metadata.py \
      pysteps/tests/test_cf_netcdf_reference.py \
      pysteps/tests/test_utils_conversion.py \
      pysteps/tests/test_utils_transformation.py \
      pysteps/tests/test_utils_dimension.py \
      pysteps/tests/test_exporters.py \
      pysteps/tests/test_io_nowcast_importers.py \
      pysteps/tests/test_io_readers.py \
      pysteps/tests/test_io_fmi_pgm.py \
      pysteps/tests/test_io_fmi_geotiff.py \
      pysteps/tests/test_io_mch_gif.py \
      pysteps/tests/test_io_bom_rf3.py \
      pysteps/tests/test_io_opera_hdf5.py \
      pysteps/tests/test_nowcasts_anvil.py

    python -m pytest -q pysteps/tests/test_importer_decorator.py
    python -m pytest -q pysteps/tests/test_nowcasts_steps.py \
      -k 'callback or default_steps_norain'

Archive tests need the configured sample files. The local FMI GeoTIFF entry
needed two corrections in a temporary configuration: remove ``.tif`` from
``fn_pattern`` because ``fn_ext`` already supplies it, and set
``importer='fmi_geotiff'``. The user's configuration was not modified. Set
``PYSTEPSRC`` to the corrected test configuration when reproducing that run.

Generate 19 synthetic normalized/derived cases and optionally one archived
radar forecast with::

    python -m pysteps.tests.cf_netcdf_stage2_acceptance \
      --output /tmp/cf-stage2 --with-archive

Without ``--with-archive``, no local sample archive is needed. The archived
example reads three MCH frames from 15 May 2015, converts to rate, aggregates to
2 km, explicitly transforms to dBR with threshold 0.1 mm/h and floor -15,
estimates LK motion and extrapolates leads 1, 2 and 4. It returns to physical
rate before validation. All 20 datasets preserve physical values, masks and
coordinates through direct xarray serialization/reopening, with raw netCDF4
time/member dtype checks. This exercises shared metadata without claiming the
future file APIs. The report records source-file hashes, package versions,
output inventory and the actual numerical/mask assertions.

Independent CF validation uses compliance-checker 6.1.0, cf-units 3.3.1 and
standard-name table 95, the same isolated toolchain as stage 1::

    CF_STANDARD_NAME_TABLE=/path/to/v95.xml compliance-checker \
      -t cf:1.8 -c strict -f json_new -o /tmp/cf-stage2-checks.json \
      /tmp/cf-stage2/*.nc

Nineteen files pass the unmodified strict checker. The archived Swiss Oblique
Mercator example has one checker false positive: its required-attribute schema
uses ``azimuth`` instead of the convention's ``azimuth_of_central_line``.
The written field already has the latter and full WKT. See
`CF-1.8 Appendix F, Oblique Mercator
<https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_oblique_mercator>`_.

A second run corrects only that schema key in memory; all 20 files then satisfy
every scored point, with strict criteria and no skipped checks. No file
metadata is changed for the checker. Reproduce this explicitly with the checker
installed in the active Python environment::

    CF_STANDARD_NAME_TABLE=/path/to/v95.xml python - \
      -t cf:1.8 -c strict -f json_new -o /tmp/cf-stage2-corrected.json \
      /tmp/cf-stage2/*.nc <<'PY'
    import runpy
    import sys
    from pathlib import Path
    from compliance_checker.cf.appendix_f import grid_mapping_dict17

    required = grid_mapping_dict17["oblique_mercator"][0]
    assert "azimuth" in required and "azimuth_of_central_line" not in required
    grid_mapping_dict17["oblique_mercator"][0] = tuple(
        "azimuth_of_central_line" if key == "azimuth" else key
        for key in required
    )
    runpy.run_path(
        str(Path(sys.executable).parent / "cchecker.py"), run_name="__main__"
    )
    PY

The retained :download:`cf_netcdf_stage2_validation.json` records both results,
the exact correction, inventory and archive provenance. The files themselves
are generated artifacts. No packed-storage precision, performance, new
importer/exporter acceptance, second high-level CF reader or general forecast
skill is claimed. Those remain subsequent stages.
