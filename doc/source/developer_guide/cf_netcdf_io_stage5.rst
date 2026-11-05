.. _cf-netcdf-io-stage5:

CF NetCDF stage 5 storage
=========================

Implemented from the committed stages 3/4 baseline
``a7c9c99815b89674b90909b84f1d6b791174ecb7``. Signed int16 packing is now an
explicit option for physical variables. Floating output remains the default.
Compression and chunk controls apply to both the bulk API and legacy callbacks.
The complete acceptance matrix and contribution preparation remain stages 6/7.

Packing and the missing code
----------------------------

For 0.1 mm/h storage, use the actual active variable name::

    from pysteps import io

    variable = dataset.attrs["precip_var"]
    io.export_netcdf_cf(
        dataset, "rain.nc",
        encoding={variable: {
            "dtype": "int16",
            "scale_factor": 0.1,
            "add_offset": 0.0,
            "_FillValue": -32768,
        }},
    )

``add_offset=0`` and ``_FillValue=-32768`` are the new int16 defaults; an
explicit positive finite ``scale_factor`` is required. The gain is specified
in the field's canonical physical units after unit normalization. In CF the
attribute is named ``scale_factor``. A storage multiplier of 10 corresponds
to a CF scale_factor of 0.1, since decoding multiplies the stored code by 0.1.

Encoding performs these operations explicitly:

1. Preserve the missing mask, including a supplied NumPy masked array, before
   any ndarray conversion. Reject unmasked infinities and negative physical
   rates/depths; valid negative dBZ is permitted.
2. Compute ``(physical - add_offset) / scale_factor`` in float64 and apply
   ``np.rint`` for integer storage, with ties to even in the computed code
   domain. No thresholding, clipping or wraparound is performed.
3. Check the rounded code range and collisions with the reserved fill code
   before the integer cast.
4. Assign the raw fill code directly to missing cells. Never multiply,
   divide, round or offset the fill code.
5. Write these integer codes with netCDF4 auto mask/scale disabled, preventing
   a second packing operation. The correctly typed fill value is defined when
   the NetCDF variable is created.

Decoding follows ``physical = code * scale_factor + add_offset``, after masking
raw missing codes. For example, code 13 is 1.3 mm/h, code 0 is observed zero,
and code -32768 is missing. With offset 2 mm/h, observed zero instead has code
-20; a negative stored code is therefore not automatically missing or invalid.

The nonnegative decoded range at 0.1 mm/h with offset zero is 0 to 3276.7 mm/h.
Input is accepted only when its rounded code fits and is not reserved. The
quantization bound is half the step, plus explicit floating-point read/roundoff
tolerance. The error bound is 0.05 mm/h for this profile. Decimal values very
close to a half step follow the represented float64 calculation; tests use
binary-exact half steps to verify the ties-to-even rule unambiguously.

Depth and reflectivity select their precision independently. Examples use
0.01 mm for depths and 0.1 dBZ for reflectivity; these are explicit examples,
not inherited rate defaults. A 0.01 mm int16 depth has a much smaller decoded
range than a 0.1 mm depth, so select precision for the intended totals. Interval
and cumulative time bounds and forecast origin are preserved through packing.

A custom representable fill code is supported, but a valid value that rounds
to it is rejected. This includes a zero fill code colliding with dry rainfall
and a negative dBZ code colliding with the chosen fill. A custom fill must
also lie outside the stored valid range; an interior hole is rejected. Where
CF/NUG inference from a custom fill would exclude real codes, the completed
file declares an explicit packed valid_range. For example, fill=0 with offset
2 and valid codes -20 to -7 receives valid_range=[-20, -7]. The existing callback
API retains its historical default fill unless ``fill_value`` is supplied;
pass -32768 explicitly to select this fill convention in that API.

Physical ``valid_range``/``valid_min``/``valid_max`` attributes are validated
against input values and converted to the packed domain with the storage
dtype. Incorrect validity attributes cannot silently mask valid observations
on reimport. Optional derived actual_range statistics are omitted on output
rather than carrying stale values through normalization or quantization.
Encoding that would decode to infinity or negative physical rain is rejected. Source ``_Unsigned`` metadata is not reapplied to the fresh signed
encoding. Coordinate/time/member types remain unchanged by rain packing.

Missing, dry and trace rainfall
-------------------------------

NaN, a masked value, or a finite-fill compatibility view with retained validity
flags becomes the reserved missing code. Dry rainfall remains an unmasked
zero with the zero-offset profile, even when missing cells appear next to it.
Reading and explicitly repacking a file does not apply its source gain twice;
a normal floating re-export does not inherit source packing.

A requested 0.1 mm/h step can round positive traces below approximately
0.05 mm/h to observed zero. Quantization alone cannot retain the distinction
between that trace and exactly dry rainfall. If source classification matters,
classify the unrounded physical values first and retain explicit source flags
as ancillary variables. Those flags are preserved independently of numerical
packing. No model threshold or detection classification is invented by I/O.

Compression and chunks
-----------------------

Defaults remain zlib level 4, shuffle, one member/time per chunk and spatial
tiles up to 256 by 256 cells. The measured alternatives and limits are below.
Per-variable ``encoding`` accepts ``zlib``, ``complevel``, ``shuffle``,
``fletcher32`` and ``chunksizes``. Boolean options must be actual booleans.
Chunk sizes must match the variable's dimension order and fit its dimensions.

For example, ``chunksizes=(1, 1, 128, 128)`` applies to an ensemble variable
ordered (ens_number, time, y, x), subject to the actual grid size. The
``initialize_forecast_exporter_netcdf`` compatibility API accepts these filter
and chunk controls as keyword arguments. Its bulk, timestep and member modes
continue to write through the same schema and raw-code packing implementation.
An invalid batch or incomplete forecast is not published.

Validation
-----------

The final scoped suite passes 479 tests plus two STEPS dry/callback checks.
The 80 new storage tests also pass with minimum xarray 2025.1.2. Existing
warnings are the same MRMS grid-spacing/ANVIL dry-field warnings and the older
xarray NumPy NaT deprecation warnings recorded in stage 3/4.

The storage tests include:

* Direct comparison with hand-specified raw codes for float32/float64 inputs,
  nonzero offsets, custom fill codes and rate/depth/dBZ products.
* Exact missing masks in netCDF4, xarray and the production importer. Raw fill
  codes are checked independently with automatic scale/mask disabled.
* All-dry and all-missing fields, observed traces that quantize to zero,
  declared detection flags, negative/zero dBZ, and finite-fill views.
* Every nonmissing signed int16 code for reflectivity, every nonnegative code
  for rain rate, binary half steps, range boundaries, overflow and fill
  collisions checked before casting, including decoded-value overflow and
  unsuitable offsets. The stress values test the codec, not a
  climatologically realistic range of reflectivity.
* Unpack/repack stability, floating re-export, validity limits in the packed
  domain, and rejection of non-finite source gains.
* Masked integer callback arrays with hidden zero or -32768 values in all
  three write modes. This fixes a case where np.asarray could discard a mask
  and turn hidden values into observations or invalid physical rainfall.
* Compression, shuffle, checksum and chunk controls that leave physical values
  and raw codes unchanged.

Reproduce the focused checks and file acceptance with::

    python -m pytest -q pysteps/tests/test_io_netcdf_storage.py
    python -m pysteps.tests.cf_netcdf_storage_acceptance \
      --output /tmp/cf-storage --with-archive

The acceptance helper verifies 21 packed bulk exports, three callback modes
and an independently handwritten packed input with two raw missing markers
and a nonzero offset. The archive case is the same explicitly transformed,
extrapolated and returned-to-physical MCH forecast used in stage 3/4. Omit
``--with-archive`` to run without local sample archives. NetCDF4 and xarray
reads plus the production importer compare masks and numerical errors; bounds, grids, CRS and
ensemble labels are also checked.

All 24 production files satisfy strict CF-1.8 checking with the single known
Oblique Mercator checker schema correction documented in
:ref:`cf-netcdf-io-stage2`. The unmodified checker passes 23 and reports that
known false positive on the Swiss radar case. No checks are skipped. Evidence,
source hashes and file inventories are in
:download:`cf_netcdf_stage5_validation.json`.

Storage measurements
---------------------

Run the benchmark on identical float32 physical inputs with::

    python -m pysteps.tests.cf_netcdf_storage_benchmark \
      --output /tmp/cf-storage-benchmark --repeats 3 --with-archive

The synthetic ensemble has shape (4, 8, 384, 512), 25.17 MB of float32 values,
spatial/temporal dependence, dry pixels, traces and 116,901 missing cells. The
archived forecast has shape (3, 320, 355) and 111,652 missing cells. Archive
values are explicitly converted to float32 for a consistent comparison. The
helper records input hashes, code hashes, versions, filters, chunk layouts,
file sizes and three timings for writing, full reads, one-map reads and a
single-point time series.

Reads reopen each file with netCDF4 automatic decoding and materialize the
requested data into float32. Measurements use the operating-system cache;
writing includes the public API's validation, encoding and close. They do not
measure cold-cache access or fsync durability throughput. There are no
machine-dependent timing thresholds in the regression suite. Full samples
and the observed environment are retained in
:download:`cf_netcdf_stage5_benchmark.json`.

.. list-table:: Representative zlib level 4 results
   :header-rows: 1
   :widths: 34 22 22 22

   * - Input
     - Float32 file
     - Int16 file
     - Packed reduction
   * - Synthetic ensemble
     - 9.79 MB
     - 2.73 MB
     - 72%
   * - Archived radar forecast
     - 0.259 MB
     - 0.094 MB
     - 64%

These are observed file sizes, not promised compression ratios. Uncompressed
chunked files can exceed their logical payload because partial edge chunks
are allocated. Exact bytes, write/read medians and variability are in the
benchmark record.

Level 9 gives only a few percent additional size reduction relative to level 4
while requiring substantially longer writes, especially for float32. Shuffle
reduces the packed file size in both scenarios. Tiles of 128 cells improve
point-series reads, while whole-frame chunks favour map reads/size and can
make point reads slower. Grouping four times or members makes a one-map read
decode unrelated fields. Retain the 256-cell, one-time/member default as the
general compromise; choose 128-cell tiles for workloads dominated by point
extraction. The two inputs do not establish a universal optimum.

Stages 6 and 7 still cover the full independent-reader/acceptance matrix and
upstream contribution. This stage does not evaluate forecast skill or the
entire upstream regression suite. Documentation syntax and links are checked
separately; no full Sphinx build is claimed.

References
-----------

* `CF-1.8 missing data and packed data
  <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html>`_
* `netCDF4 automatic mask/scale control
  <https://unidata.github.io/netcdf4-python/>`_
