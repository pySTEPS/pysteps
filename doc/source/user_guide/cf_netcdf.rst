.. _cf-netcdf-user-guide:

Reading and writing CF NetCDF
=============================

Use ``pysteps.io.import_netcdf_cf`` and ``export_netcdf_cf`` to exchange physical
rainfall and reflectivity on a regular projected grid. NetCDF4 is required;
lazy reading also requires ``dask[array]``. The supported dataset contract is
:ref:`cf-netcdf-io-contract` and the APIs are in the I/O reference.

Read, select and write
----------------------

.. code-block:: python

    from pysteps import io

    dataset = io.import_netcdf_cf("input.nc", variable="precip_accum")
    field = dataset[dataset.attrs["precip_var"]]
    io.export_netcdf_cf(dataset, "output.nc")

Selection uses the explicit ``variable``, then a valid dataset ``precip_var``
attribute, then a unique physical candidate. Ambiguity raises an error. All
supported fields are retained, including fields with separate time axes.
``variables=["precip_accum"]`` on export selects a subset and keeps its linked
coordinates, bounds and quality fields. Output is protected from replacement
unless ``overwrite=True``; a completed output is published atomically.

Import is eager and closes its source by default. For large files, close the
lazy dataset explicitly::

    with io.import_netcdf_cf("input.nc", load=False) as dataset:
        field = dataset[dataset.attrs["precip_var"]]
        first_maps = field.isel(time=slice(0, 2)).load()

Lazy metadata validation does not scan every physical value. Invalid values
are checked when explicitly validated or exported. ``group="forecast"`` reads
one named NetCDF group. No recursive group merge is performed.

Physical products and units
---------------------------

.. list-table:: Product meaning
   :header-rows: 1

   * - Product
     - In-memory units
     - Temporal support
   * - Instantaneous rain rate
     - mm/h
     - ``time: point``; no accumulation duration inferred.
   * - Interval mean rain rate
     - mm/h
     - ``time: mean`` and start/end bounds.
   * - Interval rainfall depth
     - mm
     - ``time: sum`` and each interval's start/end bounds.
   * - Cumulative rainfall depth
     - mm
     - ``time: sum``; rainfall accumulated since a fixed starting time.
   * - Radar reflectivity
     - dBZ
     - Point or explicitly declared interval support; negative values are valid.

The file uses ``mm h-1`` for rate. Equivalent source units, such as ``m s-1``,
are converted numerically as well as relabelled. Depth and cumulative depth
share ``lwe_thickness_of_precipitation_amount``; their bounds distinguish their
meaning. Cumulative totals must be differenced before temporal aggregation.
A reset is not treated as negative rainfall. Keep a forecast reference time
when available; it does not replace accumulation start/end bounds.

Sampling cadence is distinct from accumulation duration. The conversion
helpers use each interval's bounds::

    from pysteps.utils.conversion import to_rainrate, to_raindepth

    mean_rate = to_rainrate(depth_dataset)
    depth = to_raindepth(mean_rate)
    # An instantaneous rate requires an explicit integration assumption:
    depth = to_raindepth(point_dataset, integration="constant",
                        duration_seconds=450)

Overlapping rolling windows cannot be summed as disjoint intervals. Missing
coverage remains missing by default. Reading reflectivity does not apply a
Z-R relationship or infer a no-detection floor. Use explicit conversion
parameters when converting reflectivity to rainfall. Invert optional modelling
transforms, such as dBR, before exporting physical fields.

Projection and metadata
-----------------------

Datasets need physical x/y cell centres, a complete linked CF grid mapping
with an earth model, dated observations and consistent time bounds for
interval products. The writer stores CF projection parameters and WKT once,
without duplicating the grid for every forecast time/member. Optional
``include_latlon=True`` writes geographic coordinates. Metre/kilometre source
coordinates and projection offsets are normalized together.

Supported acceptance cases include NZTM, UTM 60S, Belgian Lambert conformal,
Australian national and radar Albers, spherical polar stereographic and Swiss
Oblique Mercator. Reading does not reproject. Geographic-only, curvilinear,
unstructured, vertical-volume, multiple-CRS and multiple-initialization
products are outside this implementation's scope. Missing CRS or time metadata
requires explicit supplied definitions; contradictory metadata is rejected.

Compact storage and missing values
----------------------------------

Floating output preserves input precision and defaults to zlib level 4,
shuffle, single time/member chunks and spatial tiles up to 256 cells. Integer
packing is opt-in and applies to canonical physical units::

    variable = dataset.attrs["precip_var"]
    io.export_netcdf_cf(dataset, "packed.nc", encoding={
        variable: {"dtype": "int16", "scale_factor": 0.1,
                   "add_offset": 0.0, "_FillValue": -32768,
                   "zlib": True, "complevel": 4, "shuffle": True}
    })

For rate this gives approximately 0.05 mm/h maximum quantization error and a
nonnegative decoded range of 0 to 3276.7 mm/h. Overflow and collisions with
the reserved missing code raise errors. Choose depth precision for the maximum
expected total; 0.01 mm steps have less range than 0.1 mm steps. Choose dBZ
precision independently. Source encoding is retained as provenance, not
silently reused for another physical quantity or output file.

NaN is missing; observed zero remains observed zero. Missing raw integer
codes are masked before scaling on input, and assigned after rounding on
output. Positive traces below roughly half a storage step can round to zero.
Classify unrounded values first and retain linked quality/detection flags if
this distinction matters. Packing does not define a rainfall threshold.

Runnable examples
-----------------

Download :download:`cf_netcdf_examples.py` and run::

    python cf_netcdf_examples.py --output /tmp/cf-user-examples

The script constructs instantaneous rate, interval mean, interval depth,
cumulative depth, negative/zero reflectivity and an ensemble with member labels
0 and 7. Each is exported as floating and int16 data and reimported with mask
and numerical checks. It uses an NZTM grid and requires no external radar data.
Use a new output directory for another run, since existing files are protected.

Compatibility and release notes
--------------------------------

* The new ``netcdf_cf`` registry entry exposes a dataset-first bulk API. The
  existing ``netcdf`` initializer/write/close callback API remains available
  for nowcasting and uses the same output schema.
* ``import_netcdf_pysteps`` delegates to the shared reader with legacy support.
  Explicit dtype/fillna/error behaviour is retained. Finite-fill views retain
  linked validity flags so a later export can recover missing cells.
* Metadata errors that previously produced guessed units, duration, CRS or
  observation time now require explicit repair. ``legacy=True`` enables
  documented legacy interpretation; it does not authorize contradictory
  definitions. ``metadata`` supplies missing ``attrs``, ``variables`` and
  ``coords`` definitions with provenance; see :ref:`cf-netcdf-io-stage34`.
* Rate/depth conversions now use bounds and distinguish cadence from duration.
  Point integration requires an explicit assumption. Scalar time and member
  coordinates remain singleton axes. Multi-variable files need clear selection.
* Optional modelling transforms must be inverted before physical export;
  dBZ itself is a physical representation and needs no inverse transform.
* The contribution requires xarray >= 2025.1.2 and pyproj >= 3.0. Iris and the
  CF checker are acceptance tools, not runtime dependencies.

Acceptance results and storage measurements are recorded in
:ref:`cf-netcdf-io-stage67` and :ref:`cf-netcdf-io-stage5`.
