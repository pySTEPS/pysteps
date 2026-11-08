.. _cf-netcdf-user-guide:

Reading and writing CF NetCDF
=============================

Use ``pysteps.io.import_netcdf_cf`` and ``export_netcdf_cf`` to exchange physical
rainfall and reflectivity on a regular projected grid. NetCDF4 is required;
lazy reading also requires ``dask[array]``. Exports use CF-1.8 conventions.
See :func:`pysteps.io.netcdf_cf.import_netcdf_cf` and
:func:`pysteps.io.netcdf_cf.export_netcdf_cf` for all API options.

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
     - ``time: point``; negative values are valid.

The file uses ``mm h-1`` for rate. Equivalent source units, such as ``m s-1``,
are converted numerically as well as relabelled. Depth and cumulative depth
share ``lwe_thickness_of_precipitation_amount``; their bounds distinguish their
meaning. Cumulative totals must be differenced before temporal aggregation.
A reset is not treated as negative rainfall. Keep a forecast reference time
when available; it does not replace accumulation start/end bounds.

These are pySTEPS normalized units. The CF standard-name table's canonical
unit for ``lwe_precipitation_rate`` is ``m s-1``; CF permits physically
equivalent units such as ``mm h-1``. Packing precision is expressed in the
pySTEPS normalized units, rather than the CF canonical units.

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

Converting interval mean rain back to dBZ cannot recover instantaneous
reflectivity. ``to_reflectivity(..., legacy=True)`` permits the historical
approximation when an existing workflow explicitly needs that proxy.

Dataset requirements
--------------------

Physical variables are identified by CF ``standard_name`` and units, rather
than their names. Rates use ``lwe_precipitation_rate``, depths use
``lwe_thickness_of_precipitation_amount``, and reflectivity uses
``equivalent_reflectivity_factor``. Finite negative rain rates and depths and
infinite physical values are rejected; negative reflectivity is valid.

The active field has dimensions ``(time, y, x)`` or
``(ens_number, time, y, x)``. Scalar valid-time and member coordinates become
singleton axes. Other physical variables may retain separate time axes.
Member coordinates use ``standard_name="realization"``; labels, including
zero and nonconsecutive integers, are preserved.

Times are decoded UTC ``datetime64`` values with microsecond resolution.
Non-Gregorian calendars are unsupported. Interval products require a linked
bounds array of shape ``(time, 2)`` containing start and end times. Imported
midpoint labels are retained as auxiliary source times while computational
times use interval ends. A scalar ``accutime`` cache is in minutes and must
agree with the bounds; ``time.stepsize`` describes sampling in seconds.

Threshold and dry-value attributes are optional for I/O. Reading a physical
field does not supply model-specific thresholds or make it ready for every
nowcast algorithm.

Projection and metadata
-----------------------

Datasets need physical x/y cell centres, a complete linked CF grid mapping
with an earth model, dated observations and consistent time bounds for
interval products. The writer stores CF projection parameters and WKT once,
without duplicating the grid for every forecast time/member. Optional
``include_latlon=True`` writes geographic coordinates. Metre/kilometre source
coordinates and projection offsets are normalized together.

Supported projection examples include NZTM, UTM 60S, Lambert conformal,
Albers, spherical polar stereographic and Oblique Mercator. Reading does not
reproject. Geographic-only, curvilinear, unstructured, vertical-volume,
multiple-CRS and multiple-initialization
products are outside this implementation's scope. Missing CRS or time metadata
requires explicit supplied definitions; contradictory metadata is rejected.

Supply known missing definitions through ``metadata``. For example, a field
with valid physical units and grid coordinates but no CRS or observation time
can be read with::

    import numpy as np

    dataset = io.import_netcdf_cf(
        "undated.nc",
        metadata={
            "attrs": {"projection": "EPSG:2193"},
            "coords": {"time": np.datetime64("2023-01-01T00:00:00")},
        },
    )

Use the actual CRS and observation time for the source. The optional
``metadata["variables"]`` mapping supplies missing attributes by variable
name, such as units, CF identity or cell methods. Supplied definitions must
agree with existing metadata and are recorded in processing history.

Compact storage and missing values
----------------------------------

Floating output preserves input precision and defaults to zlib level 4,
shuffle, single time/member chunks and spatial tiles up to 256 cells. Integer
packing is opt-in and applies to pySTEPS normalized physical units::

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

This is `CF packed data (section 8.1)
<https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#packed-data>`_.
Both ``scale_factor`` and ``add_offset`` are stored as float64; ``_FillValue``
and validity attributes use the stored variable's dtype. Omit ``_FillValue``
from the encoding to use the explicit reserved code ``-32768`` for int16.
Setting it to ``None`` is rejected for packed physical fields. This is a
pySTEPS writer policy that keeps missing-code assignment and collision checks
explicit; NetCDF itself also permits an implicit default fill value.

Packing quantizes values before writing; zlib and shuffle then operate
losslessly on the stored codes. Optional ``fletcher32=True`` checks those
stored bytes for corruption and does not measure quantization error. It is
compatible with this packing, but should not be combined with lossy HDF5
compression filters. The default is ``False``.

Source packing and compression settings are cleared before export because
decoded values may have changed units or precision. Reusing the source
encoding could apply an obsolete storage contract. Source settings remain
available as provenance; explicit output encoding defines the new file.

NaN is missing; observed zero remains observed zero. Missing raw integer
codes are masked before scaling on input, and assigned after rounding on
output. Positive traces below roughly half a storage step can round to zero.
Classify unrounded values first and retain linked quality/detection flags if
this distinction matters. Packing does not define a rainfall threshold.

Runnable examples
-----------------

For an end-to-end radar workflow, see
:doc:`../auto_examples/write_steps_netcdf`. It generates a four-member STEPS
ensemble and writes it together with the three physical input fields in one
NetCDF file, using separate observation and forecast time axes. Both rain-rate
variables use int16 packing at 0.1 mm/h precision with compression. It then
reads the file back, checks raw codes, missing masks, quantization error and
metadata, and plots the stored fields.

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
  ``coords`` definitions with provenance, as illustrated above. Legacy interval
  fields without bounds need an explicit ``accutime`` in minutes, interpreted
  as the interval ending at the supplied valid time.
* Rate/depth conversions now use bounds and distinguish cadence from duration.
  Point integration requires an explicit assumption. Scalar time and member
  coordinates remain singleton axes. Multi-variable files need clear selection.
* Optional modelling transforms must be inverted before physical export;
  dBZ itself is a physical representation and needs no inverse transform.
* CF NetCDF I/O requires xarray >= 2025.1.2 and pyproj >= 3.0. Iris and the
  CF checker are development tools, not runtime dependencies.
