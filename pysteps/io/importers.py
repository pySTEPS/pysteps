# -*- coding: utf-8 -*-
"""
pysteps.io.importers
====================

Methods for importing files containing two-dimensional radar mosaics.

The methods in this module implement the following interface::

    import_xxx(filename, optional arguments)

where **xxx** is the name (or abbreviation) of the file format and filename
is the name of the input file.

The output of each method is a three-element tuple containing a two-dimensional
radar mosaic, the corresponding quality field and a metadata dictionary. If the
file contains no quality information, the quality field is set to None. Pixels
containing missing data are set to nan.

The metadata dictionary contains the following recommended key-value pairs:

.. tabularcolumns:: |p{2cm}|L|

+------------------+----------------------------------------------------------+
|       Key        |                Value                                     |
+==================+==========================================================+
|   projection     | PROJ.4-compatible projection definition                  |
+------------------+----------------------------------------------------------+
|   x1             | x-coordinate of the lower-left corner of the data raster |
+------------------+----------------------------------------------------------+
|   y1             | y-coordinate of the lower-left corner of the data raster |
+------------------+----------------------------------------------------------+
|   x2             | x-coordinate of the upper-right corner of the data raster|
+------------------+----------------------------------------------------------+
|   y2             | y-coordinate of the upper-right corner of the data raster|
+------------------+----------------------------------------------------------+
|   xpixelsize     | grid resolution in x-direction                           |
+------------------+----------------------------------------------------------+
|   ypixelsize     | grid resolution in y-direction                           |
+------------------+----------------------------------------------------------+
|   cartesian_unit | the physical unit of the cartesian x- and y-coordinates: |
|                  | e.g. 'm' or 'km'                                         |
+------------------+----------------------------------------------------------+
|   yorigin        | a string specifying the location of the first element in |
|                  | the data raster w.r.t. y-axis:                           |
|                  | 'upper' = upper border                                   |
|                  | 'lower' = lower border                                   |
+------------------+----------------------------------------------------------+
|   institution    | name of the institution who provides the data            |
+------------------+----------------------------------------------------------+
|   unit           | the physical unit of the data: 'mm/h', 'mm' or 'dBZ'     |
+------------------+----------------------------------------------------------+
|   transform      | the transformation of the data: None, 'dB', 'Box-Cox' or |
|                  | others                                                   |
+------------------+----------------------------------------------------------+
|   accutime       | the accumulation time in minutes of the data, float      |
+------------------+----------------------------------------------------------+
|   threshold      | the rain/no rain threshold with the same unit,           |
|                  | transformation and accutime of the data.                 |
+------------------+----------------------------------------------------------+
|   zerovalue      | the value assigned to the no rain pixels with the same   |
|                  | unit, transformation and accutime of the data.           |
+------------------+----------------------------------------------------------+
|   zr_a           | the Z-R constant a in Z = a*R**b                         |
+------------------+----------------------------------------------------------+
|   zr_b           | the Z-R exponent b in Z = a*R**b                         |
+------------------+----------------------------------------------------------+

Available Importers
-------------------

.. autosummary::
    :toctree: ../generated/

    import_bom_rf3
    import_fmi_geotiff
    import_fmi_pgm
    import_knmi_hdf5
    import_mch_gif
    import_mch_hdf5
    import_mch_metranet
    import_mrms_grib
    import_odim_hdf5
    import_opera_hdf5
    import_saf_crri
"""

import gzip
import os
from functools import partial

import numpy as np
from matplotlib.pyplot import imread

from pysteps.decorators import postprocess_import
from pysteps.exceptions import DataModelError
from pysteps.exceptions import MissingOptionalDependency
from pysteps.utils import aggregate_fields

try:
    import gdalconst
    from osgeo import gdal, osr

    GDAL_IMPORTED = True
except ImportError:
    GDAL_IMPORTED = False

try:
    import h5py

    H5PY_IMPORTED = True
except ImportError:
    H5PY_IMPORTED = False

try:
    import metranet

    METRANET_IMPORTED = True
except ImportError:
    METRANET_IMPORTED = False

try:
    import netCDF4

    NETCDF4_IMPORTED = True
except ImportError:
    NETCDF4_IMPORTED = False

try:
    from PIL import Image

    PIL_IMPORTED = True
except ImportError:
    PIL_IMPORTED = False

try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False

try:
    import pygrib

    PYGRIB_IMPORTED = True
except ImportError:
    PYGRIB_IMPORTED = False


def _check_coords_range(selected_range, coordinate, full_range):
    """Check that the coordinates range arguments follow the expected pattern in
    the **import_mrms_grib** function."""

    if selected_range is None:
        return sorted(full_range)

    if not isinstance(selected_range, (list, tuple)):

        if len(selected_range) != 2:
            raise ValueError(
                f"The {coordinate} range must be None or a two-element tuple or list"
            )

        selected_range = list(selected_range)  # Make mutable

        for i in range(2):
            if selected_range[i] is None:
                selected_range[i] = full_range

        selected_range.sort()

    return tuple(selected_range)


def _get_grib_projection(grib_msg):
    """Get the projection parameters from the grib file."""
    projparams = grib_msg.projparams

    # Some versions of pygrib defines the regular lat/lon projections as "cyl",
    # which causes errors in pyproj and cartopy. Here we replace it for "longlat".
    if projparams["proj"] == "cyl":
        projparams["proj"] = "longlat"

    # Grib C tables (3-2)
    # https://apps.ecmwf.int/codes/grib/format/grib2/ctables/3/2
    # https://en.wikibooks.org/wiki/PROJ.4
    _grib_shapes_of_earth = dict()
    _grib_shapes_of_earth[0] = {"R": 6367470}
    _grib_shapes_of_earth[1] = {"R": 6367470}
    _grib_shapes_of_earth[2] = {"ellps": "IAU76"}
    _grib_shapes_of_earth[4] = {"ellps": "GRS80"}
    _grib_shapes_of_earth[5] = {"ellps": "WGS84"}
    _grib_shapes_of_earth[6] = {"R": 6371229}
    _grib_shapes_of_earth[8] = {"datum": "WGS84", "R": 6371200}
    _grib_shapes_of_earth[9] = {"datum": "OSGB36"}

    # pygrib defines the ellipsoids using "a" and "b" only.
    # Here we replace the for the PROJ.4 SpheroidCodes if they are available.
    if grib_msg["shapeOfTheEarth"] in _grib_shapes_of_earth:
        keys_to_remove = ["a", "b"]
        for key in keys_to_remove:
            if key in projparams:
                del projparams[key]

        projparams.update(_grib_shapes_of_earth[grib_msg["shapeOfTheEarth"]])

    return projparams


def _get_threshold_value(precip):
    """
    Get the the rain/no rain threshold with the same unit, transformation and
    accutime of the data.
    If all the values are NaNs, the returned value is `np.nan`.
    Otherwise, np.min(precip[precip > precip.min()]) is returned.

    Returns
    -------
    threshold: float
    """
    valid_mask = np.isfinite(precip)
    if valid_mask.any():
        _precip = precip[valid_mask]
        min_precip = _precip.min()
        above_min_mask = _precip > min_precip
        if above_min_mask.any():
            return np.min(_precip[above_min_mask])
        else:
            return min_precip
    else:
        return np.nan


@postprocess_import(dtype="float32")
def import_mrms_grib(filename, extent=None, window_size=4, **kwargs):
    """
    Importer for NSSL's Multi-Radar/Multi-Sensor System
    ([MRMS](https://www.nssl.noaa.gov/projects/mrms/)) rainrate product
    (grib format).

    The rainrate values are expressed in mm/h, and the dimensions of the data
    array are [latitude, longitude]. The first grid point (0,0) corresponds to
    the upper left corner of the domain, while (last i, last j) denote the
    lower right corner.

    Due to the large size of the dataset (3500 x 7000), a float32 type is used
    by default to reduce the memory footprint. However, be aware that when this
    array is passed to a pystep function, it may be converted to double
    precision, doubling the memory footprint.
    To change the precision of the data, use the *dtype* keyword.

    Also, by default, the original data is downscaled by 4
    (resulting in a ~4 km grid spacing).
    In case that the original grid spacing is needed, use `window_size=1`.
    But be aware that a single composite in double precipitation will
    require 186 Mb of memory.

    Finally, if desired, the precipitation data can be extracted over a
    sub region of the full domain using the `extent` keyword.
    By default, the entire domain is returned.

    Notes
    -----
    In the MRMS grib files, "-3" is used to represent "No Coverage" or
    "Missing data". However, in this reader replace those values by the value
    specified in the `fillna` argument (NaN by default).

    Note that "missing values" are not the same as "no precipitation" values.
    Missing values indicates regions with no valid measures.
    While zero precipitation indicates regions with valid measurements,
    but with no precipitation detected.

    Parameters
    ----------
    filename: str
        Name of the file to import.
    extent: None or array-like
        Longitude and latitude range (in degrees) of the data to be retrieved.
        (min_lon, max_lon, min_lat, max_lat).
        By default (None), the entire domain is retrieved.
        The extent can be in any form that can be converted to a flat array
        of 4 elements array (e.g., lists or tuples).
    window_size: array_like or int
        Array containing down-sampling integer factor along each axis.
        If an integer value is given, the same block shape is used for all the
        image dimensions.
        Default: window_size=4.

    {extra_kwargs_doc}

    Returns
    -------
    precipitation: 2D array, float32
        Precipitation field in mm/h. The dimensions are [latitude, longitude].
        The first grid point (0,0) corresponds to the upper left corner of the
        domain, while (last i, last j) denote the lower right corner.
    quality: None
        Not implement.
    metadata: dict
        Associated metadata (pixel sizes, map projections, etc.).
    """

    del kwargs

    if not PYGRIB_IMPORTED:
        raise MissingOptionalDependency(
            "pygrib package is required to import NCEP's MRMS products but it is not installed"
        )

    try:
        grib_file = pygrib.open(filename)
    except OSError:
        raise OSError(f"Error opening NCEP's MRMS file. " f"File Not Found: {filename}")

    if isinstance(window_size, int):
        window_size = (window_size, window_size)

    if extent is not None:
        extent = np.asarray(extent)
        if (extent.ndim != 1) or (extent.size != 4):
            raise ValueError(
                "The extent must be None or a flat array with 4 elements.\n"
                f"Received: extent.shape = {str(extent.shape)}"
            )

    # The MRMS grib file contain one message with the precipitation intensity
    grib_file.rewind()
    grib_msg = grib_file.read(1)[0]  # Read the only message

    # -------------------------
    # Read the grid information

    lr_lon = grib_msg["longitudeOfLastGridPointInDegrees"]
    lr_lat = grib_msg["latitudeOfLastGridPointInDegrees"]

    ul_lon = grib_msg["longitudeOfFirstGridPointInDegrees"]
    ul_lat = grib_msg["latitudeOfFirstGridPointInDegrees"]

    # Ni - Number of points along a latitude circle (west-east)
    # Nj - Number of points along a longitude meridian (south-north)
    # The lat/lon grid has a 0.01 degrees spacing.
    lats = np.linspace(ul_lat, lr_lat, grib_msg["Nj"])
    lons = np.linspace(ul_lon, lr_lon, grib_msg["Ni"])

    precip = grib_msg.values
    no_data_mask = precip == -3  # Missing values

    # Create a function with default arguments for aggregate_fields
    block_reduce = partial(aggregate_fields, method="mean", trim=True)

    if window_size != (1, 1):
        # Downscale data
        lats = block_reduce(lats, window_size[0])
        lons = block_reduce(lons, window_size[1])

        # Update the limits
        ul_lat, lr_lat = lats[0], lats[-1]  # Lat from North to south!
        ul_lon, lr_lon = lons[0], lons[-1]

        precip[no_data_mask] = 0  # block_reduce does not handle nan values
        precip = block_reduce(precip, window_size, axis=(0, 1))

        # Consider that if a single invalid observation is located in the block,
        # then mark that value as invalid.
        no_data_mask = block_reduce(
            no_data_mask.astype("int"), window_size, axis=(0, 1)
        ).astype(bool)

    lons, lats = np.meshgrid(lons, lats)
    precip[no_data_mask] = np.nan

    if extent is not None:
        # clip domain
        ul_lon, lr_lon = _check_coords_range(
            (extent[0], extent[1]), "longitude", (ul_lon, lr_lon)
        )

        lr_lat, ul_lat = _check_coords_range(
            (extent[2], extent[3]), "latitude", (ul_lat, lr_lat)
        )

        mask_lat = (lats >= lr_lat) & (lats <= ul_lat)
        mask_lon = (lons >= ul_lon) & (lons <= lr_lon)

        nlats = np.count_nonzero(mask_lat[:, 0])
        nlons = np.count_nonzero(mask_lon[0, :])

        precip = precip[mask_lon & mask_lat].reshape(nlats, nlons)

    proj_params = _get_grib_projection(grib_msg)
    pr = pyproj.Proj(proj_params)
    proj_def = " ".join([f"+{key}={value} " for key, value in proj_params.items()])

    xsize = grib_msg["iDirectionIncrementInDegrees"] * window_size[0]
    ysize = grib_msg["jDirectionIncrementInDegrees"] * window_size[1]

    x1, y1 = pr(ul_lon, lr_lat)
    x2, y2 = pr(lr_lon, ul_lat)

    metadata = dict(
        institution="NOAA National Severe Storms Laboratory",
        xpixelsize=xsize,
        ypixelsize=ysize,
        unit="mm/h",
        accutime=2.0,
        transform=None,
        zerovalue=0,
        projection=proj_def.strip(),
        yorigin="upper",
        threshold=_get_threshold_value(precip),
        x1=x1 - xsize / 2,
        x2=x2 + xsize / 2,
        y1=y1 - ysize / 2,
        y2=y2 + ysize / 2,
        cartesian_unit="degrees",
    )

    return precip, None, metadata


@postprocess_import()
def import_bom_rf3(filename, **kwargs):
    """Import a NetCDF radar rainfall product from the BoM Rainfields3.

    Parameters
    ----------
    filename: str
        Name of the file to import.

    {extra_kwargs_doc}

    Returns
    -------
    out: tuple
        A three-element tuple containing the rainfall field in mm/h imported
        from the Bureau RF3 netcdf, the quality field and the metadata. The
        quality field is currently set to None.
    """
    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required to import BoM Rainfields3 products "
            "but it is not installed"
        )

    precip = _import_bom_rf3_data(filename)

    geodata = _import_bom_rf3_geodata(filename)
    metadata = geodata
    # TODO(import_bom_rf3): Add missing georeferencing data.

    metadata["transform"] = None
    metadata["zerovalue"] = np.nanmin(precip)
    metadata["threshold"] = _get_threshold_value(precip)

    return precip, None, metadata


def _import_bom_rf3_data(filename):
    ds_rainfall = netCDF4.Dataset(filename)
    if "precipitation" in ds_rainfall.variables.keys():
        precipitation = ds_rainfall.variables["precipitation"][:]
    else:
        precipitation = None
    ds_rainfall.close()

    return precipitation


def _import_bom_rf3_geodata(filename):
    geodata = {}

    ds_rainfall = netCDF4.Dataset(filename)

    if "proj" in ds_rainfall.variables.keys():
        projection = ds_rainfall.variables["proj"]
        if getattr(projection, "grid_mapping_name") == "albers_conical_equal_area":
            projdef = "+proj=aea "
            lon_0 = getattr(projection, "longitude_of_central_meridian")
            projdef += " +lon_0=" + f"{lon_0:.3f}"
            lat_0 = getattr(projection, "latitude_of_projection_origin")
            projdef += " +lat_0=" + f"{lat_0:.3f}"
            standard_parallels = getattr(projection, "standard_parallel")
            projdef += " +lat_1=" + f"{standard_parallels[0]:.3f}"
            projdef += " +lat_2=" + f"{standard_parallels[1]:.3f}"
        else:
            projdef = None
    geodata["projection"] = projdef

    if "valid_min" in ds_rainfall.variables["x"].ncattrs():
        xmin = getattr(ds_rainfall.variables["x"], "valid_min")
        xmax = getattr(ds_rainfall.variables["x"], "valid_max")
        ymin = getattr(ds_rainfall.variables["y"], "valid_min")
        ymax = getattr(ds_rainfall.variables["y"], "valid_max")
    else:
        xmin = min(ds_rainfall.variables["x"])
        xmax = max(ds_rainfall.variables["x"])
        ymin = min(ds_rainfall.variables["y"])
        ymax = max(ds_rainfall.variables["y"])

    xpixelsize = abs(ds_rainfall.variables["x"][1] - ds_rainfall.variables["x"][0])
    ypixelsize = abs(ds_rainfall.variables["y"][1] - ds_rainfall.variables["y"][0])
    factor_scale = 1.0
    if "units" in ds_rainfall.variables["x"].ncattrs():
        if getattr(ds_rainfall.variables["x"], "units") == "km":
            factor_scale = 1000.0

    geodata["x1"] = xmin * factor_scale
    geodata["y1"] = ymin * factor_scale
    geodata["x2"] = xmax * factor_scale
    geodata["y2"] = ymax * factor_scale
    geodata["xpixelsize"] = xpixelsize * factor_scale
    geodata["ypixelsize"] = ypixelsize * factor_scale
    geodata["cartesian_unit"] = "m"
    geodata["yorigin"] = "upper"  # TODO(_import_bom_rf3_geodata): check this

    # get the accumulation period
    valid_time = None

    if "valid_time" in ds_rainfall.variables.keys():
        times = ds_rainfall.variables["valid_time"]
        calendar = "standard"
        if "calendar" in times.ncattrs():
            calendar = times.calendar
        valid_time = netCDF4.num2date(times[:], units=times.units, calendar=calendar)

    start_time = None
    if "start_time" in ds_rainfall.variables.keys():
        times = ds_rainfall.variables["start_time"]
        calendar = "standard"
        if "calendar" in times.ncattrs():
            calendar = times.calendar
        start_time = netCDF4.num2date(times[:], units=times.units, calendar=calendar)

    time_step = None

    if start_time is not None:
        if valid_time is not None:
            time_step = (valid_time - start_time).seconds // 60

    geodata["accutime"] = time_step

    # get the unit of precipitation
    if "units" in ds_rainfall.variables["precipitation"].ncattrs():
        units = getattr(ds_rainfall.variables["precipitation"], "units")
        if units in ("kg m-2", "mm"):
            geodata["unit"] = "mm"

    geodata["institution"] = "Commonwealth of Australia, Bureau of Meteorology"
    ds_rainfall.close()

    return geodata


@postprocess_import()
def import_fmi_geotiff(filename, **kwargs):
    """Import a reflectivity field (dBZ) from an FMI GeoTIFF file.

    Parameters
    ----------
    filename: str
        Name of the file to import.

    {extra_kwargs_doc}

    Returns
    -------
    out: tuple
        A three-element tuple containing the precipitation field,
        the associated quality field and metadata.
        The quality field is currently set to None.
    """
    if not GDAL_IMPORTED:
        raise MissingOptionalDependency(
            "gdal package is required to import "
            "FMI's radar reflectivity composite in GeoTIFF format "
            "but it is not installed"
        )

    f = gdal.Open(filename, gdalconst.GA_ReadOnly)

    rb = f.GetRasterBand(1)
    precip = rb.ReadAsArray()
    mask = precip == 255
    precip = precip.astype(float) * rb.GetScale() + rb.GetOffset()
    precip = (precip - 64.0) / 2.0
    precip[mask] = np.nan

    sr = osr.SpatialReference()
    pr = f.GetProjection()
    sr.ImportFromWkt(pr)

    projdef = sr.ExportToProj4()

    gt = f.GetGeoTransform()

    metadata = {}

    metadata["projection"] = projdef
    metadata["x1"] = gt[0]
    metadata["y1"] = gt[3] + gt[5] * f.RasterYSize
    metadata["x2"] = metadata["x1"] + gt[1] * f.RasterXSize
    metadata["y2"] = gt[3]
    metadata["xpixelsize"] = abs(gt[1])
    metadata["ypixelsize"] = abs(gt[5])
    if gt[5] < 0:
        metadata["yorigin"] = "upper"
    else:
        metadata["yorigin"] = "lower"
    metadata["institution"] = "Finnish Meteorological Institute"
    metadata["unit"] = rb.GetUnitType()
    metadata["transform"] = None
    metadata["accutime"] = 5.0
    metadata["threshold"] = _get_threshold_value(precip)
    metadata["zerovalue"] = np.nanmin(precip)
    metadata["cartesian_unit"] = "m"

    return precip, None, metadata


@postprocess_import()
def import_fmi_pgm(filename, gzipped=False, **kwargs):
    """Import a 8-bit PGM radar reflectivity composite from the FMI archive.

    Parameters
    ----------
    filename: str
        Name of the file to import.
    gzipped: bool
        If True, the input file is treated as a compressed gzip file.

    {extra_kwargs_doc}

    Returns
    -------
    out: tuple
        A three-element tuple containing the reflectivity composite in dBZ
        and the associated quality field and metadata. The quality field is
        currently set to None.

    Notes
    -----
    Reading georeferencing metadata is supported only for stereographic
    projection. For other projections, the keys related to georeferencing are
    not set.
    """
    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required to import "
            "FMI's radar reflectivity composite "
            "but it is not installed"
        )

    if gzipped is False:
        precip = imread(filename)
    else:
        precip = imread(gzip.open(filename, "r"))
    pgm_metadata = _import_fmi_pgm_metadata(filename, gzipped=gzipped)
    geodata = _import_fmi_pgm_geodata(pgm_metadata)

    mask = precip == pgm_metadata["missingval"]
    precip = precip.astype(float)
    precip[mask] = np.nan
    precip = (precip - 64.0) / 2.0

    metadata = geodata
    metadata["institution"] = "Finnish Meteorological Institute"
    metadata["accutime"] = 5.0
    metadata["unit"] = "dBZ"
    metadata["transform"] = "dB"
    metadata["zerovalue"] = np.nanmin(precip)
    metadata["threshold"] = _get_threshold_value(precip)
    metadata["zr_a"] = 223.0
    metadata["zr_b"] = 1.53

    return precip, None, metadata


def _import_fmi_pgm_geodata(metadata):
    geodata = {}

    projdef = ""

    if "type" in metadata.keys() and metadata["type"][0] == "stereographic":
        projdef += "+proj=stere "
        projdef += " +lon_0=" + metadata["centrallongitude"][0] + "E"
        projdef += " +lat_0=" + metadata["centrallatitude"][0] + "N"
        projdef += " +lat_ts=" + metadata["truelatitude"][0]
        # These are hard-coded because the projection definition
        # is missing from the PGM files.
        projdef += " +a=6371288"
        projdef += " +x_0=380886.310"
        projdef += " +y_0=3395677.920"
        projdef += " +no_defs"
        #
        geodata["projection"] = projdef

        ll_lon, ll_lat = [float(v) for v in metadata["bottomleft"]]
        ur_lon, ur_lat = [float(v) for v in metadata["topright"]]

        pr = pyproj.Proj(projdef)
        x1, y1 = pr(ll_lon, ll_lat)
        x2, y2 = pr(ur_lon, ur_lat)

        geodata["x1"] = x1
        geodata["y1"] = y1
        geodata["x2"] = x2
        geodata["y2"] = y2
        geodata["cartesian_unit"] = "m"
        geodata["xpixelsize"] = float(metadata["metersperpixel_x"][0])
        geodata["ypixelsize"] = float(metadata["metersperpixel_y"][0])

        geodata["yorigin"] = "upper"

    return geodata


def _import_fmi_pgm_metadata(filename, gzipped=False):
    metadata = {}

    if not gzipped:
        f = open(filename, "rb")
    else:
        f = gzip.open(filename, "rb")

    file_line = f.readline()
    while not file_line.startswith(b"#"):
        file_line = f.readline()
    while file_line.startswith(b"#"):
        x = file_line.decode()
        x = x[1:].strip().split(" ")
        if len(x) >= 2:
            k = x[0]
            v = x[1:]
            metadata[k] = v
        else:
            file_line = f.readline()
            continue
        file_line = f.readline()
    file_line = f.readline().decode()
    metadata["missingval"] = int(file_line)
    f.close()

    return metadata


@postprocess_import()
def import_knmi_hdf5(filename, qty="ACRR", accutime=5.0, pixelsize=1.0, **kwargs):
    """Import a precipitation or reflectivity field (and optionally the quality
    field) from a HDF5 file conforming to the KNMI Data Centre specification.

    Parameters
    ----------
    filename: str
        Name of the file to import.
    qty: {'ACRR', 'DBZH'}
        The quantity to read from the file. The currently supported identifiers
        are: 'ACRR'=hourly rainfall accumulation (mm) and 'DBZH'=max-reflectivity
        (dBZ). The default value is 'ACRR'.
    accutime: float
        The accumulation time of the dataset in minutes. A 5 min accumulation
        is used as default, but hourly, daily and monthly accumulations
        are also available.
    pixelsize: float
        The pixel size of a raster cell in kilometers. The default value for the
        KNMI datasets is a 1 km grid cell size, but datasets with 2.4 km pixel
        size are also available.

    {extra_kwargs_doc}

    Returns
    -------
    out: tuple
        A three-element tuple containing precipitation accumulation [mm] /
        reflectivity [dBZ] of the KNMI product, the associated quality field
        and metadata. The quality field is currently set to None.

    Notes
    -----
    Every KNMI data type has a slightly different naming convention. The
    standard setup is based on the accumulated rainfall product on 1 km2 spatial
    and 5 min temporal resolution.
    See https://data.knmi.nl/datasets?q=radar for a list of all available KNMI
    radar data.
    """

    # TODO: Add quality field.

    if not H5PY_IMPORTED:
        raise MissingOptionalDependency(
            "h5py package is required to import "
            "KNMI's radar datasets "
            "but it is not installed"
        )

    if qty not in ["ACRR", "DBZH"]:
        raise ValueError(
            "unknown quantity %s: the available options are 'ACRR' and 'DBZH' "
        )

    ####
    # Precipitation fields
    ####

    f = h5py.File(filename, "r")
    dset = f["image1"]["image_data"]
    precip_intermediate = np.copy(dset)  # copy the content

    # In case precip is a rainfall accumulation (ACRR), precip is divided by 100.0,
    # because the data is saved as hundreds of mm (so, as integers). 65535 is
    # the no data value. The precision of the data is two decimals (0.01 mm).
    if qty == "ACRR":
        precip = np.where(
            precip_intermediate == 65535, np.NaN, precip_intermediate / 100.0
        )

    # In case reflectivities are imported, the no data value is 255. Values are
    # saved as integers. The reflectivities are not directly saved in dBZ, but
    # as: dBZ = 0.5 * pixel_value - 32.0 (this used to be 31.5).
    if qty == "DBZH":
        precip = np.where(
            precip_intermediate == 255, np.NaN, precip_intermediate * 0.5 - 32.0
        )

    if precip is None:
        raise IOError("requested quantity not found")

    ####
    # Meta data
    ####

    metadata = {}

    if qty == "ACRR":
        unit = "mm"
        transform = None
    elif qty == "DBZH":
        unit = "dBZ"
        transform = "dB"

    # The 'where' group of mch- and Opera-data, is called 'geographic' in the
    # KNMI data.
    geographic = f["geographic"]
    proj4str = geographic["map_projection"].attrs["projection_proj4_params"].decode()
    pr = pyproj.Proj(proj4str)
    metadata["projection"] = proj4str

    # Get coordinates
    latlon_corners = geographic.attrs["geo_product_corners"]
    ll_lat = latlon_corners[1]
    ll_lon = latlon_corners[0]
    ur_lat = latlon_corners[5]
    ur_lon = latlon_corners[4]
    lr_lat = latlon_corners[7]
    lr_lon = latlon_corners[6]
    ul_lat = latlon_corners[3]
    ul_lon = latlon_corners[2]

    ll_x, ll_y = pr(ll_lon, ll_lat)
    ur_x, ur_y = pr(ur_lon, ur_lat)
    lr_x, lr_y = pr(lr_lon, lr_lat)
    ul_x, ul_y = pr(ul_lon, ul_lat)
    x1 = min(ll_x, ul_x)
    y1 = min(ll_y, lr_y)
    x2 = max(lr_x, ur_x)
    y2 = max(ul_y, ur_y)

    # Fill in the metadata
    metadata["x1"] = x1
    metadata["y1"] = y1
    metadata["x2"] = x2
    metadata["y2"] = y2
    metadata["xpixelsize"] = pixelsize
    metadata["ypixelsize"] = pixelsize
    metadata["cartesian_unit"] = "km"
    metadata["yorigin"] = "upper"
    metadata["institution"] = "KNMI - Royal Netherlands Meteorological Institute"
    metadata["accutime"] = accutime
    metadata["unit"] = unit
    metadata["transform"] = transform
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = _get_threshold_value(precip)
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6

    f.close()

    return precip, None, metadata


@postprocess_import()
def import_mch_gif(filename, product, unit, accutime, **kwargs):
    """Import a 8-bit gif radar reflectivity composite from the MeteoSwiss
    archive.

    Parameters
    ----------
    filename: str
        Name of the file to import.
    product: {"AQC", "CPC", "RZC", "AZC"}
        The name of the MeteoSwiss QPE product.\n
        Currently supported prducts:

        +------+----------------------------+
        | Name |          Product           |
        +======+============================+
        | AQC  |     Acquire                |
        +------+----------------------------+
        | CPC  |     CombiPrecip            |
        +------+----------------------------+
        | RZC  |     Precip                 |
        +------+----------------------------+
        | AZC  |     RZC accumulation       |
        +------+----------------------------+

    unit: {"mm/h", "mm", "dBZ"}
        the physical unit of the data
    accutime: float
        the accumulation time in minutes of the data

    {extra_kwargs_doc}

    Returns
    -------
    out: tuple
        A three-element tuple containing the precipitation field in mm/h imported
        from a MeteoSwiss gif file and the associated quality field and metadata.
        The quality field is currently set to None.
    """
    if not PIL_IMPORTED:
        raise MissingOptionalDependency(
            "PIL package is required to import "
            "radar reflectivity composite from MeteoSwiss"
            "but it is not installed"
        )

    geodata = _import_mch_geodata()

    metadata = geodata

    # import gif file
    img = Image.open(filename)

    if product.lower() in ["azc", "rzc", "precip"]:

        # convert 8-bit GIF colortable to RGB values
        img_rgb = img.convert("RGB")

        # load lookup table
        if product.lower() == "azc":
            lut_filename = os.path.join(
                os.path.dirname(__file__), "mch_lut_8bit_Metranet_AZC_V104.txt"
            )
        else:
            lut_filename = os.path.join(
                os.path.dirname(__file__), "mch_lut_8bit_Metranet_v103.txt"
            )
        lut = np.genfromtxt(lut_filename, skip_header=1)
        lut = dict(zip(zip(lut[:, 1], lut[:, 2], lut[:, 3]), lut[:, -1]))

        # apply lookup table conversion
        precip = np.zeros(len(img_rgb.getdata()))
        for i, dn in enumerate(img_rgb.getdata()):
            precip[i] = lut.get(dn, np.nan)

        # convert to original shape
        width, height = img.size
        precip = precip.reshape(height, width)

        # set values outside observational range to NaN,
        # and values in non-precipitating areas to zero.
        precip[precip < 0] = 0
        precip[precip > 9999] = np.nan

    elif product.lower() in ["aqc", "cpc", "acquire ", "combiprecip"]:

        # convert digital numbers to physical values
        img = np.array(img).astype(int)

        # build lookup table [mm/5min]
        lut = np.zeros(256)
        a = 316.0
        b = 1.5
        for i in range(256):
            if (i < 2) or (i > 250 and i < 255):
                lut[i] = 0.0
            elif i == 255:
                lut[i] = np.nan
            else:
                lut[i] = (10.0 ** ((i - 71.5) / 20.0) / a) ** (1.0 / b)

        # apply lookup table
        precip = lut[img]

    else:
        raise ValueError("unknown product %s" % product)

    metadata["accutime"] = accutime
    metadata["unit"] = unit
    metadata["transform"] = None
    metadata["zerovalue"] = np.nanmin(precip)
    metadata["threshold"] = _get_threshold_value(precip)
    metadata["institution"] = "MeteoSwiss"
    metadata["product"] = product
    metadata["zr_a"] = 316.0
    metadata["zr_b"] = 1.5

    return precip, None, metadata


@postprocess_import()
def import_mch_hdf5(filename, qty="RATE", **kwargs):
    """Import a precipitation field (and optionally the quality field) from a
    MeteoSwiss HDF5 file conforming to the ODIM specification.

    Parameters
    ----------
    filename: str
        Name of the file to import.
    qty: {'RATE', 'ACRR', 'DBZH'}
        The quantity to read from the file. The currently supported identitiers
        are: 'RATE'=instantaneous rain rate (mm/h), 'ACRR'=hourly rainfall
        accumulation (mm) and 'DBZH'=max-reflectivity (dBZ). The default value
        is 'RATE'.

    {extra_kwargs_doc}

    Returns
    -------
    out: tuple
        A three-element tuple containing the OPERA product for the requested
        quantity and the associated quality field and metadata. The quality
        field is read from the file if it contains a dataset whose quantity
        identifier is 'QIND'.
    """
    if not H5PY_IMPORTED:
        raise MissingOptionalDependency(
            "h5py package is required to import "
            "radar reflectivity composites using ODIM HDF5 specification "
            "but it is not installed"
        )

    if qty not in ["ACRR", "DBZH", "RATE"]:
        raise ValueError(
            "unknown quantity %s: the available options are 'ACRR', 'DBZH' and 'RATE'"
        )

    f = h5py.File(filename, "r")

    precip = None
    quality = None

    for dsg in f.items():
        if dsg[0].startswith("dataset"):
            what_grp_found = False
            # check if the "what" group is in the "dataset" group
            if "what" in list(dsg[1].keys()):
                qty_, gain, offset, nodata, undetect = _read_mch_hdf5_what_group(
                    dsg[1]["what"]
                )
                what_grp_found = True

            for dg in dsg[1].items():
                if dg[0][0:4] == "data":
                    # check if the "what" group is in the "data" group
                    if "what" in list(dg[1].keys()):
                        (
                            qty_,
                            gain,
                            offset,
                            nodata,
                            undetect,
                        ) = _read_mch_hdf5_what_group(dg[1]["what"])
                    elif not what_grp_found:
                        raise DataModelError(
                            "Non ODIM compliant file: "
                            "no what group found from {} "
                            "or its subgroups".format(dg[0])
                        )

                    if qty_.decode() in [qty, "QIND"]:
                        arr = dg[1]["data"][...]
                        mask_n = arr == nodata
                        mask_u = arr == undetect
                        mask = np.logical_and(~mask_u, ~mask_n)

                        if qty_.decode() == qty:
                            precip = np.empty(arr.shape)
                            precip[mask] = arr[mask] * gain + offset
                            precip[mask_u] = np.nan
                            precip[mask_n] = np.nan
                        elif qty_.decode() == "QIND":
                            quality = np.empty(arr.shape, dtype=float)
                            quality[mask] = arr[mask]
                            quality[~mask] = np.nan

    if precip is None:
        raise IOError("requested quantity %s not found" % qty)

    where = f["where"]

    geodata = _import_mch_geodata()
    metadata = geodata

    # TODO: use those from the hdf5 file instead
    # xpixelsize = where.attrs["xscale"] * 1000.0
    # ypixelsize = where.attrs["yscale"] * 1000.0
    # xsize = where.attrs["xsize"]
    # ysize = where.attrs["ysize"]

    if qty == "ACRR":
        unit = "mm"
        transform = None
    elif qty == "DBZH":
        unit = "dBZ"
        transform = "dB"
    else:
        unit = "mm/h"
        transform = None

    if np.any(np.isfinite(precip)):
        thr = np.nanmin(precip[precip > np.nanmin(precip)])
    else:
        thr = np.nan

    metadata.update(
        {
            "yorigin": "upper",
            "institution": "MeteoSwiss",
            "accutime": 5.0,
            "unit": unit,
            "transform": transform,
            "zerovalue": np.nanmin(precip),
            "threshold": thr,
            "zr_a": 316.0,
            "zr_b": 1.5,
        }
    )

    f.close()

    return precip, quality, metadata


def _read_mch_hdf5_what_group(whatgrp):
    qty = whatgrp.attrs["quantity"] if "quantity" in whatgrp.attrs.keys() else "RATE"
    gain = whatgrp.attrs["gain"] if "gain" in whatgrp.attrs.keys() else 1.0
    offset = whatgrp.attrs["offset"] if "offset" in whatgrp.attrs.keys() else 0.0
    nodata = whatgrp.attrs["nodata"] if "nodata" in whatgrp.attrs.keys() else 0
    undetect = whatgrp.attrs["undetect"] if "undetect" in whatgrp.attrs.keys() else -1.0

    return qty, gain, offset, nodata, undetect


@postprocess_import()
def import_mch_metranet(filename, product, unit, accutime):
    """Import a 8-bit bin radar reflectivity composite from the MeteoSwiss
    archive.

    Parameters
    ----------
    filename: str
        Name of the file to import.
    product: {"AQC", "CPC", "RZC", "AZC"}
        The name of the MeteoSwiss QPE product.\n
        Currently supported prducts:

        +------+----------------------------+
        | Name |          Product           |
        +======+============================+
        | AQC  |     Acquire                |
        +------+----------------------------+
        | CPC  |     CombiPrecip            |
        +------+----------------------------+
        | RZC  |     Precip                 |
        +------+----------------------------+
        | AZC  |     RZC accumulation       |
        +------+----------------------------+

    unit: {"mm/h", "mm", "dBZ"}
        the physical unit of the data
    accutime: float
        the accumulation time in minutes of the data

    {extra_kwargs_doc}

    Returns
    -------

    out: tuple
        A three-element tuple containing the precipitation field in mm/h imported
        from a MeteoSwiss gif file and the associated quality field and metadata.
        The quality field is currently set to None.
    """
    if not METRANET_IMPORTED:
        raise MissingOptionalDependency(
            "metranet package needed for importing MeteoSwiss "
            "radar composites but it is not installed"
        )

    ret = metranet.read_file(filename, physic_value=True, verbose=False)
    precip = ret.data

    geodata = _import_mch_geodata()

    # read metranet
    metadata = geodata
    metadata["institution"] = "MeteoSwiss"
    metadata["accutime"] = accutime
    metadata["unit"] = unit
    metadata["transform"] = None
    metadata["zerovalue"] = np.nanmin(precip)
    metadata["threshold"] = _get_threshold_value(precip)
    metadata["zr_a"] = 316.0
    metadata["zr_b"] = 1.5

    return precip, None, metadata


def _import_mch_geodata():
    """Swiss radar domain CCS4
    These are all hard-coded because the georeferencing is missing from the gif files.
    """

    geodata = {}

    # LV03 Swiss projection definition in Proj4
    projdef = ""
    projdef += "+proj=somerc "
    projdef += " +lon_0=7.43958333333333"
    projdef += " +lat_0=46.9524055555556"
    projdef += " +k_0=1"
    projdef += " +x_0=600000"
    projdef += " +y_0=200000"
    projdef += " +ellps=bessel"
    projdef += " +towgs84=674.374,15.056,405.346,0,0,0,0"
    projdef += " +units=m"
    projdef += " +no_defs"
    geodata["projection"] = projdef

    geodata["x1"] = 255000.0
    geodata["y1"] = -160000.0
    geodata["x2"] = 965000.0
    geodata["y2"] = 480000.0

    geodata["xpixelsize"] = 1000.0
    geodata["ypixelsize"] = 1000.0
    geodata["cartesian_unit"] = "m"
    geodata["yorigin"] = "upper"

    return geodata


@postprocess_import()
def import_odim_hdf5(filename, qty="RATE", **kwargs):
    """Import a precipitation field (and optionally the quality field) from a
    HDF5 file conforming to the ODIM specification.
    **Important:** Currently, only the Pan-European (OPERA) and the
    Dipartimento della Protezione Civile (DPC) radar composites are correctly supported.
    Other ODIM-compliant files may not be read correctly.

    Parameters
    ----------
    filename: str
        Name of the file to import.
    qty: {'RATE', 'ACRR', 'DBZH'}
        The quantity to read from the file. The currently supported identitiers
        are: 'RATE'=instantaneous rain rate (mm/h), 'ACRR'=hourly rainfall
        accumulation (mm) and 'DBZH'=max-reflectivity (dBZ). The default value
        is 'RATE'.

    {extra_kwargs_doc}

    Returns
    -------
    out: tuple
        A three-element tuple containing the OPERA product for the requested
        quantity and the associated quality field and metadata. The quality
        field is read from the file if it contains a dataset whose quantity
        identifier is 'QIND'.
    """
    if not H5PY_IMPORTED:
        raise MissingOptionalDependency(
            "h5py package is required to import "
            "radar reflectivity composites using ODIM HDF5 specification "
            "but it is not installed"
        )

    if qty not in ["ACRR", "DBZH", "RATE"]:
        raise ValueError(
            "unknown quantity %s: the available options are 'ACRR', 'DBZH' and 'RATE'"
        )

    f = h5py.File(filename, "r")

    precip = None
    quality = None

    for dsg in f.items():
        if dsg[0].startswith("dataset"):
            what_grp_found = False
            # check if the "what" group is in the "dataset" group
            if "what" in list(dsg[1].keys()):
                if "quantity" in dsg[1]["what"].attrs.keys():
                    qty_, gain, offset, nodata, undetect = _read_opera_hdf5_what_group(
                        dsg[1]["what"]
                    )
                    what_grp_found = True

            for dg in dsg[1].items():
                if dg[0][0:4] == "data":
                    # check if the "what" group is in the "data" group
                    if "what" in list(dg[1].keys()):
                        (
                            qty_,
                            gain,
                            offset,
                            nodata,
                            undetect,
                        ) = _read_opera_hdf5_what_group(dg[1]["what"])
                    elif not what_grp_found:
                        raise DataModelError(
                            "Non ODIM compliant file: "
                            "no what group found from {} "
                            "or its subgroups".format(dg[0])
                        )

                    if qty_.decode() in [qty, "QIND"]:
                        arr = dg[1]["data"][...]
                        mask_n = arr == nodata
                        mask_u = arr == undetect
                        mask = np.logical_and(~mask_u, ~mask_n)

                        if qty_.decode() == qty:
                            precip = np.empty(arr.shape)
                            precip[mask] = arr[mask] * gain + offset
                            if qty != "DBZH":
                                precip[mask_u] = offset
                            else:
                                precip[mask_u] = -30.0
                            precip[mask_n] = np.nan
                        elif qty_.decode() == "QIND":
                            quality = np.empty(arr.shape, dtype=float)
                            quality[mask] = arr[mask]
                            quality[~mask] = np.nan
                    if quality is None:
                        for dgg in dg[
                            1
                        ].items():  # da qui  ----------------------------
                            if dgg[0][0:7] == "quality":
                                quality_keys = list(dgg[1].keys())
                                if "what" in quality_keys:
                                    (
                                        qty_,
                                        gain,
                                        offset,
                                        nodata,
                                        undetect,
                                    ) = _read_opera_hdf5_what_group(dgg[1]["what"])
                                if qty_.decode() == "QIND":
                                    arr = dgg[1]["data"][...]
                                    mask_n = arr == nodata
                                    mask_u = arr == undetect
                                    mask = np.logical_and(~mask_u, ~mask_n)
                                    quality = np.empty(arr.shape)  # , dtype=float)
                                    quality[mask] = arr[mask] * gain + offset
                                    quality[
                                        ~mask
                                    ] = np.nan  # a qui -----------------------------

    if precip is None:
        raise IOError("requested quantity %s not found" % qty)

    where = f["where"]
    proj4str = where.attrs["projdef"].decode()
    pr = pyproj.Proj(proj4str)

    ll_lat = where.attrs["LL_lat"]
    ll_lon = where.attrs["LL_lon"]
    ur_lat = where.attrs["UR_lat"]
    ur_lon = where.attrs["UR_lon"]
    if (
        "LR_lat" in where.attrs.keys()
        and "LR_lon" in where.attrs.keys()
        and "UL_lat" in where.attrs.keys()
        and "UL_lon" in where.attrs.keys()
    ):
        lr_lat = float(where.attrs["LR_lat"])
        lr_lon = float(where.attrs["LR_lon"])
        ul_lat = float(where.attrs["UL_lat"])
        ul_lon = float(where.attrs["UL_lon"])
        full_cornerpts = True
    else:
        full_cornerpts = False

    ll_x, ll_y = pr(ll_lon, ll_lat)
    ur_x, ur_y = pr(ur_lon, ur_lat)

    if full_cornerpts:
        lr_x, lr_y = pr(lr_lon, lr_lat)
        ul_x, ul_y = pr(ul_lon, ul_lat)
        x1 = min(ll_x, ul_x)
        y1 = min(ll_y, lr_y)
        x2 = max(lr_x, ur_x)
        y2 = max(ul_y, ur_y)
    else:
        x1 = ll_x
        y1 = ll_y
        x2 = ur_x
        y2 = ur_y

    if "xscale" in where.attrs.keys() and "yscale" in where.attrs.keys():
        xpixelsize = where.attrs["xscale"]
        ypixelsize = where.attrs["yscale"]
    else:
        xpixelsize = None
        ypixelsize = None

    if qty == "ACRR":
        unit = "mm"
        transform = None
    elif qty == "DBZH":
        unit = "dBZ"
        transform = "dB"
    else:
        unit = "mm/h"
        transform = None

    metadata = {
        "projection": proj4str,
        "ll_lon": ll_lon,
        "ll_lat": ll_lat,
        "ur_lon": ur_lon,
        "ur_lat": ur_lat,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "xpixelsize": xpixelsize,
        "ypixelsize": ypixelsize,
        "cartesian_unit": "m",
        "yorigin": "upper",
        "institution": "Odyssey datacentre",
        "accutime": 15.0,
        "unit": unit,
        "transform": transform,
        "zerovalue": np.nanmin(precip),
        "threshold": _get_threshold_value(precip),
    }

    f.close()

    return precip, quality, metadata


def import_opera_hdf5(filename, qty="RATE", **kwargs):
    """
    Wrapper to :py:func:`pysteps.io.importers.import_odim_hdf5`
    to maintain backward compatibility with previous pysteps versions.

    **Important:** Use :py:func:`~pysteps.io.importers.import_odim_hdf5` instead.
    """
    return import_odim_hdf5(filename, qty=qty, **kwargs)


def _read_opera_hdf5_what_group(whatgrp):
    qty = whatgrp.attrs["quantity"]
    gain = whatgrp.attrs["gain"] if "gain" in whatgrp.attrs.keys() else 1.0
    offset = whatgrp.attrs["offset"] if "offset" in whatgrp.attrs.keys() else 0.0
    nodata = whatgrp.attrs["nodata"] if "nodata" in whatgrp.attrs.keys() else np.nan
    undetect = whatgrp.attrs["undetect"] if "undetect" in whatgrp.attrs.keys() else 0.0

    return qty, gain, offset, nodata, undetect


@postprocess_import()
def import_saf_crri(filename, extent=None, **kwargs):
    """Import a NetCDF radar rainfall product from the Convective Rainfall Rate
    Intensity (CRRI) product from the Satellite Application Facilities (SAF).

    Product description available on http://www.nwcsaf.org/crr_description
    (last visited Jan 26, 2020).

    Parameters
    ----------
    filename: str
        Name of the file to import.
    extent: scalars (left, right, bottom, top), optional
        The spatial extent specified in data coordinates.
        If None, the full extent is imported.

    {extra_kwargs_doc}

    Returns
    -------
    out: tuple
        A three-element tuple containing the rainfall field in mm/h, the quality
        field and the metadata imported from the CRRI SAF netcdf file.
        The quality field includes values [1, 2, 4, 8, 16, 24, 32] meaning
        "nodata", "internal_consistency", "temporal_consistency", "good",
        "questionable", "bad", and "interpolated", respectively.
    """
    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required to import CRRI SAF products "
            "but it is not installed"
        )

    geodata = _import_saf_crri_geodata(filename)
    metadata = geodata

    if extent:
        xcoord = (
            np.arange(metadata["x1"], metadata["x2"], metadata["xpixelsize"])
            + metadata["xpixelsize"] / 2
        )
        ycoord = (
            np.arange(metadata["y1"], metadata["y2"], metadata["ypixelsize"])
            + metadata["ypixelsize"] / 2
        )
        ycoord = ycoord[::-1]  # yorigin = "upper"
        idx_x = np.logical_and(xcoord < extent[1], xcoord > extent[0])
        idx_y = np.logical_and(ycoord < extent[3], ycoord > extent[2])

        # update geodata
        metadata["x1"] = xcoord[idx_x].min() - metadata["xpixelsize"] / 2
        metadata["x2"] = xcoord[idx_x].max() + metadata["xpixelsize"] / 2
        metadata["y1"] = ycoord[idx_y].min() - metadata["ypixelsize"] / 2
        metadata["y2"] = ycoord[idx_y].max() + metadata["ypixelsize"] / 2

    else:

        idx_x = None
        idx_y = None

    precip, quality = _import_saf_crri_data(filename, idx_x, idx_y)

    metadata["transform"] = None
    metadata["zerovalue"] = np.nanmin(precip)
    metadata["threshold"] = _get_threshold_value(precip)

    return precip, quality, metadata


def _import_saf_crri_data(filename, idx_x=None, idx_y=None):
    ds_rainfall = netCDF4.Dataset(filename)
    if "crr_intensity" in ds_rainfall.variables.keys():
        if idx_x is not None:
            data = np.array(ds_rainfall.variables["crr_intensity"][idx_y, idx_x])
            quality = np.array(ds_rainfall.variables["crr_quality"][idx_y, idx_x])
        else:
            data = np.array(ds_rainfall.variables["crr_intensity"])
            quality = np.array(ds_rainfall.variables["crr_quality"])
        precipitation = np.where(data == 65535, np.nan, data)
    else:
        precipitation = None
        quality = None
    ds_rainfall.close()

    return precipitation, quality


def _import_saf_crri_geodata(filename):
    geodata = {}

    ds_rainfall = netCDF4.Dataset(filename)

    # get projection
    projdef = ds_rainfall.getncattr("gdal_projection")
    geodata["projection"] = projdef

    # get x1, y1, x2, y2, xpixelsize, ypixelsize, yorigin
    geotable = ds_rainfall.getncattr("gdal_geotransform_table")
    xmin = ds_rainfall.getncattr("gdal_xgeo_up_left")
    xmax = ds_rainfall.getncattr("gdal_xgeo_low_right")
    ymin = ds_rainfall.getncattr("gdal_ygeo_low_right")
    ymax = ds_rainfall.getncattr("gdal_ygeo_up_left")
    xpixelsize = abs(geotable[1])
    ypixelsize = abs(geotable[5])
    geodata["x1"] = xmin
    geodata["y1"] = ymin
    geodata["x2"] = xmax
    geodata["y2"] = ymax
    geodata["xpixelsize"] = xpixelsize
    geodata["ypixelsize"] = ypixelsize
    geodata["cartesian_unit"] = "m"
    geodata["yorigin"] = "upper"

    # get the accumulation period
    geodata["accutime"] = None

    # get the unit of precipitation
    geodata["unit"] = ds_rainfall.variables["crr_intensity"].units

    # get institution
    geodata["institution"] = ds_rainfall.getncattr("institution")

    ds_rainfall.close()

    return geodata
