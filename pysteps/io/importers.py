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
|    projection    | PROJ.4-compatible projection definition                  |
+------------------+----------------------------------------------------------+
|    x1            | x-coordinate of the lower-left corner of the data raster |
|                  | (meters)                                                 |
+------------------+----------------------------------------------------------+
|    y1            | y-coordinate of the lower-left corner of the data raster |
|                  | (meters)                                                 |
+------------------+----------------------------------------------------------+
|    x2            | x-coordinate of the upper-right corner of the data raster|
|                  | (meters)                                                 |
+------------------+----------------------------------------------------------+
|    y2            | y-coordinate of the upper-right corner of the data raster|
|                  | (meters)                                                 |
+------------------+----------------------------------------------------------+
|    xpixelsize    | grid resolution in x-direction (meters)                  |
+------------------+----------------------------------------------------------+
|    ypixelsize    | grid resolution in y-direction (meters)                  |
+------------------+----------------------------------------------------------+
|    yorigin       | a string specifying the location of the first element in |
|                  | the data raster w.r.t. y-axis:                           |
|                  | 'upper' = upper border                                   |
|                  | 'lower' = lower border                                   |
+------------------+----------------------------------------------------------+
|    institution   | name of the institution who provides the data            |
+------------------+----------------------------------------------------------+
|    unit          | the physical unit of the data: 'mm/h', 'mm' or 'dBZ'     |
+------------------+----------------------------------------------------------+
|    transform     | the transformation of the data: None, 'dB', 'Box-Cox' or |
|                  | others                                                   |
+------------------+----------------------------------------------------------+
|    accutime      | the accumulation time in minutes of the data, float      |
+------------------+----------------------------------------------------------+
|    threshold     | the rain/no rain threshold with the same unit,           |
|                  | transformation and accutime of the data.                 |
+------------------+----------------------------------------------------------+
|    zerovalue     | the value assigned to the no rain pixels with the same   |
|                  | unit, transformation and accutime of the data.           |
+------------------+----------------------------------------------------------+
|    zr_a          | the Z-R constant a in Z = a*R**b                         |
+------------------+----------------------------------------------------------+
|    zr_b          | the Z-R exponent b in Z = a*R**b                         |
+------------------+----------------------------------------------------------+

Available Importers
-------------------

.. autosummary::
    :toctree: ../generated/

    import_bom_rf3
    import_fmi_geotiff
    import_fmi_pgm
    import_mch_gif
    import_mch_hdf5
    import_mch_metranet
    import_opera_hdf5
    import_knmi_hdf5
"""

import gzip
import os

import numpy as np
from matplotlib.pyplot import imread

from pysteps.exceptions import DataModelError
from pysteps.exceptions import MissingOptionalDependency

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
    import PIL

    PIL_IMPORTED = True
except ImportError:
    PIL_IMPORTED = False

try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False


def import_bom_rf3(filename, **kwargs):
    """Import a NetCDF radar rainfall product from the BoM Rainfields3.

    Parameters
    ----------

    filename : str
        Name of the file to import.

    Returns
    -------

    out : tuple
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
    if np.any(np.isfinite(precip)):
        metadata["threshold"] = np.nanmin(precip[precip > np.nanmin(precip)])
    else:
        metadata["threshold"] = np.nan

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


def import_fmi_geotiff(filename, **kwargs):
    """Import a reflectivity field (dBZ) from an FMI GeoTIFF file.

    Parameters
    ----------

    filename : str
        Name of the file to import.

    Returns
    -------

    out : tuple
        A three-element tuple containing the precipitation field, the associated
        quality field and metadata. The quality field is currently set to None.
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
    precip_min = np.nanmin(precip)
    metadata["threshold"] = np.nanmin(precip[precip > precip_min])
    metadata["zerovalue"] = precip_min

    return precip, None, metadata


def import_fmi_pgm(filename, **kwargs):
    """Import a 8-bit PGM radar reflectivity composite from the FMI archive.

    Parameters
    ----------

    filename : str
        Name of the file to import.

    Other Parameters
    ----------------

    gzipped : bool
        If True, the input file is treated as a compressed gzip file.

    Returns
    -------

    out : tuple
        A three-element tuple containing the reflectivity composite in dBZ
        and the associated quality field and metadata. The quality field is
        currently set to None.
    """
    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required to import "
            "FMI's radar reflectivity composite "
            "but it is not installed"
        )

    gzipped = kwargs.get("gzipped", False)

    pgm_metadata = _import_fmi_pgm_metadata(filename, gzipped=gzipped)

    if gzipped is False:
        precip = imread(filename)
    else:
        precip = imread(gzip.open(filename, "r"))
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
    if np.any(np.isfinite(precip)):
        metadata["threshold"] = np.nanmin(precip[precip > np.nanmin(precip)])
    else:
        metadata["threshold"] = np.nan
    metadata["zr_a"] = 223.0
    metadata["zr_b"] = 1.53

    return precip, None, metadata


def _import_fmi_pgm_geodata(metadata):
    geodata = {}

    projdef = ""

    if metadata["type"][0] != "stereographic":
        raise ValueError("unknown projection %s" % metadata["type"][0])
    projdef += "+proj=stere "
    projdef += " +lon_0=" + metadata["centrallongitude"][0] + "E"
    projdef += " +lat_0=" + metadata["centrallatitude"][0] + "N"
    projdef += " +lat_ts=" + metadata["truelatitude"][0]
    # These are hard-coded because the projection definition is missing from the
    # PGM files.
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


def import_mch_gif(filename, product, unit, accutime):
    """Import a 8-bit gif radar reflectivity composite from the MeteoSwiss
    archive.

    Parameters
    ----------

    filename : str
        Name of the file to import.

    product : {"AQC", "CPC", "RZC", "AZC"}
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

    unit : {"mm/h", "mm", "dBZ"}
        the physical unit of the data

    accutime : float
        the accumulation time in minutes of the data

    Returns
    -------

    out : tuple
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
    B = PIL.Image.open(filename)

    if product.lower() in ["azc", "rzc", "precip"]:

        # convert 8-bit GIF colortable to RGB values
        Brgb = B.convert("RGB")

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
        precip = np.zeros(len(Brgb.getdata()))
        for i, dn in enumerate(Brgb.getdata()):
            precip[i] = lut.get(dn, np.nan)

        # convert to original shape
        width, height = B.size
        precip = precip.reshape(height, width)

        # set values outside observational range to NaN,
        # and values in non-precipitating areas to zero.
        precip[precip < 0] = 0
        precip[precip > 9999] = np.nan

    elif product.lower() in ["aqc", "cpc", "acquire ", "combiprecip"]:

        # convert digital numbers to physical values
        B = np.array(B, dtype=int)

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
        precip = lut[B]

    else:
        raise ValueError("unknown product %s" % product)

    metadata["accutime"] = accutime
    metadata["unit"] = unit
    metadata["transform"] = None
    metadata["zerovalue"] = np.nanmin(precip)
    if np.any(precip > np.nanmin(precip)):
        metadata["threshold"] = np.nanmin(precip[precip > np.nanmin(precip)])
    else:
        metadata["threshold"] = np.nan
    metadata["institution"] = "MeteoSwiss"
    metadata["product"] = product
    metadata["zr_a"] = 316.0
    metadata["zr_b"] = 1.5

    return precip, None, metadata


def import_mch_hdf5(filename, **kwargs):
    """Import a precipitation field (and optionally the quality field) from a
    MeteoSwiss HDF5 file conforming to the ODIM specification.

    Parameters
    ----------

    filename : str
        Name of the file to import.

    Other Parameters
    ----------------

    qty : {'RATE', 'ACRR', 'DBZH'}
        The quantity to read from the file. The currently supported identitiers
        are: 'RATE'=instantaneous rain rate (mm/h), 'ACRR'=hourly rainfall
        accumulation (mm) and 'DBZH'=max-reflectivity (dBZ). The default value
        is 'RATE'.

    Returns
    -------

    out : tuple
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

    qty = kwargs.get("qty", "RATE")

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
                        qty_, gain, offset, nodata, undetect = _read_mch_hdf5_what_group(
                            dg[1]["what"]
                        )
                    elif not what_grp_found:
                        raise DataModelError(
                            "Non ODIM compilant file: "
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
    proj4str = where.attrs["projdef"].decode()  # is empty ...

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


def import_mch_metranet(filename, product, unit, accutime):
    """Import a 8-bit bin radar reflectivity composite from the MeteoSwiss
    archive.

    Parameters
    ----------

    filename : str
        Name of the file to import.

    product : {"AQC", "CPC", "RZC", "AZC"}
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

    unit : {"mm/h", "mm", "dBZ"}
        the physical unit of the data

    accutime : float
        the accumulation time in minutes of the data

    Returns
    -------

    out : tuple
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
    if np.isnan(metadata["zerovalue"]):
        metadata["threshold"] = np.nan
    else:
        metadata["threshold"] = np.nanmin(precip[precip > metadata["zerovalue"]])
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

    geodata["yorigin"] = "upper"

    return geodata


def import_opera_hdf5(filename, **kwargs):
    """Import a precipitation field (and optionally the quality field) from an
    OPERA HDF5 file conforming to the ODIM specification.

    Parameters
    ----------

    filename : str
        Name of the file to import.

    Other Parameters
    ----------------

    qty : {'RATE', 'ACRR', 'DBZH'}
        The quantity to read from the file. The currently supported identitiers
        are: 'RATE'=instantaneous rain rate (mm/h), 'ACRR'=hourly rainfall
        accumulation (mm) and 'DBZH'=max-reflectivity (dBZ). The default value
        is 'RATE'.

    Returns
    -------

    out : tuple
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

    qty = kwargs.get("qty", "RATE")

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
                qty_, gain, offset, nodata, undetect = _read_opera_hdf5_what_group(
                    dsg[1]["what"]
                )
                what_grp_found = True

            for dg in dsg[1].items():
                if dg[0][0:4] == "data":
                    # check if the "what" group is in the "data" group
                    if "what" in list(dg[1].keys()):
                        qty_, gain, offset, nodata, undetect = _read_opera_hdf5_what_group(
                            dg[1]["what"]
                        )
                    elif not what_grp_found:
                        raise DataModelError(
                            "Non ODIM compilant file: "
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
                            precip[mask_u] = 0.0
                            precip[mask_n] = np.nan
                        elif qty_.decode() == "QIND":
                            quality = np.empty(arr.shape, dtype=float)
                            quality[mask] = arr[mask]
                            quality[~mask] = np.nan

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

    if np.any(np.isfinite(precip)):
        thr = np.nanmin(precip[precip > np.nanmin(precip)])
    else:
        thr = np.nan

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
        "yorigin": "upper",
        "institution": "Odyssey datacentre",
        "accutime": 15.0,
        "unit": unit,
        "transform": transform,
        "zerovalue": np.nanmin(precip),
        "threshold": thr,
    }

    f.close()

    return precip, quality, metadata


def _read_mch_hdf5_what_group(whatgrp):
    qty = whatgrp.attrs["quantity"] if "quantity" in whatgrp.attrs.keys() else "RATE"
    gain = whatgrp.attrs["gain"] if "gain" in whatgrp.attrs.keys() else 1.0
    offset = whatgrp.attrs["offset"] if "offset" in whatgrp.attrs.keys() else 0.0
    nodata = whatgrp.attrs["nodata"] if "nodata" in whatgrp.attrs.keys() else 0
    undetect = whatgrp.attrs["undetect"] if "undetect" in whatgrp.attrs.keys() else -1.0

    return qty, gain, offset, nodata, undetect


def _read_opera_hdf5_what_group(whatgrp):
    qty = whatgrp.attrs["quantity"]
    gain = whatgrp.attrs["gain"] if "gain" in whatgrp.attrs.keys() else 1.0
    offset = whatgrp.attrs["offset"] if "offset" in whatgrp.attrs.keys() else 0.0
    nodata = whatgrp.attrs["nodata"] if "nodata" in whatgrp.attrs.keys() else np.nan
    undetect = whatgrp.attrs["undetect"] if "undetect" in whatgrp.attrs.keys() else 0.0

    return qty, gain, offset, nodata, undetect


def import_knmi_hdf5(filename, **kwargs):
    """Import a precipitation or reflectivity field (and optionally the quality
    field) from a HDF5 file conforming to the KNMI Data Centre specification.

    Parameters
    ----------

    filename : str
        Name of the file to import.

    Other Parameters
    ----------------

    qty : {'ACRR', 'DBZH'}
        The quantity to read from the file. The currently supported identifiers
        are: 'ACRR'=hourly rainfall accumulation (mm) and 'DBZH'=max-reflectivity
        (dBZ). The default value is 'ACRR'.

    accutime : float
        The accumulation time of the dataset in minutes. A 5 min accumulation
        is used as default, but hourly, daily and monthly accumulations
        are also available.

    pixelsize: float
        The pixel size of a raster cell in meters. The default value for the KNMI
        datasets is 1000 m grid cell size, but datasets with 2400 m pixel size
        are also available.

    Returns
    -------

    out : tuple
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

    ###
    # Options for kwargs.get
    ###

    # The unit in the 2D fields: either hourly rainfall accumulation (ACRR) or
    # reflectivity (DBZH)
    qty = kwargs.get("qty", "ACRR")

    if qty not in ["ACRR", "DBZH"]:
        raise ValueError(
            "unknown quantity %s: the available options are 'ACRR' and 'DBZH' "
        )

    # The time step. Generally, the 5 min data is used, but also hourly, daily
    # and monthly accumulations are present.
    accutime = kwargs.get("accutime", 5.0)
    # The pixel size. Recommended is to use KNMI datasets with 1 km grid cell size.
    # 1.0 or 2.4 km datasets are available - give pixelsize in meters
    pixelsize = kwargs.get("pixelsize", 1000.0)

    ####
    # Precipitation fields
    ####

    f = h5py.File(filename, "r")
    dset = f["image1"]["image_data"]
    precip_intermediate = np.copy(dset)  # copy the content

    # In case R is a rainfall accumulation (ACRR), R is divided by 100.0,
    # because the data is saved as hundreds of mm (so, as integers). 65535 is
    # the no data value. The precision of the data is two decimals (0.01 mm).
    if qty == "ACRR":
        precip = np.where(precip_intermediate == 65535,
                          np.NaN,
                          precip_intermediate / 100.0)

    # In case reflectivities are imported, the no data value is 255. Values are
    # saved as integers. The reflectivities are not directly saved in dBZ, but
    # as: dBZ = 0.5 * pixel_value - 32.0 (this used to be 31.5).
    if qty == "DBZH":
        precip = np.where(precip_intermediate == 255,
                          np.NaN,
                          precip_intermediate * 0.5 - 32.0)

    if precip is None:
        raise IOError("requested quantity not found")

    # TODO: Check if the reflectivity conversion equation is still up to date (unfortunately not well documented)

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
    y2 = min(ll_y, lr_y)
    x2 = max(lr_x, ur_x)
    y1 = max(ul_y, ur_y)

    # Fill in the metadata
    metadata["x1"] = x1 * 1000.0
    metadata["y1"] = y1 * 1000.0
    metadata["x2"] = x2 * 1000.0
    metadata["y2"] = y2 * 1000.0
    metadata["xpixelsize"] = pixelsize
    metadata["ypixelsize"] = pixelsize
    metadata["yorigin"] = "upper"
    metadata["institution"] = "KNMI - Royal Netherlands Meteorological Institute"
    metadata["accutime"] = accutime
    metadata["unit"] = unit
    metadata["transform"] = transform
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = np.nanmin(precip[precip > np.nanmin(precip)])
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6

    f.close()

    return precip, None, metadata
