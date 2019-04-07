"""
pysteps.io.importers
====================

Methods for importing files containing 2d precipitation fields.

The methods in this module implement the following interface::

    import_xxx(filename, optional arguments)

where **xxx** is the name (or abbreviation) of the file format and filename
is the name of the input file.

The output of each method is a three-element tuple containing a two-dimensional
precipitation field, the corresponding quality field and a metadata dictionary.
If the file contains no quality information, the quality field is set to None.
Pixels containing missing data are set to nan.

The metadata dictionary contains the following mandatory key-value pairs:

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
|    timestep      | time step of the input data (minutes)                    |
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

Available Importers
-------------------

.. autosummary::
    :toctree: ../generated/

    import_bom_rf3
    import_fmi_pgm
    import_mch_gif
    import_mch_hdf5
    import_mch_metranet
    import_odim_hdf5
    import_knmi_hdf5
"""


import datetime
import gzip
from matplotlib.pyplot import imread
import numpy as np
import os

from pysteps.exceptions import DataModelError
from pysteps.exceptions import MissingOptionalDependency

try:
    import h5py

    h5py_imported = True
except ImportError:
    h5py_imported = False
try:
    import metranet

    metranet_imported = True
except ImportError:
    metranet_imported = False
try:
    import netCDF4

    netcdf4_imported = True
except ImportError:
    netcdf4_imported = False
try:
    import PIL

    pil_imported = True
except ImportError:
    pil_imported = False
try:
    import pyproj

    pyproj_imported = True
except ImportError:
    pyproj_imported = False


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
    if not netcdf4_imported:
        raise MissingOptionalDependency(
            "netCDF4 package is required to import BoM Rainfields3 products "
            "but it is not installed"
        )

    R = _import_bom_rf3_data(filename)

    geodata = _import_bom_rf3_geodata(filename)
    metadata = geodata
    # TODO(import_bom_rf3): Add missing georeferencing data.

    metadata["transform"] = None
    metadata["zerovalue"] = np.nanmin(R)
    if np.any(np.isfinite(R)):
        metadata["threshold"] = np.nanmin(R[R > np.nanmin(R)])
    else:
        metadata["threshold"] = np.nan

    return R, None, metadata


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

    if "units" in ds_rainfall.variables["x"].ncattrs():
        if getattr(ds_rainfall.variables["x"], "units") == "km":
            geodata["x1"] = xmin * 1000
            geodata["y1"] = ymin * 1000
            geodata["x2"] = xmax * 1000
            geodata["y2"] = ymax * 1000

    xpixelsize = (
        abs(ds_rainfall.variables["x"][1] - ds_rainfall.variables["x"][0]) * 1000.0
    )
    ypixelsize = (
        abs(ds_rainfall.variables["y"][1] - ds_rainfall.variables["y"][0]) * 1000
    )
    geodata["xpixelsize"] = xpixelsize
    geodata["ypixelsize"] = ypixelsize
    geodata["yorigin"] = "upper"  # TODO(_import_bom_rf3_geodata): check this

    # get the accumulation period
    valid_time = None

    if "valid_time" in ds_rainfall.variables.keys():
        valid_time = datetime.datetime.utcfromtimestamp(
            ds_rainfall.variables["valid_time"][:]
        )
    
    start_time = None
    if "start_time" in ds_rainfall.variables.keys():
        start_time = datetime.datetime.utcfromtimestamp(
            ds_rainfall.variables["start_time"][:]
        )
    
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
    if not pyproj_imported:
        raise MissingOptionalDependency(
            "pyproj package is required to import "
            "FMI's radar reflectivity composite "
            "but it is not installed"
        )

    gzipped = kwargs.get("gzipped", False)

    pgm_metadata = _import_fmi_pgm_metadata(filename, gzipped=gzipped)

    if gzipped == False:
        R = imread(filename)
    else:
        R = imread(gzip.open(filename, "r"))
    geodata = _import_fmi_pgm_geodata(pgm_metadata)

    MASK = R == pgm_metadata["missingval"]
    R = R.astype(float)
    R[MASK] = np.nan
    R = (R - 64.0) / 2.0

    metadata = geodata
    metadata["institution"] = "Finnish Meteorological Institute"
    metadata["accutime"] = 5.0
    metadata["unit"] = "dBZ"
    metadata["transform"] = "dB"
    metadata["zerovalue"] = np.nanmin(R)
    if np.any(np.isfinite(R)):
        metadata["threshold"] = np.nanmin(R[R > np.nanmin(R)])
    else:
        metadata["threshold"] = np.nan

    return R, None, metadata


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

    l = f.readline()
    while not l.startswith(b"#"):
        l = f.readline()
    while l.startswith(b"#"):
        x = l.decode()
        x = x[1:].strip().split(" ")
        if len(x) >= 2:
            k = x[0]
            v = x[1:]
            metadata[k] = v
        else:
            l = f.readline()
            continue
        l = f.readline()
    l = f.readline().decode()
    metadata["missingval"] = int(l)
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
    if not pil_imported:
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
        R = np.zeros(len(Brgb.getdata()))
        for i, dn in enumerate(Brgb.getdata()):
            R[i] = lut.get(dn, np.nan)

        # convert to original shape
        width, height = B.size
        R = R.reshape(height, width)

        # set values outside observational range to NaN,
        # and values in non-precipitating areas to zero.
        R[R < 0] = 0
        R[R > 9999] = np.nan

    elif product.lower() in ["aqc", "cpc", "acquire ", "combiprecip"]:

        # convert digital numbers to physical values
        B = np.array(B, dtype=int)

        # build lookup table [mm/5min]
        lut = np.zeros(256)
        A = 316.0
        b = 1.5
        for i in range(256):
            if (i < 2) or (i > 250 and i < 255):
                lut[i] = 0.0
            elif i == 255:
                lut[i] = np.nan
            else:
                lut[i] = (10.0 ** ((i - 71.5) / 20.0) / A) ** (1.0 / b)

        # apply lookup table
        R = lut[B]

    else:
        raise ValueError("unknown product %s" % product)

    metadata["accutime"] = accutime
    metadata["unit"] = unit
    metadata["transform"] = None
    metadata["zerovalue"] = np.nanmin(R)
    if np.any(R > np.nanmin(R)):
        metadata["threshold"] = np.nanmin(R[R > np.nanmin(R)])
    else:
        metadata["threshold"] = np.nan
    metadata["institution"] = "MeteoSwiss"
    metadata["product"] = product

    return R, None, metadata


def import_mch_hdf5(filename, **kwargs):
    """Import a precipitation field (and optionally the quality field) from a HDF5
    file conforming to the ODIM specification.

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
    if not h5py_imported:
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

    R = None
    Q = None

    for dsg in f.items():
        if dsg[0][0:7] == "dataset":
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
                        ARR = dg[1]["data"][...]
                        MASK_N = ARR == nodata
                        MASK_U = ARR == undetect
                        MASK = np.logical_and(~MASK_U, ~MASK_N)

                        if qty_.decode() == qty:
                            R = np.empty(ARR.shape)
                            R[MASK] = ARR[MASK] * gain + offset
                            R[MASK_U] = np.nan
                            R[MASK_N] = np.nan
                        elif qty_.decode() == "QIND":
                            Q = np.empty(ARR.shape, dtype=float)
                            Q[MASK] = ARR[MASK]
                            Q[~MASK] = np.nan

    if R is None:
        raise IOError("requested quantity %s not found" % qty)

    where = f["where"]
    proj4str = where.attrs["projdef"].decode()  # is emtpy ...

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

    if np.any(np.isfinite(R)):
        thr = np.nanmin(R[R > np.nanmin(R)])
    else:
        thr = np.nan

    metadata.update(
        {
            "yorigin": "upper",
            "institution": "MeteoSwiss",
            "accutime": 5.0,
            "unit": unit,
            "transform": transform,
            "zerovalue": np.nanmin(R),
            "threshold": thr,
        }
    )

    f.close()

    return R, Q, metadata


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
    if not metranet_imported:
        raise MissingOptionalDependency(
            "metranet package needed for importing MeteoSwiss "
            "radar composites but it is not installed"
        )

    ret = metranet.read_file(filename, physic_value=True, verbose=False)
    R = ret.data

    geodata = _import_mch_geodata()

    # read metranet
    metadata = geodata
    metadata["institution"] = "MeteoSwiss"
    metadata["accutime"] = accutime
    metadata["unit"] = unit
    metadata["transform"] = None
    metadata["zerovalue"] = np.nanmin(R)
    if np.isnan(metadata["zerovalue"]):
        metadata["threshold"] = np.nan
    else:
        metadata["threshold"] = np.nanmin(R[R > metadata["zerovalue"]])

    return R, None, metadata


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


def import_odim_hdf5(filename, **kwargs):
    """Import a precipitation field (and optionally the quality field) from a
    HDF5 file conforming to the ODIM specification.

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
    if not h5py_imported:
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

    R = None
    Q = None

    for dsg in f.items():
        if dsg[0][0:7] == "dataset":
            what_grp_found = False
            # check if the "what" group is in the "dataset" group
            if "what" in list(dsg[1].keys()):
                qty_, gain, offset, nodata, undetect = _read_odim_hdf5_what_group(
                    dsg[1]["what"]
                )
                what_grp_found = True

            for dg in dsg[1].items():
                if dg[0][0:4] == "data":
                    # check if the "what" group is in the "data" group
                    if "what" in list(dg[1].keys()):
                        qty_, gain, offset, nodata, undetect = _read_odim_hdf5_what_group(
                            dg[1]["what"]
                        )
                    elif not what_grp_found:
                        raise DataModelError(
                            "Non ODIM compilant file: "
                            "no what group found from {} "
                            "or its subgroups".format(dg[0])
                        )

                    if qty_.decode() in [qty, "QIND"]:
                        ARR = dg[1]["data"][...]
                        MASK_N = ARR == nodata
                        MASK_U = ARR == undetect
                        MASK = np.logical_and(~MASK_U, ~MASK_N)

                        if qty_.decode() == qty:
                            R = np.empty(ARR.shape)
                            R[MASK] = ARR[MASK] * gain + offset
                            R[MASK_U] = 0.0
                            R[MASK_N] = np.nan
                        elif qty_.decode() == "QIND":
                            Q = np.empty(ARR.shape, dtype=float)
                            Q[MASK] = ARR[MASK]
                            Q[~MASK] = np.nan

    if R is None:
        raise IOError("requested quantity %s not found" % qty)

    where = f["where"]
    proj4str = where.attrs["projdef"].decode()
    pr = pyproj.Proj(proj4str)

    LL_lat = where.attrs["LL_lat"]
    LL_lon = where.attrs["LL_lon"]
    UR_lat = where.attrs["UR_lat"]
    UR_lon = where.attrs["UR_lon"]
    if (
        "LR_lat" in where.attrs.keys()
        and "LR_lon" in where.attrs.keys()
        and "UL_lat" in where.attrs.keys()
        and "UL_lon" in where.attrs.keys()
    ):
        LR_lat = float(where.attrs["LR_lat"])
        LR_lon = float(where.attrs["LR_lon"])
        UL_lat = float(where.attrs["UL_lat"])
        UL_lon = float(where.attrs["UL_lon"])
        full_cornerpts = True
    else:
        full_cornerpts = False

    LL_x, LL_y = pr(LL_lon, LL_lat)
    UR_x, UR_y = pr(UR_lon, UR_lat)
    if full_cornerpts:
        LR_x, LR_y = pr(LR_lon, LR_lat)
        UL_x, UL_y = pr(UL_lon, UL_lat)
        x1 = min(LL_x, UL_x)
        y1 = min(LL_y, LR_y)
        x2 = max(LR_x, UR_x)
        y2 = max(UL_y, UR_y)
    else:
        x1 = LL_x
        y1 = LL_y
        x2 = UR_x
        y2 = UR_y

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

    if np.any(np.isfinite(R)):
        thr = np.nanmin(R[R > np.nanmin(R)])
    else:
        thr = np.nan

    metadata = {
        "projection": proj4str,
        "ll_lon": LL_lon,
        "ll_lat": LL_lat,
        "ur_lon": UR_lon,
        "ur_lat": UR_lat,
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
        "zerovalue": np.nanmin(R),
        "threshold": thr,
    }

    f.close()

    return R, Q, metadata


def _read_mch_hdf5_what_group(whatgrp):

    qty = whatgrp.attrs["quantity"] if "quantity" in whatgrp.attrs.keys() else "RATE"
    gain = whatgrp.attrs["gain"] if "gain" in whatgrp.attrs.keys() else 1.0
    offset = whatgrp.attrs["offset"] if "offset" in whatgrp.attrs.keys() else 0.0
    nodata = whatgrp.attrs["nodata"] if "nodata" in whatgrp.attrs.keys() else 0
    undetect = whatgrp.attrs["undetect"] if "undetect" in whatgrp.attrs.keys() else -1.0

    return qty, gain, offset, nodata, undetect


def _read_odim_hdf5_what_group(whatgrp):

    qty = whatgrp.attrs["quantity"]
    gain = whatgrp.attrs["gain"] if "gain" in whatgrp.attrs.keys() else 1.0
    offset = whatgrp.attrs["offset"] if "offset" in whatgrp.attrs.keys() else 0.0
    nodata = whatgrp.attrs["nodata"] if "nodata" in whatgrp.attrs.keys() else np.nan
    undetect = whatgrp.attrs["undetect"] if "undetect" in whatgrp.attrs.keys() else 0.0

    return qty, gain, offset, nodata, undetect


def import_knmi_hdf5(filename, **kwargs):
    """Import a precipitation field (and optionally the quality field) from a
    HDF5 file conforming to the KNMI Data Centre specification.

    Parameters
    ----------
    filename : str
        Name of the file to import.

    Other Parameters
    ----------------
    accutime : float
        The accumulation time of the dataset in minutes.
    pixelsize: float
        The pixelsize of a raster cell in meters.

    Returns
    -------
    out : tuple
        A three-element tuple containing precipitation accumulation of the KNMI
        product, the associated quality field and metadata. The quality
        field is currently set to None.

    """

    # TODO: Add quality field.

    if not h5py_imported:
        raise Exception("h5py not imported")

    # Generally, the 5 min. data is used, but also hourly, daily and monthly
    # accumulations are present.
    accutime = kwargs.get("accutime", 5.0)
    pixelsize = kwargs.get(
        "pixelsize", 1000.0
    )  # 1.0 or 2.4 km datasets are available - give pixelsize in meters

    ####
    # Precipitation fields
    ####

    f = h5py.File(filename, "r")
    dset = f["image1"]["image_data"]
    R_intermediate = np.copy(dset)  # copy the content
    R = np.where(R_intermediate == 65535, np.NaN, R_intermediate / 100.0)
    # R is divided by 100.0, because the data is saved as hundreds of mm (so, as integers)
    # 65535 is the no data value
    # Precision of the data is two decimals (0.01 mm).

    if R is None:
        raise IOError("requested quantity [mm] not found")

    ####
    # Meta data
    ####

    metadata = {}

    # The 'where' group of mch- and Opera-data, is called 'geographic' in the
    # KNMI data.
    geographic = f["geographic"]
    proj4str = geographic["map_projection"].attrs["projection_proj4_params"].decode()
    pr = pyproj.Proj(proj4str)
    metadata["projection"] = proj4str

    # Get coordinates
    latlon_corners = geographic.attrs["geo_product_corners"]
    LL_lat = latlon_corners[1]
    LL_lon = latlon_corners[0]
    UR_lat = latlon_corners[5]
    UR_lon = latlon_corners[4]
    LR_lat = latlon_corners[7]
    LR_lon = latlon_corners[6]
    UL_lat = latlon_corners[3]
    UL_lon = latlon_corners[2]

    LL_x, LL_y = pr(LL_lon, LL_lat)
    UR_x, UR_y = pr(UR_lon, UR_lat)
    LR_x, LR_y = pr(LR_lon, LR_lat)
    UL_x, UL_y = pr(UL_lon, UL_lat)
    x1 = min(LL_x, UL_x)
    y2 = min(LL_y, LR_y)
    x2 = max(LR_x, UR_x)
    y1 = max(UL_y, UR_y)

    # Fill in the metadata
    metadata["x1"] = x1 * 1000.
    metadata["y1"] = y1 * 1000.
    metadata["x2"] = x2 * 1000.
    metadata["y2"] = y2 * 1000.

    metadata["xpixelsize"] = pixelsize
    metadata["ypixelsize"] = pixelsize

    metadata["yorigin"] = "upper"
    metadata["institution"] = "KNMI - Royal Netherlands Meteorological Institute"
    metadata["accutime"] = accutime
    metadata["unit"] = "mm"
    metadata["transform"] = None
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = np.nanmin(R[R > np.nanmin(R)])

    f.close()

    return R, None, metadata
