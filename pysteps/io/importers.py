"""Methods for importing files containing 2d precipitation fields.

The methods in this module implement the following interface:

  import_xxx(filename, optional arguments)

where xxx is the name (or abbreviation) of the file format and filename is the
name of the input file.

The output of each method is a three-element tuple containing the two-dimensional
precipitation field, the corresponding quality field and a metadata dictionary.
If the file contains no quality information, the quality field is set to None.
Pixels containing missing data are set to nan.

The metadata dictionary contains the following mandatory key-value pairs:
    projection   PROJ.4-compatible projection definition
    x1           x-coordinate of the lower-left corner of the data raster
    y1           y-coordinate of the lower-left corner of the data raster
    x2           x-coordinate of the upper-right corner of the data raster
    y2           y-coordinate of the upper-right corner of the data raster
    xpixelsize   grid resolution in x-direction (meters)
    ypixelsize   grid resolution in y-direction (meters)
    yorigin      a string specifying the location of the first element in
                 the data raster w.r.t. y-axis:
                 'upper' = upper border
                 'lower' = lower border
    institution  name of the institution who provides the data
    timestep     time step of the input data (minutes)
    unit         the unit of the data: 'mm/h', 'mm' or 'dBZ'
"""

import datetime
import gzip
from matplotlib.pyplot import imread
import numpy as np
import os
try:
    import h5py
    h5py_imported = True
except ImportError:
    h5py_imported = False
try:
    from netCDF4 import Dataset
    netcdf4_imported = True
except ImportError:
    netcdf4_imported = False
try:
    from PIL import Image
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
        raise Exception("netCDF4 not imported")

    R = _import_bom_rf3_data(filename)
    metadata = _import_bom_rf3_metadata(filename)
    projdef = _import_bom_rf3_geodata(filename)
    metadata["projection"] = projdef
    # TODO: Add missing georeferencing data.

    return R, None, metadata

def _import_bom_rf3_data(filename):
    print(filename)
    ds_rainfall = Dataset(filename)
    if ('precipitation' in ds_rainfall.variables.keys()):
        precipitation = ds_rainfall.variables['precipitation'][:]
        # estimate time-step to transform from mm to mm/h
        if ('valid_time' in ds_rainfall.variables.keys()):
            valid_time = datetime.datetime.utcfromtimestamp(
                ds_rainfall.variables['valid_time'][:])
        else:
            valid_time = None
        if ('start_time' in ds_rainfall.variables.keys()):
            start_time = datetime.datetime.utcfromtimestamp(
                ds_rainfall.variables['start_time'][:])
        else:
            start_time = None
        if start_time is not None:
            if valid_time is not None:
                time_step = (valid_time-start_time).seconds//60
        if time_step:
            factor_rain = 60./time_step
            precipitation = precipitation*factor_rain
    else:
        precipitation = None
    ds_rainfall.close()

    return precipitation

def _import_bom_rf3_geodata(filename):
    ds_rainfall = Dataset(filename)
    if ('proj' in ds_rainfall.variables.keys()):
        projection = ds_rainfall.variables['proj']
        if (getattr(projection, 'grid_mapping_name') ==
                "albers_conical_equal_area"):
            projdef = "+proj=aea "
            lon_0 = getattr(projection,
                            'longitude_of_central_meridian')
            projdef += " +lon_0=" + str(lon_0)
            lat_0 = getattr(projection,
                            'latitude_of_projection_origin')
            projdef += " +lat_0=" + str(lat_0)
            standard_parallels = getattr(projection,
                                         'standard_parallel')
            projdef += " +lat_1=" + str(standard_parallels[0])
            projdef += " +lat_2=" + str(standard_parallels[1])
        else:
            projdef = None
    ds_rainfall.close()

    return projdef

def _import_bom_rf3_metadata(filename):
    metadata = {}
    metadata["institution"] = "Bureau of Meteorology"
    metadata["accutime"]    = 6
    metadata["unit"]        = "mm/h"

    return metadata

def import_fmi_pgm(filename, **kwargs):
    """Import a 8-bit PGM radar reflectivity composite from the FMI archive.

    Parameters
    ----------
    filename : str
        Name of the file to import.

    Optional kwargs
    ---------------
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
        raise Exception("pyproj not imported")

    gzipped = kwargs.get("gzipped", False)

    pgm_metadata = _import_fmi_pgm_metadata(filename, gzipped=gzipped)

    if gzipped == False:
        R = imread(filename)
    else:
        R = imread(gzip.open(filename, 'r'))
    geodata = _import_fmi_pgm_geodata(pgm_metadata)

    MASK = R == pgm_metadata["missingval"]
    R = R.astype(float)
    R[MASK] = np.nan
    R = (R - 64.0) / 2.0

    metadata = geodata
    metadata["institution"] = "Finnish Meteorological Institute"
    metadata["accutime"]    = 5
    metadata["unit"]        = "dBZ"

    return R,None,metadata

def _import_fmi_pgm_geodata(metadata):
    geodata = {}

    projdef = ""

    if metadata["type"][0] != "stereographic":
        raise ValueError("unknown projection %s" % metadata["type"][0])
    projdef += "+proj=stere "
    projdef += " +lon_0=" + metadata["centrallongitude"][0] + 'E'
    projdef += " +lat_0=" + metadata["centrallatitude"][0] + 'N'
    projdef += " +lat_ts=" + metadata["truelatitude"][0]
    # These are hard-coded because the projection definition is missing from the
    # PGM files.
    projdef += " +a=6371288"
    projdef += " +x_0=380886.310"
    projdef += " +y_0=3395677.920"
    projdef += " +no_defs"
    #
    geodata["projection"] = projdef

    ll_lon,ll_lat = [float(v) for v in metadata["bottomleft"]]
    ur_lon,ur_lat = [float(v) for v in metadata["topright"]]

    pr = pyproj.Proj(projdef)
    x1,y1 = pr(ll_lon, ll_lat)
    x2,y2 = pr(ur_lon, ur_lat)

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

    if gzipped == False:
        f = open(filename, 'r')
    else:
        f = gzip.open(filename, 'r')

    l = f.readline().decode()
    while l[0] != '#':
        l = f.readline().decode()
    while l[0] == '#':
        x = l[1:].strip().split(' ')
        if len(x) >= 2:
            k = x[0]
            v = x[1:]
            metadata[k] = v
        else:
            l = f.readline().decode()
            continue
        l = f.readline().decode()
    l = f.readline().decode()
    metadata["missingval"] = int(l)
    f.close()

    return metadata

def import_mch_gif(filename, **kwargs):
    """Import a 8-bit gif radar reflectivity composite from the MeteoSwiss
    archive.

    Parameters
    ----------
    filename : str
        Name of the file to import.

    Optional kwargs
    ---------------
    product : string
        The name of the MeteoSwiss QPE product.
		Options:
			- "AQC" (AQUIRE)
			- "RZC" (PRECIP)

    Returns
    -------
    out : tuple
        A three-element tuple containing the precipitation field in mm/h imported
        from a MeteoSwiss gif file and the associated quality field and metadata.
        The quality field is currently set to None.
    """
    if not pil_imported:
        raise Exception("PIL not imported")

    product = kwargs.get("product", "AQC")

    geodata = _import_mch_gif_geodata()

    # import gif file
    B = Image.open(filename)
    
    # convert digital numbers to physical values
    if product == "AQC":
        B = np.array(B, dtype=int)

        # build lookup table [mm/5min]
        lut = np.zeros(256)
        A = 316.0; b = 1.5
        for i in range(256):
            if (i < 2) or (i > 250 and i < 255):
                lut[i] = 0.0
            elif (i == 255):
                lut[i] = np.nan
            else:
                lut[i] = (10.**((i - 71.2)/20.0)/A)**(1.0/b)

        # apply lookup table [mm/h]
        R = lut[B]*12

    elif product == "RZC":

        # convert 8-bit GIF colortable to RGB values
        Brgb = B.convert('RGB')

        # load lookup table
        lut_filename = os.path.join(os.path.dirname(__file__), "mch_lut_8bit_Metranet_v103.txt")
        lut = np.genfromtxt(lut_filename, skip_header=1)
        lut = dict(zip(zip(lut[:, 1], lut[:,2], lut[:,3]), lut[:,-1]))

        # apply lookup table conversion
        R = np.zeros(len(Brgb.getdata()))
        for i,dn in enumerate(Brgb.getdata()):
            R[i] = lut.get(dn,-1.0)
            
        # convert to original shape
        width, height = B.size
        R = R.reshape(height,width)


        # set values outside observational range to NaN,
        # and values in non-precipitating areas to zero.
        R[R<0] = 0
        R[R>1000] = np.nan
        
    else:
        raise ValueError("unknown product %s" % product)

    metadata = geodata
    metadata["institution"] = "MeteoSwiss"
    metadata["accutime"]    = 5
    metadata["unit"]        = "mm/h"

    return R,None,metadata

def _import_mch_gif_geodata():
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

    geodata["x1"] = 255000
    geodata["y1"] = -160000
    geodata["x2"] = 965000
    geodata["y2"] = 480000

    geodata["xpixelsize"] = 1000
    geodata["ypixelsize"] = 1000

    geodata["yorigin"] = "upper"

    return geodata

def import_odim_hdf5(filename, **kwargs):
    """Read a precipitation field (and optionally the quality field) from a HDF5 
    file conforming to the ODIM specification.

    Parameters
    ----------
    filename : str
        Name of the file to import.

    Optional kwargs
    ---------------
    qty : str
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
        raise Exception("h5py not imported")

    qty = kwargs.get("qty", "RATE")

    if qty not in ["ACRR", "DBZH", "RATE"]:
        raise ValueError("unknown quantity %s: the available options are 'ACRR', 'DBZH' and 'RATE'")

    f = h5py.File(filename, 'r')

    R = None
    Q = None

    for dsg in f.items():
        if dsg[0][0:7] == "dataset":
            what_grp_found = False
            # check if the "what" group is in the "dataset" group
            if "what" in list(dsg[1].keys()):
                qty_,gain,offset,nodata,undetect = _read_odimhdf5_what_group(dsg[1]["what"])
                what_grp_found = True

            for dg in dsg[1].items():
                if dg[0][0:4] == "data":
                    # check if the "what" group is in the "data" group
                    if "what" in list(dg[1].keys()):
                        qty_,gain,offset,nodata,undetect = _read_h5_what_group(dg[1]["what"])
                    elif what_grp_found == False:
                        raise Exception("no what group found from %s or its subgroups" % dg[0])

                    if qty_.decode() in [qty, "QIND"]:
                        ARR = dg[1]["data"][...]
                        MASK_N = ARR == nodata
                        MASK_U = ARR == undetect
                        MASK = np.logical_and(~MASK_U, ~MASK_N)

                        if qty_.decode() == qty:
                            R = np.empty(ARR.shape)
                            R[MASK]   = ARR[MASK] * gain + offset
                            R[MASK_U] = 0.0
                            R[MASK_N] = np.nan
                        elif qty_.decode() == "QIND":
                            Q = np.empty(ARR.shape, dtype=float)
                            Q[MASK]  = ARR[MASK]
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
    if "LR_lat" in where.attrs.keys() and "LR_lon" in where.attrs.keys() and \
        "UL_lat" in where.attrs.keys() and "UL_lon" in where.attrs.keys():
        LR_lat = float(where.attrs["LR_lat"])
        LR_lon = float(where.attrs["LR_lon"])
        UL_lat = float(where.attrs["UL_lat"])
        UL_lon = float(where.attrs["UL_lon"])
        full_cornerpts = True
    else:
        full_cornerpts = False

    LL_x,LL_y = pr(LL_lon, LL_lat)
    UR_x,UR_y = pr(UR_lon, UR_lat)
    if full_cornerpts:
        LR_x,LR_y = pr(LR_lon, LR_lat)
        UL_x,UL_y = pr(UL_lon, UL_lat)
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
    elif qty == "DBZH":
        unit = "dBZ"
    else:
        unit = "mm/h"

    metadata = {"projection":proj4str,
                "ll_lon":LL_lon,
                "ll_lat":LL_lat,
                "ur_lon":UR_lon,
                "ur_lat":UR_lat,
                "x1":x1,
                "y1":y1,
                "x2":x2,
                "y2":y2,
                "xpixelsize":xpixelsize,
                "ypixelsize":ypixelsize,
                "institution": "Odyssey datacentre",
                "accutime":15,
                "unit":unit}

    f.close()

    return R,Q,metadata

def _read_odim_hdf5_what_group(whatgrp):
    qty      = whatgrp.attrs["quantity"]
    gain     = whatgrp.attrs["gain"]     if "gain" in whatgrp.attrs.keys() else 1.0
    offset   = whatgrp.attrs["offset"]   if "offset" in whatgrp.attrs.keys() else 0.0
    nodata   = whatgrp.attrs["nodata"]   if "nodata" in whatgrp.attrs.keys() else nan
    undetect = whatgrp.attrs["undetect"] if "undetect" in whatgrp.attrs.keys() else 0.0

    return qty,gain,offset,nodata,undetect

