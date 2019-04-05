"""
pysteps.io.nowcast_importers
============================

Methods for importing nowcast files.

The methods in this module implement the following interface::

  import_xxx(filename, optional arguments)

where xxx is the name (or abbreviation) of the file format and filename is the
name of the input file.

The output of each method is a two-element tuple containing the nowcast array
and a metadata dictionary.

The metadata dictionary contains the following mandatory key-value pairs:

.. tabularcolumns:: |p{2cm}|L|

+-------------------+----------------------------------------------------------+
|       Key         |                Value                                     |
+===================+==========================================================+
|    projection     | PROJ.4-compatible projection definition                  |
+-------------------+----------------------------------------------------------+
|    x1             | x-coordinate of the lower-left corner of the data raster |
|                   | (meters)                                                 |
+-------------------+----------------------------------------------------------+
|    y1             | y-coordinate of the lower-left corner of the data raster |
|                   | (meters)                                                 |
+-------------------+----------------------------------------------------------+
|    x2             | x-coordinate of the upper-right corner of the data raster|
|                   | (meters)                                                 |
+-------------------+----------------------------------------------------------+
|    y2             | y-coordinate of the upper-right corner of the data raster|
|                   | (meters)                                                 |
+-------------------+----------------------------------------------------------+
|    xpixelsize     | grid resolution in x-direction (meters)                  |
+-------------------+----------------------------------------------------------+
|    ypixelsize     | grid resolution in y-direction (meters)                  |
+-------------------+----------------------------------------------------------+
|    yorigin        | a string specifying the location of the first element in |
|                   | the data raster w.r.t. y-axis:                           |
|                   | 'upper' = upper border                                   |
|                   | 'lower' = lower border                                   |
+-------------------+----------------------------------------------------------+
|    institution    | name of the institution who provides the data            |
+-------------------+----------------------------------------------------------+
|    timestep       | time step of the input data (minutes)                    |
+-------------------+----------------------------------------------------------+
|    unit           | the physical unit of the data: 'mm/h', 'mm' or 'dBZ'     |
+-------------------+----------------------------------------------------------+
|    transform      | the transformation of the data: None, 'dB', 'Box-Cox' or |
|                   | others                                                   |
+-------------------+----------------------------------------------------------+
|    accutime       | the accumulation time in minutes of the data, float      |
+-------------------+----------------------------------------------------------+
|    threshold      | the rain/no rain threshold with the same unit,           |
|                   | transformation and accutime of the data.                 |
+-------------------+----------------------------------------------------------+
|    zerovalue      | it is the value assigned to the no rain pixels with the  |
|                   | same unit, transformation and accutime of the data.      |
+-------------------+----------------------------------------------------------+

Available Nowcast Importers
---------------------------

.. autosummary::
    :toctree: ../generated/

    import_netcdf_pysteps
"""

import numpy as np

from pysteps.exceptions import MissingOptionalDependency, DataModelError

try:
    import netCDF4
    netcdf4_imported = True
except ImportError:
    netcdf4_imported = False

def import_netcdf_pysteps(filename, **kwargs):
    """Read a nowcast or a nowcast ensemble from a NetCDF file conforming to the
    CF 1.7 specification."""
    if not netcdf4_imported:
        raise MissingOptionalDependency(
            "netCDF4 package is required to import pysteps netcdf "
            "nowcasts but it is not installed")
    try:
        ds = netCDF4.Dataset(filename, 'r')

        var_names = list(ds.variables.keys())

        if "precip_intensity" in var_names:
            R         = ds.variables["precip_intensity"]
            unit      = "mm/h" 
            accutime  = None
            transform = None
        elif "precip_accum" in var_names:
            R         = ds.variables["precip_accum"]
            unit      = "mm"
            accutime  = None
            transform = None
        elif "hourly_precip_accum" in var_names:
            R         = ds.variables["hourly_precip_accum"]
            unit      = "mm"
            accutime  = 60.
            transform = None
        elif "reflectivity" in var_names:
            R         = ds.variables["reflectivity"]
            unit      = "dBZ"
            accutime  = None
            transform = "dB"
        else:
            raise DataModelError(
                "Non CF compilant file: "
                "the netCDF file does not contain any supported variable name.\n"
                "Supported names: 'precip_intensity', 'hourly_precip_accum', "
                "or 'reflectivity'\n"
                "file: "+filename)

        R = R[...].squeeze().astype(float)

        metadata = {}
        
        time_var = ds.variables['time']
        leadtimes = time_var[:]/60. # minutes leadtime
        metadata["leadtimes"] = leadtimes
        timestamps = netCDF4.num2date(time_var[:], time_var.units)
        metadata["timestamps"] = timestamps

        projdef = ""
        if "polar_stereographic" in var_names:
            vn = "polar_stereographic"

            attr_dict = {}
            for attr_name in ds.variables[vn].ncattrs():
                attr_dict[attr_name] = ds[vn].getncattr(attr_name)

            proj_str = _convert_grid_mapping_to_proj4(attr_dict)
            metadata["projection"] = proj_str
            
        # geodata
        metadata["xpixelsize"] = abs(ds.variables['xc'][1] - ds.variables['xc'][0])
        metadata["ypixelsize"] = abs(ds.variables['yc'][1] - ds.variables['yc'][0])
        
        xmin = np.min(ds.variables['xc']) - 0.5*metadata["xpixelsize"]
        xmax = np.max(ds.variables['xc']) + 0.5*metadata["xpixelsize"]
        ymin = np.min(ds.variables['yc']) - 0.5*metadata["ypixelsize"]
        ymax = np.max(ds.variables['yc']) + 0.5*metadata["ypixelsize"]

        # TODO: this is only a quick solution
        metadata["x1"] = xmin
        metadata["y1"] = ymin
        metadata["x2"] = xmax
        metadata["y2"] = ymax

        metadata["yorigin"] = "upper" # TODO: check this

        # TODO: Read the metadata to the dictionary.
        if accutime is None:
            accutime = leadtimes[1] - leadtimes[0]
        metadata["accutime"] = accutime
        metadata["unit"] = unit
        metadata["transform"] = transform
        metadata["zerovalue"] = np.nanmin(R)
        metadata["threshold"] = np.nanmin(R[R>np.nanmin(R)])

        ds.close()

        return R, metadata
    except Exception as er:
        print('There was an error processing the file', er)
        return None, None


def _convert_grid_mapping_to_proj4(grid_mapping):
    gm_keys = list(grid_mapping.keys())

    # TODO: implement more projection types here
    if grid_mapping["grid_mapping_name"] == "polar_stereographic":
        proj_str = "+proj=stere"
        proj_str += " +lon_0=%s" % grid_mapping["straight_vertical_longitude_from_pole"]
        proj_str += " +lat_0=%s" % grid_mapping["latitude_of_projection_origin"]
        if "standard_parallel" in gm_keys:
            proj_str += " +lat_ts=%s" % grid_mapping["standard_parallel"]
        if "scale_factor_at_projection_origin" in gm_keys:
            proj_str += " +k_0=%s" % grid_mapping["scale_factor_at_projection_origin"]
        proj_str += " +x_0=%s" % grid_mapping["false_easting"]
        proj_str += " +y_0=%s" % grid_mapping["false_northing"]

        return proj_str
    else:
        return None
        
def import_netcdf_cosmo1(filename, timebounds=None):
    """Read a COSMO-1 forecast from a NetCDF file conforming to the
    CF 1.7 specification."""   
    if not netcdf4_imported:
        raise Exception("netCDF4 not imported")

    ds = netCDF4.Dataset(filename, 'r')

    var_names = list(ds.variables.keys())

    metadata = {}

    time_var = ds.variables['time']
    if timebounds is None:
        timebounds = np.ones(len(time_var[:]), dtype=bool)
    leadtimes = np.array(time_var[timebounds])/60. # minutes leadtime
    timestamps = np.array([netCDF4.num2date(time,time_var.units) for time in time_var[timebounds]])
    metadata["leadtimes"] = leadtimes
    metadata["timestamps"] = timestamps
    f = np.vectorize(lambda x: x.replace(microsecond=0))
    metadata["timestamps"] = f(metadata["timestamps"])

    no_data = ds.variables[var_names[-1]]._FillValue 
    R = ds.variables[var_names[-1]][timebounds, :, :]
    R = R[...].squeeze().astype(float)
    R=R[:,::-1,:] # flip upside-down
    R[R==no_data] = np.nan

    # TODO: Read the metadata to the dictionary.
    time_step = (timestamps[1] - timestamps[0]).total_seconds()/60
    metadata["accutime"] = time_step
    metadata["unit"] = "mm"
    metadata["transform"] = None
    metadata["zerovalue"] = np.nanmin(R)
    if np.any(np.isfinite(R)):
        metadata["threshold"] = np.nanmin(R[R > np.nanmin(R)])
    else:
        metadata["threshold"] = np.nan

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
    metadata["projection"] = projdef

    metadata["x1"] = 255000.
    metadata["y1"] = -160000.
    metadata["x2"] = 965000.
    metadata["y2"] = 480000.

    metadata["xpixelsize"] = 1000.
    metadata["ypixelsize"] = 1000.

    metadata["yorigin"] = "upper"

    ds.close()

    return R,metadata


def import_netcdf_cosmoe(filename):
    """Read a COSMO-E ensemble forecast from a NetCDF file conforming to the
    CF 1.7 specification."""   
    if not netcdf4_imported:
        raise Exception("netCDF4 not imported")

    ds = netCDF4.Dataset(filename, 'r')

    var_names = list(ds.variables.keys())

    R = ds.variables[var_names[-1]] 
    no_data = R._FillValue 
    R = R[...].squeeze().astype(float)
    R=R[:,:,::-1,:] # flip upside-down
    R[R==no_data] = np.nan

    metadata = {}

    time_var = ds.variables['time']
    leadtimes = np.array(time_var[:])/60. # minutes leadtime
    metadata["leadtimes"] = np.array(leadtimes)
    timestamps = [netCDF4.num2date(time,time_var.units) for time in time_var[:]]
    metadata["timestamps"] = np.array(timestamps)

    # remove milliseconds
    f = np.vectorize(lambda x: x.replace(microsecond=0))
    metadata["timestamps"] = f(metadata["timestamps"])

    # swpax axes: [time, eps, y, x] -> [eps, time, y, x]
    R = np.swapaxes(R, 0, 1)

    # TODO: Read the metadata to the dictionary.
    time_step = (timestamps[1] - timestamps[0]).total_seconds()/60
    metadata["accutime"] = time_step
    metadata["unit"] = "mm"
    metadata["transform"] = None
    metadata["zerovalue"] = np.nanmin(R)
    if np.any(np.isfinite(R)):
        metadata["threshold"] = np.nanmin(R[R > np.nanmin(R)])
    else:
        metadata["threshold"] = np.nan
    metadata["eps"] = R.shape[0]

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
    metadata["projection"] = projdef

    metadata["x1"] = 255000.
    metadata["y1"] = -160000.
    metadata["x2"] = 965000.
    metadata["y2"] = 480000.

    metadata["xpixelsize"] = 1000.
    metadata["ypixelsize"] = 1000.

    metadata["yorigin"] = "upper"

    ds.close()

    return R,metadata 
