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
|    zerovalue     | it is the value assigned to the no rain pixels with the  |
|                  | same unit, transformation and accutime of the data.      |
+------------------+----------------------------------------------------------+

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

    NETCDF4_IMPORTED = True
except ImportError:
    NETCDF4_IMPORTED = False


def import_netcdf_pysteps(filename, **kwargs):
    """Read a nowcast or a nowcast ensemble from a NetCDF file conforming to the
    CF 1.7 specification."""
    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required to import pysteps netcdf "
            "nowcasts but it is not installed"
        )
    try:
        ds = netCDF4.Dataset(filename, "r")

        var_names = list(ds.variables.keys())

        if "precip_intensity" in var_names:
            precip = ds.variables["precip_intensity"]
            unit = "mm/h"
            accutime = None
            transform = None
        elif "precip_accum" in var_names:
            precip = ds.variables["precip_accum"]
            unit = "mm"
            accutime = None
            transform = None
        elif "hourly_precip_accum" in var_names:
            precip = ds.variables["hourly_precip_accum"]
            unit = "mm"
            accutime = 60.0
            transform = None
        elif "reflectivity" in var_names:
            precip = ds.variables["reflectivity"]
            unit = "dBZ"
            accutime = None
            transform = "dB"
        else:
            raise DataModelError(
                "Non CF compilant file: "
                "the netCDF file does not contain any "
                "supported variable name.\n"
                "Supported names: 'precip_intensity', 'hourly_precip_accum', "
                "or 'reflectivity'\n"
                "file: " + filename
            )

        precip = precip[...].squeeze().astype(float)

        metadata = {}

        time_var = ds.variables["time"]
        leadtimes = time_var[:] / 60.0  # minutes leadtime
        metadata["leadtimes"] = leadtimes
        timestamps = netCDF4.num2date(time_var[:], time_var.units)
        metadata["timestamps"] = timestamps

        if "polar_stereographic" in var_names:
            vn = "polar_stereographic"

            attr_dict = {}
            for attr_name in ds.variables[vn].ncattrs():
                attr_dict[attr_name] = ds[vn].getncattr(attr_name)

            proj_str = _convert_grid_mapping_to_proj4(attr_dict)
            metadata["projection"] = proj_str

        # geodata
        metadata["xpixelsize"] = abs(ds.variables["x"][1] -
                                     ds.variables["x"][0])
        metadata["ypixelsize"] = abs(ds.variables["y"][1] -
                                     ds.variables["y"][0])

        xmin = np.min(ds.variables["x"]) - 0.5 * metadata["xpixelsize"]
        xmax = np.max(ds.variables["x"]) + 0.5 * metadata["xpixelsize"]
        ymin = np.min(ds.variables["y"]) - 0.5 * metadata["ypixelsize"]
        ymax = np.max(ds.variables["y"]) + 0.5 * metadata["ypixelsize"]

        # TODO: this is only a quick solution
        metadata["x1"] = xmin
        metadata["y1"] = ymin
        metadata["x2"] = xmax
        metadata["y2"] = ymax

        metadata["yorigin"] = "upper"  # TODO: check this

        # TODO: Read the metadata to the dictionary.
        if accutime is None:
            accutime = leadtimes[1] - leadtimes[0]
        metadata["accutime"] = accutime
        metadata["unit"] = unit
        metadata["transform"] = transform
        metadata["zerovalue"] = np.nanmin(precip)
        metadata["threshold"] = np.nanmin(precip[precip > np.nanmin(precip)])

        ds.close()

        return precip, metadata
    except Exception as er:
        print("There was an error processing the file", er)
        return None, None


def _convert_grid_mapping_to_proj4(grid_mapping):
    gm_keys = list(grid_mapping.keys())

    # TODO: implement more projection types here
    if grid_mapping["grid_mapping_name"] == "polar_stereographic":
        proj_str = "+proj=stere"
        proj_str += " +lon_0=%s" \
            % grid_mapping["straight_vertical_longitude_from_pole"]
        proj_str += " +lat_0=%s" \
            % grid_mapping["latitude_of_projection_origin"]
        if "standard_parallel" in gm_keys:
            proj_str += " +lat_ts=%s" % grid_mapping["standard_parallel"]
        if "scale_factor_at_projection_origin" in gm_keys:
            proj_str += " +k_0=%s" \
                % grid_mapping["scale_factor_at_projection_origin"]
        proj_str += " +x_0=%s" % grid_mapping["false_easting"]
        proj_str += " +y_0=%s" % grid_mapping["false_northing"]

        return proj_str
    else:
        return None
