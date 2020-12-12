# -*- coding: utf-8 -*-
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
+------------------+----------------------------------------------------------+
|    y1            | y-coordinate of the lower-left corner of the data raster |
+------------------+----------------------------------------------------------+
|    x2            | x-coordinate of the upper-right corner of the data raster|
+------------------+----------------------------------------------------------+
|    y2            | y-coordinate of the upper-right corner of the data raster|
+------------------+----------------------------------------------------------+
|    xpixelsize    | grid resolution in x-direction                           |
+------------------+----------------------------------------------------------+
|    ypixelsize    | grid resolution in y-direction                           |
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

from pysteps.decorators import postprocess_import
from pysteps.exceptions import MissingOptionalDependency, DataModelError

try:
    import netCDF4

    NETCDF4_IMPORTED = True
except ImportError:
    NETCDF4_IMPORTED = False


@postprocess_import(dtype="single")
def import_netcdf_pysteps(filename, onerror="warn", **kwargs):
    """Read a nowcast or an ensemble of nowcasts from a NetCDF file conforming
    to the CF 1.7 specification.

    If an error occurs during the import, the corresponding error message
    is shown, and ( None, None ) is returned.

    Parameters
    ----------
    filename: str
        Name of the file to import.
    onerror: str
        Define the behavior if an exception is raised during the import.
        - "warn": Print an error message and return (None, None)
        - "raise": Raise an exception

    {extra_kwargs_doc}

    Returns
    -------
    precipitation: 2D array, float32
        Precipitation field in mm/h. The dimensions are [latitude, longitude].
        The first grid point (0,0) corresponds to the upper left corner of the
        domain, while (last i, last j) denote the lower right corner.
    metadata: dict
        Associated metadata (pixel sizes, map projections, etc.).
    """
    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required to import pysteps netcdf "
            "nowcasts but it is not installed"
        )

    onerror = onerror.lower()
    if onerror not in ["warn", "raise"]:
        raise ValueError("'onerror' keyword must be 'warn' or 'raise'.")

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

        if isinstance(precip, np.ma.MaskedArray):
            invalid_mask = np.ma.getmaskarray(precip)
            precip = precip.data
            precip[invalid_mask] = np.nan

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
        metadata["xpixelsize"] = abs(ds.variables["x"][1] - ds.variables["x"][0])
        metadata["ypixelsize"] = abs(ds.variables["y"][1] - ds.variables["y"][0])

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
        if (accutime is None) and (leadtimes.size > 1):
            accutime = leadtimes[1] - leadtimes[0]
        metadata["accutime"] = accutime
        metadata["unit"] = unit
        metadata["transform"] = transform
        metadata["zerovalue"] = np.nanmin(precip)
        metadata["threshold"] = np.nanmin(precip[precip > np.nanmin(precip)])

        ds.close()

        return precip, metadata
    except Exception as er:
        if onerror == "warn":
            print("There was an error processing the file", er)
            return None, None
        else:
            raise er


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
