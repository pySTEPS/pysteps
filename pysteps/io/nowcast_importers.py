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
import xarray as xr

try:
    import netCDF4

    NETCDF4_IMPORTED = True
except ImportError:
    NETCDF4_IMPORTED = False


def import_netcdf_pysteps(filename, onerror="warn", **kwargs):
    """
    Read a nowcast or an ensemble of nowcasts from a NetCDF file conforming
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
        dataset = xr.open_dataset(filename)
        return dataset
    except Exception as er:
        if onerror == "warn":
            print("There was an error processing the file", er)
            return None, None
        else:
            raise er
