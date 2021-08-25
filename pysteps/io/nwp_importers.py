"""
pysteps.io.nwp_importers
====================

Methods for importing files containing two-dimensional NWP mosaics.

The methods in this module implement the following interface::

    nwp_import_xxx(filename, optional arguments)

where **xxx** is the name (or abbreviation) of the file format and filename
is the name of the input file.

The output of each method is a xarray DataArray containing
forecast rainfall fields in mm/timestep and the metadata as attributes.

The metadata should contain the following recommended key-value pairs:

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

    import_bom_nwp_xr
    import_rmi_nwp_xr
    import_knmi_nwp_xr
"""

import numpy as np
import xarray as xr

from pysteps.decorators import postprocess_import
from pysteps.exceptions import MissingOptionalDependency

try:
    import netCDF4

    NETCDF4_IMPORTED = True
except ImportError:
    NETCDF4_IMPORTED = False


@postprocess_import()
def import_bom_nwp_xr(filename, **kwargs):
    """Import a NetCDF with NWP rainfall forecasts regridded to a BoM Rainfields3
    using xarray.

    Parameters
    ----------
    filename: str
        Name of the file to import.

    {extra_kwargs_doc}

    Returns
    -------
    out_da : xr.DataArray
        A xarray DataArray containing forecast rainfall fields in mm/timestep
        imported from a netcdf, and the metadata.
    """

    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required to import BoM NWP regridded rainfall "
            "products but it is not installed"
        )

    ds = _import_bom_nwp_data_xr(filename, **kwargs)
    ds_meta = _import_bom_nwp_geodata_xr(ds, **kwargs)

    # rename varname_time (def: time) to t
    varname_time = kwargs.get("varname_time", "time")
    ds_meta = ds_meta.rename({varname_time: "t"})
    varname_time = "t"

    # if data variable is named accum_prcp
    # it is assumed that NWP rainfall data is accumulated
    # so it needs to be disagregated by time step
    varname = kwargs.get("varname", "accum_prcp")
    if varname == "accum_prcp":
        print("Rainfall values are accumulated. Disaggregating by time step")
        accum_prcp = ds_meta[varname]
        precipitation = accum_prcp - accum_prcp.shift({varname_time: 1})
        precipitation = precipitation.dropna(varname_time, "all")
        # update/copy attributes
        precipitation.name = "precipitation"
        # copy attributes
        precipitation.attrs.update({**accum_prcp.attrs})
        # update attributes
        precipitation.attrs.update({"standard_name": "precipitation_amount"})
    else:
        precipitation = ds_meta[varname]

    return precipitation


def _import_bom_nwp_data_xr(filename, **kwargs):

    varname_time = kwargs.get("varname_time", "time")
    chunks = kwargs.get("chunks", {varname_time: 1})

    ds_rainfall = xr.open_mfdataset(
        filename,
        combine="nested",
        concat_dim=varname_time,
        chunks=chunks,
        lock=False,
        parallel=True,
    )

    return ds_rainfall


def _import_bom_nwp_geodata_xr(
    ds_in,
    **kwargs,
):

    varname = kwargs.get("varname", "accum_prcp")
    varname_time = kwargs.get("varname_time", "time")

    # extract useful information
    # projection
    projdef = None
    if "proj" in ds_in:
        projection = ds_in.proj
        if projection.grid_mapping_name == "albers_conical_equal_area":
            projdef = "+proj=aea "
            lon_0 = projection.longitude_of_central_meridian
            projdef += f" +lon_0={lon_0:.3f}"
            lat_0 = projection.latitude_of_projection_origin
            projdef += f" +lat_0={lat_0:.3f}"
            standard_parallel = projection.standard_parallel
            projdef += f" +lat_1={standard_parallel[0]:.3f}"
            projdef += f" +lat_2={standard_parallel[1]:.3f}"

    # get the accumulation period
    time = ds_in[varname_time]
    # shift the array to calculate the time step
    delta_time = time - time.shift({varname_time: 1})
    # assuming first valid delta_time is representative of all time steps
    time_step = delta_time[1]
    time_step = time_step.values.astype("timedelta64[m]")

    # get the units of precipitation
    units = None
    if "units" in ds_in[varname].attrs:
        units = ds_in[varname].units
        if units in ("kg m-2", "mm"):
            units = "mm"
            ds_in[varname].attrs.update({"units": units})
    # get spatial boundaries and pixelsize
    # move to meters if coordiantes in kilometers
    if "units" in ds_in.x.attrs:
        if ds_in.x.units == "km":
            ds_in["x"] = ds_in.x * 1000.0
            ds_in.x.attrs.update({"units": "m"})
            ds_in["y"] = ds_in.y * 1000.0
            ds_in.y.attrs.update({"units": "m"})

    xmin = ds_in.x.min().values
    xmax = ds_in.x.max().values
    ymin = ds_in.y.min().values
    ymax = ds_in.y.max().values
    xpixelsize = abs(ds_in.x[1] - ds_in.x[0])
    ypixelsize = abs(ds_in.y[1] - ds_in.y[0])

    cartesian_unit = ds_in.x.units

    # Add metadata needed by pySTEPS as attrs in X and Y variables

    ds_in.x.attrs.update(
        {
            # TODO: Remove before final 2.0 version
            "x1": xmin,
            "x2": xmax,
            "cartesian_unit": cartesian_unit,
        }
    )

    ds_in.y.attrs.update(
        {
            # TODO: Remove before final 2.0 version
            "y1": ymin,
            "y2": ymax,
            "cartesian_unit": cartesian_unit,
        }
    )

    # Add metadata needed by pySTEPS as attrs in rainfall variable
    da_rainfall = ds_in[varname].isel({varname_time: 0})

    ds_in[varname].attrs.update(
        {
            "transform": None,
            "unit": units,  # copy 'units' in 'unit' for legacy reasons
            "projection": projdef,
            "accutime": time_step,
            "zr_a": None,
            "zr_b": None,
            "zerovalue": np.nanmin(da_rainfall),
            "institution": "Commonwealth of Australia, Bureau of Meteorology",
            "threshold": _get_threshold_value(da_rainfall.values),
            # TODO(_import_bom_rf3_geodata_xr): Remove before final 2.0 version
            "yorigin": "upper",
            "xpixelsize": xpixelsize.values,
            "ypixelsize": ypixelsize.values,
        }
    )

    return ds_in


@postprocess_import()
def import_rmi_nwp_xr(filename, **kwargs):
    """Import a NetCDF with NWP rainfall forecasts from RMI using xarray.

    Parameters
    ----------
    filename: str
        Name of the file to import.

    {extra_kwargs_doc}

    Returns
    -------
    out_da : xr.DataArray
        A xarray DataArray containing forecast rainfall fields in mm/timestep
        imported from a netcdf, and the metadata.
    """

    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required to import RMI NWP rainfall "
            "products but it is not installed"
        )

    ds = _import_rmi_nwp_data_xr(filename, **kwargs)
    ds_meta = _import_rmi_nwp_geodata_xr(ds, **kwargs)

    # rename varname_time (def: time) to t
    varname_time = kwargs.get("varname_time", "time")
    ds_meta = ds_meta.rename({varname_time: "t"})
    varname_time = "t"

    # if data variable is named accum_prcp
    # it is assumed that NWP rainfall data is accumulated
    # so it needs to be disagregated by time step
    varname = kwargs.get("varname", "precipitation")
    if varname == "accum_prcp":
        print("Rainfall values are accumulated. Disaggregating by time step")
        accum_prcp = ds_meta[varname]
        precipitation = accum_prcp - accum_prcp.shift({varname_time: 1})
        precipitation = precipitation.dropna(varname_time, "all")
        # update/copy attributes
        precipitation.name = "precipitation"
        # copy attributes
        precipitation.attrs.update({**accum_prcp.attrs})
        # update attributes
        precipitation.attrs.update({"standard_name": "precipitation_amount"})
    else:
        precipitation = ds_meta[varname]

    return precipitation


def _import_rmi_nwp_data_xr(filename, **kwargs):

    varname_time = kwargs.get("varname_time", "time")
    chunks = kwargs.get("chunks", {varname_time: 1})

    ds_rainfall = xr.open_mfdataset(
        filename,
        combine="nested",
        concat_dim=varname_time,
        chunks=chunks,
        lock=False,
        parallel=True,
    )

    return ds_rainfall


def _import_rmi_nwp_geodata_xr(
    ds_in,
    **kwargs,
):

    varname = kwargs.get("varname", "precipitation")
    varname_time = kwargs.get("varname_time", "time")
    projdef = None
    if "proj4string" in ds_in.attrs:
        projdef = ds_in.proj4string

    # get the accumulation period
    time = ds_in[varname_time]
    # shift the array to calculate the time step
    delta_time = time - time.shift({varname_time: 1})
    # assuming first valid delta_time is representative of all time steps
    time_step = delta_time[1]
    time_step = time_step.values.astype("timedelta64[m]")

    # get the units of precipitation
    units = None
    if "units" in ds_in[varname].attrs:
        units = ds_in[varname].units
        if units in ("kg m-2", "mm"):
            units = "mm"
            ds_in[varname].attrs.update({"units": units})
    # get spatial boundaries and pixelsize
    # move to meters if coordinates in kilometers
    if "units" in ds_in.x.attrs:
        if ds_in.x.units == "km":
            ds_in["x"] = ds_in.x * 1000.0
            ds_in.x.attrs.update({"units": "m"})
            ds_in["y"] = ds_in.y * 1000.0
            ds_in.y.attrs.update({"units": "m"})

    xmin = ds_in.x.min().values
    xmax = ds_in.x.max().values
    ymin = ds_in.y.min().values
    ymax = ds_in.y.max().values
    xpixelsize = abs(ds_in.x[1] - ds_in.x[0])
    ypixelsize = abs(ds_in.y[1] - ds_in.y[0])

    cartesian_unit = ds_in.x.units

    # Add metadata needed by pySTEPS as attrs in X and Y variables

    ds_in.x.attrs.update(
        {
            # TODO: Remove before final 2.0 version
            "x1": xmin,
            "x2": xmax,
            "cartesian_unit": cartesian_unit,
        }
    )

    ds_in.y.attrs.update(
        {
            # TODO: Remove before final 2.0 version
            "y1": ymin,
            "y2": ymax,
            "cartesian_unit": cartesian_unit,
        }
    )

    # Add metadata needed by pySTEPS as attrs in rainfall variable
    da_rainfall = ds_in[varname].isel({varname_time: 0})

    ds_in[varname].attrs.update(
        {
            "transform": None,
            "unit": units,  # copy 'units' in 'unit' for legacy reasons
            "projection": projdef,
            "accutime": time_step,
            "zr_a": None,
            "zr_b": None,
            "zerovalue": np.nanmin(da_rainfall),
            "institution": "Royal Meteorological Institute of Belgium",
            "threshold": _get_threshold_value(da_rainfall.values),
            # TODO(_import_bom_rf3_geodata_xr): Remove before final 2.0 version
            "yorigin": "upper",
            "xpixelsize": xpixelsize.values,
            "ypixelsize": ypixelsize.values,
        }
    )

    return ds_in


def import_knmi_nwp_xr(filename, **kwargs):
    """Import a NetCDF with HARMONIE NWP rainfall forecasts from KNMI using
    xarray.

    Parameters
    ----------
    filename: str
        Name of the file to import.

    {extra_kwargs_doc}

    Returns
    -------
    out_da : xr.DataArray
        A xarray DataArray containing forecast rainfall fields in mm/timestep
        imported from a netcdf, and the metadata.
    """

    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required to import BoM NWP regrided rainfall "
            "products but it is not installed"
        )

    ds = _import_knmi_nwp_data_xr(filename, **kwargs)
    ds_meta = _import_knmi_nwp_geodata_xr(ds, **kwargs)

    # rename varname_time (def: time) to t
    varname_time = kwargs.get("varname_time", "time")
    ds_meta = ds_meta.rename({varname_time: "t"})
    varname_time = "t"

    # if data variable is named accum_prcp
    # it is assumed that NWP rainfall data is accumulated
    # so it needs to be disagregated by time step
    varname = kwargs.get("varname", "P_fc")
    if varname == "P_fc":
        precipitation = ds_meta[varname]
        precipitation = precipitation.dropna(varname_time, "all")
        # update/copy attributes
        precipitation.name = "precipitation"
        # update attributes
        precipitation.attrs.update({"standard_name": "precipitation_amount"})
    else:
        precipitation = ds_meta[varname]

    return precipitation


def _import_knmi_nwp_data_xr(filename, **kwargs):

    varname_time = kwargs.get("varname_time", "time")
    chunks = kwargs.get("chunks", {varname_time: 1})

    ds_rainfall = xr.open_mfdataset(
        filename,
        combine="nested",
        concat_dim=varname_time,
        chunks=chunks,
        lock=False,
        parallel=True,
    )

    return ds_rainfall


def _import_knmi_nwp_geodata_xr(
    ds_in,
    **kwargs,
):

    varname = kwargs.get("varname", "P_fc")
    varname_time = kwargs.get("varname_time", "time")
    # Get the projection string
    projdef = ds_in.crs.proj4_params

    # Get the accumulation period
    time = ds_in[varname_time]
    # Shift the array to calculate the time step
    delta_time = time - time.shift({varname_time: 1})
    # Assuming first valid delta_time is representative of all time steps
    time_step = delta_time[1]
    time_step = time_step.values.astype("timedelta64[m]")

    # Get the units of precipitation
    units = None
    if "units" in ds_in[varname].attrs:
        units = ds_in[varname].units
        if units in ("kg m-2", "mm"):
            units = "mm"
            ds_in[varname].attrs.update({"units": units})

    # Get spatial boundaries and pixelsize
    xmin = ds_in.x.min().values
    xmax = ds_in.x.max().values
    ymin = ds_in.y.min().values
    ymax = ds_in.y.max().values
    xpixelsize = abs(ds_in.x[1] - ds_in.x[0])
    ypixelsize = abs(ds_in.y[1] - ds_in.y[0])

    cartesian_unit = ds_in.x.units

    # Add metadata needed by pySTEPS as attrs in X and Y variables

    ds_in.x.attrs.update(
        {
            # TODO: Remove before final 2.0 version
            "x1": xmin,
            "x2": xmax,
            "cartesian_unit": cartesian_unit,
        }
    )

    ds_in.y.attrs.update(
        {
            # TODO: Remove before final 2.0 version
            "y1": ymin,
            "y2": ymax,
            "cartesian_unit": cartesian_unit,
        }
    )

    # Add metadata needed by pySTEPS as attrs in rainfall variable
    da_rainfall = ds_in[varname].isel({varname_time: 0})

    ds_in[varname].attrs.update(
        {
            "transform": None,
            "unit": units,  # copy 'units' in 'unit' for legacy reasons
            "projection": projdef,
            "accutime": time_step,
            "zr_a": None,
            "zr_b": None,
            "zerovalue": np.nanmin(da_rainfall),
            "institution": ds_in.attrs["institution"],
            "threshold": _get_threshold_value(da_rainfall.values),
            # TODO(_import_bom_rf3_geodata_xr): Remove before final 2.0 version
            "yorigin": "lower",
            "xpixelsize": xpixelsize.values,
            "ypixelsize": ypixelsize.values,
        }
    )

    return ds_in


def _get_threshold_value(precip):
    """
    Get the rain/no rain threshold with the same unit, transformation and
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
