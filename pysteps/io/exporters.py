# -*- coding: utf-8 -*-
"""
pysteps.io.exporters
====================

Methods for exporting forecasts of 2d precipitation fields into various file
formats.

Each exporter method in this module has its own initialization function that
implements the following interface::

  initialize_forecast_exporter_xxx(outpath, outfnprefix, startdate, timestep,
                                   n_timesteps, shape, metadata,
                                   n_ens_members=1,
                                   incremental=None, **kwargs)

where xxx specifies the file format.

This function creates the output files and writes the metadata. See the
documentation of the initialization methods for the format of the output files
and their names. The datasets are written by calling
:py:func:`pysteps.io.exporters.export_forecast_dataset`, and the files are
closed by calling :py:func:`pysteps.io.exporters.close_forecast_files`.

The arguments of initialize_forecast_exporter_xxx are described in the
following table:

.. tabularcolumns:: |p{2cm}|p{2cm}|L|

+---------------+-------------------+-----------------------------------------+
|   Argument    | Type/values       |             Description                 |
+===============+===================+=========================================+
| outpath       | str               | output path                             |
+---------------+-------------------+-----------------------------------------+
| outfnprefix   | str               | prefix of output file names             |
+---------------+-------------------+-----------------------------------------+
| startdate     | datetime.datetime | start date of the forecast              |
+---------------+-------------------+-----------------------------------------+
| timestep      | int               | length of the forecast time step        |
|               |                   | (minutes)                               |
+---------------+-------------------+-----------------------------------------+
| n_timesteps   | int               | number of time steps in the forecast    |
|               |                   | this argument is ignored if             |
|               |                   | incremental is set to 'timestep'.       |
+---------------+-------------------+-----------------------------------------+
| shape         | tuple             | two-element tuple defining the shape    |
|               |                   | (height,width) of the forecast grids    |
+---------------+-------------------+-----------------------------------------+
| metadata      | dict              | metadata dictionary containing the      |
|               |                   | projection,x1,x2,y1,y2 and unit         |
|               |                   | attributes described in the             |
|               |                   | documentation of pysteps.io.importers   |
+---------------+-------------------+-----------------------------------------+
| n_ens_members | int               | number of ensemble members in the       |
|               |                   | forecast                                |
|               |                   | this argument is ignored if incremental |
|               |                   | is set to 'member'                      |
+---------------+-------------------+-----------------------------------------+
| incremental   | {None, 'timestep',| allow incremental writing of datasets   |
|               | 'member'}         | the available options are:              |
|               |                   | 'timestep' = write a forecast or a      |
|               |                   | forecast ensemble for a given           |
|               |                   | time step                               |
|               |                   | 'member' = write a forecast sequence    |
|               |                   | for a given ensemble member             |
+---------------+-------------------+-----------------------------------------+

Optional exporter-specific arguments are passed with ``kwargs``.
The return value is a dictionary containing an exporter object.
This can be used with :py:func:`pysteps.io.exporters.export_forecast_dataset`
to write the datasets to the output files.

Available Exporters
-------------------

.. autosummary::
    :toctree: ../generated/

    initialize_forecast_exporter_geotiff
    initialize_forecast_exporter_kineros
    initialize_forecast_exporter_netcdf

Generic functions
-----------------

.. autosummary::
    :toctree: ../generated/

    export_forecast_dataset
    close_forecast_files
"""

import os
from datetime import datetime

import numpy as np

from pysteps.exceptions import MissingOptionalDependency

try:
    from osgeo import gdal, osr

    GDAL_IMPORTED = True
except ImportError:
    GDAL_IMPORTED = False
try:
    import netCDF4

    NETCDF4_IMPORTED = True
except ImportError:
    NETCDF4_IMPORTED = False
try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False


def initialize_forecast_exporter_geotiff(
    outpath,
    outfnprefix,
    startdate,
    timestep,
    n_timesteps,
    shape,
    metadata,
    n_ens_members=1,
    incremental=None,
    **kwargs,
):
    """
    Initialize a GeoTIFF forecast exporter.

    The output files are named as '<outfnprefix>_<startdate>_<t>.tif', where
    startdate is in YYmmddHHMM format and t is lead time (minutes). GDAL needs
    to be installed to use this exporter.

    Parameters
    ----------
    outpath: str
        Output path.

    outfnprefix: str
        Prefix for output file names.

    startdate: datetime.datetime
        Start date of the forecast.

    timestep: int
        Time step of the forecast (minutes).

    n_timesteps: int
        Number of time steps in the forecast. This argument is ignored if
        incremental is set to 'timestep'.

    shape: tuple of int
        Two-element tuple defining the shape (height,width) of the forecast
        grids.

    metadata: dict
        Metadata dictionary containing the projection,x1,x2,y1,y2 and unit
        attributes described in the documentation of
        :py:mod:`pysteps.io.importers`.

    n_ens_members: int
        Number of ensemble members in the forecast.

    incremental: {None,'timestep'}, optional
        Allow incremental writing of datasets into the GeoTIFF files. Set to
        'timestep' to enable writing forecasts or forecast ensembles separately
        for each time step. If set to None, incremental writing is disabled and
        the whole forecast is written in a single function call. The 'member'
        option is not currently implemented.

    Returns
    -------
    exporter: dict
        The return value is a dictionary containing an exporter object.
        This can be used with
        :py:func:`pysteps.io.exporters.export_forecast_dataset`
        to write the datasets.

    """

    if len(shape) != 2:
        raise ValueError("shape has %d elements, 2 expected" % len(shape))

    del kwargs  # kwargs not used

    if not GDAL_IMPORTED:
        raise MissingOptionalDependency(
            "gdal package is required for GeoTIFF " "exporters but it is not installed"
        )

    if incremental == "member":
        raise ValueError(
            "incremental writing of GeoTIFF files with"
            + " the 'member' option is not supported"
        )

    exporter = dict(
        method="geotiff",
        outfnprefix=outfnprefix,
        startdate=startdate,
        timestep=timestep,
        num_timesteps=n_timesteps,
        shape=shape,
        metadata=metadata,
        num_ens_members=n_ens_members,
        incremental=incremental,
        dst=[],
    )
    driver = gdal.GetDriverByName("GTiff")
    exporter["driver"] = driver

    if incremental != "timestep":
        for i in range(n_timesteps):
            outfn = _get_geotiff_filename(
                outfnprefix, startdate, n_timesteps, timestep, i
            )
            outfn = os.path.join(outpath, outfn)
            dst = _create_geotiff_file(outfn, driver, shape, metadata, n_ens_members)
            exporter["dst"].append(dst)
    else:
        exporter["num_files_written"] = 0

    return exporter


# TODO(exporters): This is a draft version of the kineros exporter.
# Revise the variable names and
# the structure of the file if necessary.


def initialize_forecast_exporter_kineros(
    outpath,
    outfnprefix,
    startdate,
    timestep,
    n_timesteps,
    shape,
    metadata,
    n_ens_members=1,
    incremental=None,
    **kwargs,
):
    """
    Initialize a KINEROS2 format exporter for the rainfall ".pre" files
    specified in https://www.tucson.ars.ag.gov/kineros/.

    Grid points are treated as individual rain gauges and a separate file is
    produced for each ensemble member. The output files are named as
    <outfnprefix>_N<n>.pre, where <n> is the index of ensemble member starting
    from zero.

    Parameters
    ----------
    outpath: str
        Output path.

    outfnprefix: str
        Prefix for output file names.

    startdate: datetime.datetime
        Start date of the forecast.

    timestep: int
        Time step of the forecast (minutes).

    n_timesteps: int
        Number of time steps in the forecast this argument is ignored if
        incremental is set to 'timestep'.

    shape: tuple of int
        Two-element tuple defining the shape (height,width) of the forecast
        grids.

    metadata: dict
        Metadata dictionary containing the projection,x1,x2,y1,y2 and unit
        attributes described in the documentation of
        :py:mod:`pysteps.io.importers`.

    n_ens_members: int
        Number of ensemble members in the forecast. This argument is ignored if
        incremental is set to 'member'.

    incremental: {None}, optional
        Currently not implemented for this method.

    Returns
    -------
    exporter: dict
        The return value is a dictionary containing an exporter object. This c
        an be used with :py:func:`pysteps.io.exporters.export_forecast_dataset`
        to write datasets into the given file format.

    """

    if incremental is not None:
        raise ValueError(
            "unknown option %s: " + "incremental writing is not supported" % incremental
        )

    exporter = {}

    # one file for each member
    n_ens_members = np.min((99, n_ens_members))
    fns = []
    for i in range(n_ens_members):
        outfn = "%s_N%02d%s" % (outfnprefix, i, ".pre")
        outfn = os.path.join(outpath, outfn)
        with open(outfn, "w") as fd:
            # write header
            fd.writelines("! pysteps-generated nowcast.\n")
            fd.writelines("! created the %s.\n" % datetime.now().strftime("%c"))
            # TODO(exporters): Add pySTEPS version here
            fd.writelines("! Member = %02d.\n" % i)
            fd.writelines("! Startdate = %s.\n" % startdate.strftime("%c"))
            fns.append(outfn)
        fd.close()

    h, w = shape

    if metadata["unit"] == "mm/h":
        var_name = "Intensity"
        var_long_name = "Intensity in mm/hr"
        var_unit = "mm/hr"
    elif metadata["unit"] == "mm":
        var_name = "Depth"
        var_long_name = "Accumulated depth in mm"
        var_unit = "mm"
    else:
        raise ValueError("unsupported unit %s" % metadata["unit"])

    xr = np.linspace(metadata["x1"], metadata["x2"], w + 1)[:-1]
    xr += 0.5 * (xr[1] - xr[0])
    yr = np.linspace(metadata["y1"], metadata["y2"], h + 1)[:-1]
    yr += 0.5 * (yr[1] - yr[0])

    xy_coords = np.stack(np.meshgrid(xr, yr))

    exporter["method"] = "kineros"
    exporter["ncfile"] = fns
    exporter["XY_coords"] = xy_coords
    exporter["var_name"] = var_name
    exporter["var_long_name"] = var_long_name
    exporter["var_unit"] = var_unit
    exporter["startdate"] = startdate
    exporter["timestep"] = timestep
    exporter["metadata"] = metadata
    exporter["incremental"] = incremental
    exporter["num_timesteps"] = n_timesteps
    exporter["num_ens_members"] = n_ens_members
    exporter["shape"] = shape

    return exporter


# TODO(exporters): This is a draft version of the netcdf exporter.
# Revise the variable names and
# the structure of the file if necessary.


def initialize_forecast_exporter_netcdf(
    outpath,
    outfnprefix,
    startdate,
    timestep,
    n_timesteps,
    shape,
    metadata,
    n_ens_members=1,
    incremental=None,
    **kwargs,
):
    """
    Initialize a netCDF forecast exporter. All outputs are written to a
    single file named as '<outfnprefix>_.nc'.

    Parameters
    ----------
    outpath: str
        Output path.
    outfnprefix: str
        Prefix for output file names.
    startdate: datetime.datetime
        Start date of the forecast.
    timestep: int
        Time step of the forecast (minutes).
    n_timesteps: int
        Number of time steps in the forecast this argument is ignored if
        incremental is set to 'timestep'.
    shape: tuple of int
        Two-element tuple defining the shape (height,width) of the forecast
        grids.
    metadata: dict
        Metadata dictionary containing the projection, x1, x2, y1, y2,
        unit attributes (projection and variable units) described in the
        documentation of :py:mod:`pysteps.io.importers`.
    n_ens_members: int
        Number of ensemble members in the forecast. This argument is ignored if
        incremental is set to 'member'.
    incremental: {None,'timestep','member'}, optional
        Allow incremental writing of datasets into the netCDF files.\n
        The available options are: 'timestep' = write a forecast or a forecast
        ensemble for  a given time step; 'member' = write a forecast sequence
        for a given ensemble member. If set to None, incremental writing is
        disabled.

    Returns
    -------
    exporter: dict
        The return value is a dictionary containing an exporter object. This c
        an be used with :py:func:`pysteps.io.exporters.export_forecast_dataset`
        to write datasets into the given file format.
    """

    if not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required for netcdf "
            "exporters but it is not installed"
        )

    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required for netcdf " "exporters but it is not installed"
        )

    if incremental not in [None, "timestep", "member"]:
        raise ValueError(
            f"unknown option {incremental}: incremental must be "
            + "'timestep' or 'member'"
        )

    if incremental == "timestep":
        n_timesteps = None
    elif incremental == "member":
        n_ens_members = None
    elif incremental is not None:
        raise ValueError(
            f"unknown argument value incremental='{str(incremental)}': "
            + "must be 'timestep' or 'member'"
        )

    n_ens_gt_one = False
    if n_ens_members is not None:
        if n_ens_members > 1:
            n_ens_gt_one = True

    exporter = {}

    outfn = os.path.join(outpath, outfnprefix + ".nc")
    ncf = netCDF4.Dataset(outfn, "w", format="NETCDF4")

    ncf.Conventions = "CF-1.7"
    ncf.title = "pysteps-generated nowcast"
    ncf.institution = "the pySTEPS community (https://pysteps.github.io)"
    ncf.source = "pysteps"  # TODO(exporters): Add pySTEPS version here
    ncf.history = ""
    ncf.references = ""
    ncf.comment = ""

    h, w = shape

    ncf.createDimension("ens_number", size=n_ens_members)
    ncf.createDimension("time", size=n_timesteps)
    ncf.createDimension("y", size=h)
    ncf.createDimension("x", size=w)

    if metadata["unit"] == "mm/h":
        var_name = "precip_intensity"
        var_standard_name = None
        var_long_name = "instantaneous precipitation rate"
        var_unit = "mm h-1"
    elif metadata["unit"] == "mm":
        var_name = "precip_accum"
        var_standard_name = None
        var_long_name = "accumulated precipitation"
        var_unit = "mm"
    elif metadata["unit"] == "dBZ":
        var_name = "reflectivity"
        var_long_name = "equivalent reflectivity factor"
        var_standard_name = "equivalent_reflectivity_factor"
        var_unit = "dBZ"
    else:
        raise ValueError("unknown unit %s" % metadata["unit"])

    xr = np.linspace(metadata["x1"], metadata["x2"], w + 1)[:-1]
    xr += 0.5 * (xr[1] - xr[0])
    yr = np.linspace(metadata["y1"], metadata["y2"], h + 1)[:-1]
    yr += 0.5 * (yr[1] - yr[0])

    # flip yr vector if yorigin is upper
    if metadata["yorigin"] == "upper":
        yr = np.flip(yr)

    var_xc = ncf.createVariable("x", np.float32, dimensions=("x",))
    var_xc[:] = xr
    var_xc.axis = "X"
    var_xc.standard_name = "projection_x_coordinate"
    var_xc.long_name = "x-coordinate in Cartesian system"
    var_xc.units = metadata["cartesian_unit"]

    var_yc = ncf.createVariable("y", np.float32, dimensions=("y",))
    var_yc[:] = yr
    var_yc.axis = "Y"
    var_yc.standard_name = "projection_y_coordinate"
    var_yc.long_name = "y-coordinate in Cartesian system"
    var_yc.units = metadata["cartesian_unit"]

    x_2d, y_2d = np.meshgrid(xr, yr)
    pr = pyproj.Proj(metadata["projection"])
    lon, lat = pr(x_2d.flatten(), y_2d.flatten(), inverse=True)

    var_lon = ncf.createVariable("lon", float, dimensions=("y", "x"))
    var_lon[:] = lon.reshape(shape)
    var_lon.standard_name = "longitude"
    var_lon.long_name = "longitude coordinate"
    # TODO(exporters): Don't hard-code the unit.
    var_lon.units = "degrees_east"

    var_lat = ncf.createVariable("lat", float, dimensions=("y", "x"))
    var_lat[:] = lat.reshape(shape)
    var_lat.standard_name = "latitude"
    var_lat.long_name = "latitude coordinate"
    # TODO(exporters): Don't hard-code the unit.
    var_lat.units = "degrees_north"

    ncf.projection = metadata["projection"]

    (
        grid_mapping_var_name,
        grid_mapping_name,
        grid_mapping_params,
    ) = _convert_proj4_to_grid_mapping(metadata["projection"])
    # skip writing the grid mapping if a matching name was not found
    if grid_mapping_var_name is not None:
        var_gm = ncf.createVariable(grid_mapping_var_name, int, dimensions=())
        var_gm.grid_mapping_name = grid_mapping_name
        for i in grid_mapping_params.items():
            var_gm.setncattr(i[0], i[1])

    if incremental == "member" or n_ens_gt_one:
        var_ens_num = ncf.createVariable("ens_number", int, dimensions=("ens_number",))
        if incremental != "member":
            var_ens_num[:] = list(range(1, n_ens_members + 1))
        var_ens_num.long_name = "ensemble member"
        var_ens_num.standard_name = "realization"
        var_ens_num.units = ""

    var_time = ncf.createVariable("time", int, dimensions=("time",))
    if incremental != "timestep":
        var_time[:] = [i * timestep * 60 for i in range(1, n_timesteps + 1)]
    var_time.long_name = "forecast time"
    startdate_str = datetime.strftime(startdate, "%Y-%m-%d %H:%M:%S")
    var_time.units = "seconds since %s" % startdate_str

    if incremental == "member" or n_ens_gt_one:
        var_f = ncf.createVariable(
            var_name,
            np.float32,
            dimensions=("ens_number", "time", "y", "x"),
            zlib=True,
            complevel=9,
        )
    else:
        var_f = ncf.createVariable(
            var_name, np.float32, dimensions=("time", "y", "x"), zlib=True, complevel=9
        )

    if var_standard_name is not None:
        var_f.standard_name = var_standard_name
    var_f.long_name = var_long_name
    var_f.coordinates = "y x"
    var_f.units = var_unit
    if grid_mapping_var_name is not None:
        var_f.grid_mapping = grid_mapping_var_name

    exporter["method"] = "netcdf"
    exporter["ncfile"] = ncf
    exporter["var_F"] = var_f
    if incremental == "member" or n_ens_gt_one:
        exporter["var_ens_num"] = var_ens_num
    exporter["var_time"] = var_time
    exporter["var_name"] = var_name
    exporter["startdate"] = startdate
    exporter["timestep"] = timestep
    exporter["metadata"] = metadata
    exporter["incremental"] = incremental
    exporter["num_timesteps"] = n_timesteps
    exporter["num_ens_members"] = n_ens_members
    exporter["shape"] = shape

    return exporter


def export_forecast_dataset(field, exporter):
    """Write a forecast array into a file.

    If the exporter was initialized with n_ens_members>1, the written dataset
    has dimensions (n_ens_members,num_timesteps,shape[0],shape[1]), where shape
    refers to the shape of the two-dimensional forecast grids. Otherwise, the
    dimensions are (num_timesteps,shape[0],shape[1]). If the exporter was
    initialized with incremental!=None, the array is appended to the existing
    dataset either along the ensemble member or time axis.

    Parameters
    ----------
    exporter: dict
        An exporter object created with any initialization method implemented
        in :py:mod:`pysteps.io.exporters`.
    field: array_like
        The array to write. The required shape depends on the choice of the
        'incremental' parameter the exporter was initialized with:

        +-----------------+---------------------------------------------------+
        |    incremental  |                    required shape                 |
        +=================+===================================================+
        |    None         | (num_ens_members,num_timesteps,shape[0],shape[1]) |
        +-----------------+---------------------------------------------------+
        |    'timestep'   | (num_ens_members,shape[0],shape[1])               |
        +-----------------+---------------------------------------------------+
        |    'member'     | (num_timesteps,shape[0],shape[1])                 |
        +-----------------+---------------------------------------------------+

        If the exporter was initialized with num_ens_members=1,
        the num_ens_members dimension is dropped.
    """

    if exporter["method"] == "netcdf" and not NETCDF4_IMPORTED:
        raise MissingOptionalDependency(
            "netCDF4 package is required for netcdf "
            "exporters but it is not installed"
        )

    if exporter["incremental"] is None:
        if exporter["num_ens_members"] > 1:
            shp = (
                exporter["num_ens_members"],
                exporter["num_timesteps"],
                exporter["shape"][0],
                exporter["shape"][1],
            )
        else:
            shp = (
                exporter["num_timesteps"],
                exporter["shape"][0],
                exporter["shape"][1],
            )
        if field.shape != shp:
            raise ValueError(
                "field has invalid shape: %s != %s" % (str(field.shape), str(shp))
            )
    elif exporter["incremental"] == "timestep":
        if exporter["num_ens_members"] > 1:
            shp = (
                exporter["num_ens_members"],
                exporter["shape"][0],
                exporter["shape"][1],
            )
        else:
            shp = exporter["shape"]
        if field.shape != shp:
            raise ValueError(
                "field has invalid shape: %s != %s" % (str(field.shape), str(shp))
            )
    elif exporter["incremental"] == "member":
        shp = (exporter["num_timesteps"], exporter["shape"][0], exporter["shape"][1])
        if field.shape != shp:
            raise ValueError(
                "field has invalid shape: %s != %s" % (str(field.shape), str(shp))
            )

    if exporter["method"] == "geotiff":
        _export_geotiff(field, exporter)
    elif exporter["method"] == "netcdf":
        _export_netcdf(field, exporter)
    elif exporter["method"] == "kineros":
        _export_kineros(field, exporter)
    else:
        raise ValueError("unknown exporter method %s" % exporter["method"])


def close_forecast_files(exporter):
    """
    Close the files associated with a forecast exporter.

    Finish writing forecasts and close the output files opened by a forecast
    exporter.

    Parameters
    ----------
    exporter: dict
        An exporter object created with any initialization method implemented
        in :py:mod:`pysteps.io.exporters`.
    """

    if exporter["method"] == "geotiff":
        pass  # NOTE: There is no explicit "close" method in GDAL.
        # The files are closed when all objects referencing to the GDAL
        # datasets are deleted (i.e. when the exporter object is deleted).
    if exporter["method"] == "kineros":
        pass  # no need to close the file
    else:
        exporter["ncfile"].close()


def _export_geotiff(F, exporter):
    def init_band(band):
        band.SetScale(1.0)
        band.SetOffset(0.0)
        band.SetUnitType(exporter["metadata"]["unit"])

    if exporter["incremental"] is None:
        for i in range(exporter["num_timesteps"]):
            if exporter["num_ens_members"] == 1:
                band = exporter["dst"][i].GetRasterBand(1)
                init_band(band)
                band.WriteArray(F[i, :, :])
            else:
                for j in range(exporter["num_ens_members"]):
                    band = exporter["dst"][i].GetRasterBand(j + 1)
                    init_band(band)
                    band.WriteArray(F[j, i, :, :])
    elif exporter["incremental"] == "timestep":
        i = exporter["num_files_written"]

        outfn = _get_geotiff_filename(
            exporter["outfnprefix"],
            exporter["startdate"],
            exporter["num_timesteps"],
            exporter["timestep"],
            i,
        )
        dst = _create_geotiff_file(
            outfn,
            exporter["driver"],
            exporter["shape"],
            exporter["metadata"],
            exporter["num_ens_members"],
        )

        for j in range(exporter["num_ens_members"]):
            band = dst.GetRasterBand(j + 1)
            init_band(band)
            if exporter["num_ens_members"] > 1:
                band.WriteArray(F[j, :, :])
            else:
                band.WriteArray(F)

        exporter["num_files_written"] += 1
    elif exporter["incremental"] == "member":
        for i in range(exporter["num_timesteps"]):
            # NOTE: This does not work because the GeoTIFF driver does not
            # support adding bands. An alternative solution needs to be
            # implemented.
            exporter["dst"][i].AddBand(gdal.GDT_Float32)
            band = exporter["dst"][i].GetRasterBand(exporter["dst"][i].RasterCount)
            init_band(band)
            band.WriteArray(F[i, :, :])


def _export_kineros(field, exporter):
    num_timesteps = exporter["num_timesteps"]
    num_ens_members = exporter["num_ens_members"]

    timestep = exporter["timestep"]
    xgrid = exporter["XY_coords"][0, :, :].flatten()
    ygrid = exporter["XY_coords"][1, :, :].flatten()

    timemin = [(t + 1) * timestep for t in range(num_timesteps)]

    if field.ndim == 3:
        field = field.reshape((1,) + field.shape)

    for n in range(num_ens_members):
        file_name = exporter["ncfile"][n]

        field_tmp = field[n, :, :, :].reshape((num_timesteps, -1))

        if exporter["var_name"] == "Depth":
            field_tmp = np.cumsum(field_tmp, axis=0)

        with open(file_name, "a") as fd:
            for m in range(field_tmp.shape[1]):
                fd.writelines("BEGIN RG%03d\n" % (m + 1))
                fd.writelines("  X = %.2f, Y = %.2f\n" % (xgrid[m], ygrid[m]))
                fd.writelines("  N = %i\n" % num_timesteps)
                fd.writelines("  TIME        %s\n" % exporter["var_name"].upper())
                fd.writelines("! (min)        (%s)\n" % exporter["var_unit"])
                for t in range(num_timesteps):
                    line_new = "{:6.1f}  {:11.2f}\n".format(timemin[t], field_tmp[t, m])
                    fd.writelines(line_new)
                fd.writelines("END\n\n")


def _export_netcdf(field, exporter):
    var_f = exporter["var_F"]

    if exporter["incremental"] is None:
        var_f[:] = field
    elif exporter["incremental"] == "timestep":
        if exporter["num_ens_members"] > 1:
            var_f[:, var_f.shape[1], :, :] = field
        else:
            var_f[var_f.shape[0], :, :] = field
        var_time = exporter["var_time"]
        var_time[len(var_time) - 1] = len(var_time) * exporter["timestep"] * 60
    else:
        var_f[var_f.shape[0], :, :, :] = field
        var_ens_num = exporter["var_ens_num"]
        var_ens_num[len(var_ens_num) - 1] = len(var_ens_num)


# TODO(exporters): Write methods for converting Proj.4 projection definitions
# into CF grid mapping attributes. Currently this has been implemented for
# the stereographic projection.
# The conversions implemented here are take from:
# https://github.com/cf-convention/cf-convention.github.io/blob/master/wkt-proj-4.md


def _convert_proj4_to_grid_mapping(proj4str):
    tokens = proj4str.split("+")

    d = {}
    for t in tokens[1:]:
        t = t.split("=")
        if len(t) > 1:
            d[t[0]] = t[1].strip()

    params = {}
    # TODO(exporters): implement more projection types here
    if d["proj"] == "stere":
        grid_mapping_var_name = "polar_stereographic"
        grid_mapping_name = "polar_stereographic"
        v = d["lon_0"] if d["lon_0"][-1] not in ["E", "W"] else d["lon_0"][:-1]
        params["straight_vertical_longitude_from_pole"] = float(v)
        v = d["lat_0"] if d["lat_0"][-1] not in ["N", "S"] else d["lat_0"][:-1]
        params["latitude_of_projection_origin"] = float(v)
        if "lat_ts" in list(d.keys()):
            params["standard_parallel"] = float(d["lat_ts"])
        elif "k_0" in list(d.keys()):
            params["scale_factor_at_projection_origin"] = float(d["k_0"])
        params["false_easting"] = float(d["x_0"])
        params["false_northing"] = float(d["y_0"])
    elif d["proj"] == "aea":  # Albers Conical Equal Area
        grid_mapping_var_name = "proj"
        grid_mapping_name = "albers_conical_equal_area"
        params["false_easting"] = float(d["x_0"]) if "x_0" in d else float(0)
        params["false_northing"] = float(d["y_0"]) if "y_0" in d else float(0)
        v = d["lon_0"] if "lon_0" in d else float(0)
        params["longitude_of_central_meridian"] = float(v)
        v = d["lat_0"] if "lat_0" in d else float(0)
        params["latitude_of_projection_origin"] = float(v)
        v1 = d["lat_1"] if "lat_1" in d else float(0)
        v2 = d["lat_2"] if "lat_2" in d else float(0)
        params["standard_parallel"] = (float(v1), float(v2))
    else:
        print("unknown projection", d["proj"])
        return None, None, None

    return grid_mapping_var_name, grid_mapping_name, params


def _create_geotiff_file(outfn, driver, shape, metadata, num_bands):
    dst = driver.Create(
        outfn,
        shape[1],
        shape[0],
        num_bands,
        gdal.GDT_Float32,
        ["COMPRESS=DEFLATE", "PREDICTOR=3"],
    )

    sx = (metadata["x2"] - metadata["x1"]) / shape[1]
    sy = (metadata["y2"] - metadata["y1"]) / shape[0]
    dst.SetGeoTransform([metadata["x1"], sx, 0.0, metadata["y2"], 0.0, -sy])

    sr = osr.SpatialReference()
    sr.ImportFromProj4(metadata["projection"])
    dst.SetProjection(sr.ExportToWkt())

    return dst


def _get_geotiff_filename(prefix, startdate, n_timesteps, timestep, timestep_index):
    if n_timesteps * timestep == 0:
        raise ValueError("n_timesteps x timestep can't be 0.")

    timestep_format_str = (
        f"{{time_str:0{int(np.floor(np.log10(n_timesteps * timestep))) + 1}d}}"
    )

    startdate_str = datetime.strftime(startdate, "%Y%m%d%H%M")

    timestep_str = timestep_format_str.format(time_str=(timestep_index + 1) * timestep)

    return f"{prefix}_{startdate_str}_{timestep_str}.tif"
