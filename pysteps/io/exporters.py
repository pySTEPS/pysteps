"""
pysteps.io.exporter
===================

Methods for exporting forecasts of 2d precipitation fields into various file
formats.

Each exporter method in this module has its own initialization function that
implements the following interface::

  initialize_forecast_exporter_xxx(filename, startdate, timestep,
                                   num_timesteps, shape, num_ens_members,
                                   metadata, incremental=None)

where xxx is the name (or abbreviation) of the file format.

This function creates the file and writes the metadata. The datasets are written
by calling :py:func:`pysteps.io.exporters.export_forecast_dataset`, and
the file is closed by calling :py:func:`pysteps.io.exporters.close_forecast_file`.

The arguments in the above are defined as follows:

.. tabularcolumns:: |p{2cm}|p{2cm}|L|

+---------------+-------------------+-----------------------------------------+
|   Argument    | Type/values       |             Description                 |
+===============+===================+=========================================+
| filename      | str               | name of the output file                 |
+---------------+-------------------+-----------------------------------------+
| startdate     | datetime.datetime | start date of the forecast              |
+---------------+-------------------+-----------------------------------------+
| timestep      | int               | time step of the forecast (minutes)     |
+---------------+-------------------+-----------------------------------------+
| n_timesteps   | int               | number of time steps in the forecast    |
|               |                   | this argument is ignored if             |
|               |                   | incremental is set to 'timestep'.       |
+---------------+-------------------+-----------------------------------------+
| shape         | tuple             | two-element tuple defining the shape    |
|               |                   | (height,width) of the forecast grids    |
+---------------+-------------------+-----------------------------------------+
| n_ens_members | int               | number of ensemble members in the       |
|               |                   | forecast. This argument is ignored if   |
|               |                   | incremental is set to 'member'          |
+---------------+-------------------+-----------------------------------------+
| metadata      | dict              | metadata dictionary containing the      |
|               |                   | projection,x1,x2,y1,y2 and unit         |
|               |                   | attributes described in the             |
|               |                   | documentation of pysteps.io.importers   |
+---------------+-------------------+-----------------------------------------+
| incremental   | {None, 'timestep',| Allow incremental writing of datasets   |
|               | 'member'}         | into the netCDF file                    |
|               |                   | the available options are:              |
|               |                   | 'timestep' = write a forecast or a      |
|               |                   | forecast ensemble for a given           |
|               |                   | time step                               |
|               |                   | 'member' = write a forecast sequence    |
|               |                   | for a given ensemble member             |
+---------------+-------------------+-----------------------------------------+

The return value is a dictionary containing an exporter object. This can be
used with :py:func:`pysteps.io.exporters.export_forecast_dataset` to write 
datasets into the given file format.

Available Exporters
-------------------

.. autosummary::
    :toctree: ../generated/

    initialize_forecast_exporter_kineros
    initialize_forecast_exporter_netcdf

Generic functions
-----------------

.. autosummary::
    :toctree: ../generated/

    export_forecast_dataset
    close_forecast_file
"""

from datetime import datetime
import numpy as np
import os
from pysteps.exceptions import MissingOptionalDependency
try:
    import netCDF4
    netcdf4_imported = True
except ImportError:
    netcdf4_imported = False
try:
    import pyproj
    pyproj_imported = True
except ImportError:
    pyproj_imported = False

# TODO(exporters): This is a draft version of the kineros exporter.
# Revise the variable names and
# the structure of the file if necessary.

def initialize_forecast_exporter_kineros(filename, startdate, timestep,
                                        n_timesteps, shape, n_ens_members,
                                        metadata, incremental=None):
    """Initialize a KINEROS2 Rainfall .pre file as specified
    in https://www.tucson.ars.ag.gov/kineros/.

    Grid points are treated as individual rain gauges and a separate file is
    produced for each ensemble memeber.
    
    Parameters
    ----------
    filename : str
        Name of the output file.
        
    startdate : datetime.datetime
        Start date of the forecast as datetime object.
        
    timestep : int
        Time step of the forecast (minutes).
        
    n_timesteps : int
        Number of time steps in the forecast this argument is ignored if         
        incremental is set to 'timestep'.
        
    shape : tuple of int
        Two-element tuple defining the shape (height,width) of the forecast 
        grids.
        
    n_ens_members : int
        Number of ensemble members in the forecast. This argument is ignored if
        incremental is set to 'member'.
        
    metadata: dict
        Metadata dictionary containing the projection,x1,x2,y1,y2 and unit      
        attributes described in the documentation of 
        :py:mod:`pysteps.io.importers`.
        
    incremental : {None}, optional
        Currently not implemented for this method.

    Returns
    -------
    exporter : dict
        The return value is a dictionary containing an exporter object. This c
        an be used with :py:func:`pysteps.io.exporters.export_forecast_dataset` 
        to write datasets into the given file format.
    
    """

    if incremental is not None:
        raise ValueError("unknown option %s: incremental writing is not supported" % incremental)

    exporter = {}

    basefn, extfn = os.path.splitext(filename)
    if extfn == "":
        extfn = ".pre"

    # one file for each member
    n_ens_members = np.min((99, n_ens_members))
    fns = []
    for i in range(n_ens_members):
        fn = "%s_N%02d%s" % (basefn, i, extfn)
        with open(fn, "w") as fd:
            # write header
            fd.writelines("! pysteps-generated nowcast.\n")
            fd.writelines("! created the %s.\n" % datetime.now().strftime("%c"))
            # TODO(exporters): Add pySTEPS version here
            fd.writelines("! Member = %02d.\n" % i)
            fd.writelines("! Startdate = %s.\n" % startdate.strftime("%c"))
            fns.append(fn)
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

    xr = np.linspace(metadata["x1"], metadata["x2"], w+1)[:-1]
    xr += 0.5 * (xr[1] - xr[0])
    yr = np.linspace(metadata["y1"], metadata["y2"], h+1)[:-1]
    yr += 0.5 * (yr[1] - yr[0])
    X, Y = np.meshgrid(xr, yr)
    XY_coords = np.stack([X, Y])

    exporter["method"] = "kineros"
    exporter["ncfile"] = fns
    exporter["XY_coords"] = XY_coords
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

def initialize_forecast_exporter_netcdf(filename, startdate, timestep,
                                        n_timesteps, shape, n_ens_members,
                                        metadata, incremental=None):
    """Initialize a netCDF forecast exporter.
    
    Parameters
    ----------
    filename : str
        Name of the output file.
        
    startdate : datetime.datetime
        Start date of the forecast as datetime object.
        
    timestep : int
        Time step of the forecast (minutes).
        
    n_timesteps : int
        Number of time steps in the forecast this argument is ignored if         
        incremental is set to 'timestep'.
        
    shape : tuple of int
        Two-element tuple defining the shape (height,width) of the forecast 
        grids.
        
    n_ens_members : int
        Number of ensemble members in the forecast. This argument is ignored if
        incremental is set to 'member'.
        
    metadata: dict
        Metadata dictionary containing the projection,x1,x2,y1,y2 and unit      
        attributes described in the documentation of 
        :py:mod:`pysteps.io.importers`.
        
    incremental : {None,'timestep','member'}, optional
        Allow incremental writing of datasets into the netCDF file.\n
        The available options are: 'timestep' = write a forecast or a forecast 
        ensemble for  a given time step; 'member' = write a forecast sequence 
        for a given ensemble member. If set to None, incremental writing is 
        disabled.

    Returns
    -------
    exporter : dict
        The return value is a dictionary containing an exporter object. This c
        an be used with :py:func:`pysteps.io.exporters.export_forecast_dataset` 
        to write datasets into the given file format.
    
    """
    if not netcdf4_imported:
        raise MissingOptionalDependency(
            "netCDF4 package is required for netcdf "
            "exporters but it is not installed")

    if not pyproj_imported:
        raise MissingOptionalDependency(
            "pyproj package is required for netcdf "
            "exporters but it is not installed")

    if incremental not in [None, "timestep", "member"]:
        raise ValueError("unknown option %s: incremental must be 'timestep' or 'member'" % incremental)

    if incremental == "timestep":
        n_timesteps = None
    elif incremental == "member":
        n_ens_members = None
    elif incremental is not None:
        raise ValueError("unknown argument value incremental='%s': must be 'timestep' or 'member'" % str(incremental))

    exporter = {}

    ncf = netCDF4.Dataset(filename, 'w', format="NETCDF4")

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

    xr = np.linspace(metadata["x1"], metadata["x2"], w+1)[:-1]
    xr += 0.5 * (xr[1] - xr[0])
    yr = np.linspace(metadata["y1"], metadata["y2"], h+1)[:-1]
    yr += 0.5 * (yr[1] - yr[0])

    var_xc = ncf.createVariable("xc", np.float32, dimensions=("x",))
    var_xc[:] = xr
    var_xc.axis = 'X'
    var_xc.standard_name = "projection_x_coordinate"
    var_xc.long_name = "x-coordinate in Cartesian system"
    # TODO(exporters): Don't hard-code the unit.
    var_xc.units = 'm'

    var_yc = ncf.createVariable("yc", np.float32, dimensions=("y",))
    var_yc[:] = yr
    var_yc.axis = 'Y'
    var_yc.standard_name = "projection_y_coordinate"
    var_yc.long_name = "y-coordinate in Cartesian system"
    # TODO(exporters): Don't hard-code the unit.
    var_yc.units = 'm'

    X, Y = np.meshgrid(xr, yr)
    pr = pyproj.Proj(metadata["projection"])
    lon, lat = pr(X.flatten(), Y.flatten(), inverse=True)

    var_lon = ncf.createVariable("lon", np.float, dimensions=("y", "x"))
    var_lon[:] = lon
    var_lon.standard_name = "longitude"
    var_lon.long_name = "longitude coordinate"
    # TODO(exporters): Don't hard-code the unit.
    var_lon.units = "degrees_east"

    var_lat = ncf.createVariable("lat", np.float, dimensions=("y", "x"))
    var_lat[:] = lat
    var_lat.standard_name = "latitude"
    var_lat.long_name = "latitude coordinate"
    # TODO(exporters): Don't hard-code the unit.
    var_lat.units = "degrees_north"

    ncf.projection = metadata["projection"]

    grid_mapping_var_name, grid_mapping_name, grid_mapping_params = \
        _convert_proj4_to_grid_mapping(metadata["projection"])
    # skip writing the grid mapping if a matching name was not found
    if grid_mapping_var_name is not None:
        var_gm = ncf.createVariable(grid_mapping_var_name, np.int,
                                    dimensions=())
        var_gm.grid_mapping_name = grid_mapping_name
        for i in grid_mapping_params.items():
            var_gm.setncattr(i[0], i[1])

    var_ens_num = ncf.createVariable("ens_number", np.int,
                                     dimensions=("ens_number",)
                                     )
    if incremental != "member":
        var_ens_num[:] = list(range(1, n_ens_members+1))
    var_ens_num.long_name = "ensemble member"
    var_ens_num.units = ""

    var_time = ncf.createVariable("time", np.int, dimensions=("time",))
    if incremental != "timestep":
        var_time[:] = [i*timestep*60 for i in range(1, n_timesteps+1)]
    var_time.long_name = "forecast time"
    startdate_str = datetime.strftime(startdate, "%Y-%m-%d %H:%M:%S")
    var_time.units = "seconds since %s" % startdate_str

    var_F = ncf.createVariable(var_name, np.float32,
                               dimensions=("ens_number", "time", "y", "x"),
                               zlib=True, complevel=9)

    if var_standard_name is not None:
        var_F.standard_name = var_standard_name
    var_F.long_name = var_long_name
    var_F.coordinates = "y x"
    var_F.units = var_unit

    exporter["method"] = "netcdf"
    exporter["ncfile"] = ncf
    exporter["var_F"] = var_F
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


def export_forecast_dataset(F, exporter):
    """Write a forecast array into a file.

    The written dataset has dimensions
    (num_ens_members,num_timesteps,shape[0],shape[1]), where shape refers to
    the shape of the two-dimensional forecast grids. If the exporter was
    initialized with incremental!=None, the array is appended to the existing
    dataset either along the ensemble member or time axis.

    Parameters
    ----------
    exporter : dict
        An exporter object created with any initialization method implemented
        in :py:mod:`pysteps.io.exporters`.
    F : array_like
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

    """
    if exporter["method"] == "netcdf" and not netcdf4_imported:
        raise MissingOptionalDependency(
            "netCDF4 package is required for netcdf "
            "exporters but it is not installed")

    if exporter["incremental"] is None:
        shp = (exporter["num_ens_members"], exporter["num_timesteps"],
               exporter["shape"][0], exporter["shape"][1])
        if F.shape != shp:
            raise ValueError("F has invalid shape: %s != %s" % (str(F.shape),str(shp)))
    elif exporter["incremental"] == "timestep":
        shp = (exporter["num_ens_members"], exporter["shape"][0],
               exporter["shape"][1])
        if F.shape != shp:
            raise ValueError("F has invalid shape: %s != %s" % (str(F.shape),str(shp)))
    elif exporter["incremental"] == "member":
        shp = (exporter["num_timesteps"], exporter["shape"][0],
               exporter["shape"][1])
        if F.shape != shp:
            raise ValueError("F has invalid shape: %s != %s" % (str(F.shape),str(shp)))

    if exporter["method"] == "netcdf":
        _export_netcdf(F, exporter)
    elif exporter["method"] == "kineros":
        _export_kineros(F, exporter)
    else:
        raise ValueError("unknown exporter method %s" % exporter["method"])


def close_forecast_file(exporter):
    """Close the file associated with a forecast exporter.

    Finish writing forecasts and close the file associated with a forecast
    exporter.

    Parameters
    ----------
    exporter : dict
        An exporter object created with any initialization method implemented
        in :py:mod:`pysteps.io.exporters`.

    """
    if exporter["method"] == "kineros":
        pass # no need to close the file
    else:
        exporter["ncfile"].close()


def _export_kineros(F, exporter):

    num_timesteps = exporter["num_timesteps"]
    num_ens_members = exporter["num_ens_members"]
    startdate = exporter["startdate"]
    timestep = exporter["timestep"]
    xgrid = exporter["XY_coords"][0, :, :].flatten()
    ygrid = exporter["XY_coords"][1, :, :].flatten()

    timemin = [(t + 1)*timestep for t in range(num_timesteps)]

    for n in range(num_ens_members):
        fn = exporter["ncfile"][n]
        F_ = F[n, :, :, :].reshape((num_timesteps, -1))
        if exporter["var_name"] == "Depth":
            F_ = np.cumsum(F_, axis=0)
        with open(fn, "a") as fd:
            for m in range(F_.shape[1]):
                fd.writelines("BEGIN RG%03d\n" % (m + 1))
                fd.writelines("  X = %.2f, Y = %.2f\n" % (xgrid[m], ygrid[m]))
                fd.writelines("  N = %i\n" % num_timesteps)
                fd.writelines("  TIME        %s\n" % exporter["var_name"].upper())
                fd.writelines("! (min)        (%s)\n" % exporter["var_unit"])
                for t in range(num_timesteps):
                    line_new = "{:6.1f}  {:11.2f}\n".format(timemin[t], F_[t, m])
                    fd.writelines(line_new)
                fd.writelines("END\n\n")


def _export_netcdf(F, exporter):
    var_F = exporter["var_F"]

    if exporter["incremental"] is None:
        var_F[:] = F
    elif exporter["incremental"] == "timestep":
        var_F[:, var_F.shape[1], :, :] = F
        var_time = exporter["var_time"]
        var_time[len(var_time)-1] = len(var_time) * exporter["timestep"] * 60
    else:
        var_F[var_F.shape[0], :, :, :] = F
        var_ens_num = exporter["var_time"]
        var_ens_num[len(var_ens_num)-1] = len(var_ens_num)


# TODO(exporters): Write methods for converting Proj.4 projection definitions
# into CF grid mapping attributes. Currently this has been implemented for
# the stereographic projection.
# The conversions implemented here are take from:
# https://github.com/cf-convention/cf-convention.github.io/blob/master/wkt-proj-4.md

def _convert_proj4_to_grid_mapping(proj4str):
    tokens = proj4str.split('+')

    d = {}
    for t in tokens[1:]:
        t = t.split('=')
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
        print('unknown projection', d["proj"])
        return None, None, None

    return grid_mapping_var_name, grid_mapping_name, params
