"""Methods for writing forecasts of 2d precipitation fields into various file 
formats."""

import numpy as np
from datetime import datetime
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

# TODO: Define an interface for exporter methods.

# TODO: Write documentation about the written dataset dimensions.

# TODO: This is a draft version of the exporter. Revise the variable names and 
# the structure of the file if necessary.
def initialize_nowcast_exporter_netcdf(filename, startdate, timestep, num_timesteps, 
                                       shape, num_ens_members, metadata, 
                                       incremental=None):
    """Initialize a netCDF nowcast exporter.
    
    Parameters
    ----------
    filename : str
        Name of the output file.
    startdate : datetime.datetime
        Start date of the nowcast.
    timestep : int
        Time step of the nowcast (minutes).
    num_timesteps : int
        Number of time steps in the nowcast. This argument is ignored if 
        incremental is set to 'timestep'.
    shape : tuple
        Two-element tuple defining the shape (height,width) of the nowcast grids.
    num_ens_members : int
        Number of ensemble members in the nowcast. This argument is ignored if 
        incremental is set to 'member'.
    metadata : dict
        Metadata dictionary containing the projection,x1,x2,y1,y2 and unit 
        attributes described in the documentation of pysteps.io.importers.
    incremental : str
        Allow incremental writing of datasets into the netCDF file. The 
        available options are: 'timestep'=write one nowcast or a nowcast 
        ensemble per each time step or 'member'=write one forecast sequence per 
        each ensemble member.
    
    Returns
    -------
    out : dict
        An exporter object that can be used with export_nowcast to write 
        datasets into the netCDF file.
    """
    if not netcdf4_imported:
        raise Exception("netCDF4 not imported")
    
    if not pyproj_imported:
        raise Exception("pyproj not imported")
    
    if incremental not in [None, "timestep", "member"]:
        raise ValueError("unknown option %s: incremental must be 'timestep' or 'member'" % incremental)
    
    if incremental == "timestep":
        num_timesteps = None
    elif incremental == "member":
        num_ens_members = None
    
    exporter = {}
    
    ncf = netCDF4.Dataset(filename, 'w', format="NETCDF4")
    
    ncf.Conventions = "CF-1.7"
    ncf.title = "pysteps-generated nowcast"
    ncf.institution = "the pySTEPS community (https://pysteps.github.io)"
    ncf.source = "pysteps" # TODO: Add pySTEPS version here
    ncf.history = ""
    ncf.references = ""
    ncf.comment = ""
    
    h,w = shape
    
    ncf.createDimension("ens_number", size=num_ens_members)
    ncf.createDimension("time", size=num_timesteps)
    ncf.createDimension("y", size=h)
    ncf.createDimension("x", size=w)
    
    if metadata["unit"] == "mm/h":
        var_name = "precip_intensity"
        var_standard_name = None
        var_long_name = "instantaneous precipitation rate"
        var_unit = "mm h-1"
    elif metadata["unit"] == "mm":
        var_name = "hourly_precip_accum"
        var_standard_name = None
        var_long_name = "hourly precipitation accumulation"
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
    # TODO: Don't hard-code the unit.
    var_xc.units = 'm'
    
    var_yc = ncf.createVariable("yc", np.float32, dimensions=("y",))
    var_yc[:] = yr
    var_yc.axis = 'Y'
    var_yc.standard_name = "projection_y_coordinate"
    var_yc.long_name = "y-coordinate in Cartesian system"
    # TODO: Don't hard-code the unit.
    var_yc.units = 'm'
    
    X,Y = np.meshgrid(xr, yr)
    pr = pyproj.Proj(metadata["projection"])
    lon,lat = pr(X.flatten(), Y.flatten(), inverse=True)
    
    var_lon = ncf.createVariable("lon", np.float, dimensions=("y", "x"))
    var_lon[:] = lon
    var_lon.standard_name = "longitude"
    var_lon.long_name     = "longitude coordinate"
    # TODO: Don't hard-code the unit.
    var_lon.units         = "degrees_east"
    
    var_lat = ncf.createVariable("lat", np.float, dimensions=("y", "x"))
    var_lat[:] = lat
    var_lat.standard_name = "latitude"
    var_lat.long_name     = "latitude coordinate"
    # TODO: Don't hard-code the unit.
    var_lat.units         = "degrees_north"
    
    ncf.projection = metadata["projection"]
    
    grid_mapping_var_name,grid_mapping_name,grid_mapping_params = \
        _convert_proj4_to_grid_mapping(metadata["projection"])
    # skip writing the grid mapping if a matching name was not found
    if grid_mapping_var_name is not None:
        var_gm = ncf.createVariable(grid_mapping_var_name, np.int, dimensions=())
        var_gm.grid_mapping_name = grid_mapping_name
        for i in grid_mapping_params.items():
            var_gm.setncattr(i[0], i[1])
    
    var_ens_num = ncf.createVariable("ens_number", np.int, dimensions=("ens_number",))
    if incremental != "member":
        var_ens_num[:] = list(range(1, num_ens_members+1))
    var_ens_num.long_name = "ensemble member"
    var_ens_num.units = ""
    
    var_time = ncf.createVariable("time", np.int, dimensions=("time",))
    if incremental != "timestep":
        var_time[:] = [i*timestep*60 for i in range(1, num_timesteps+1)]
    var_time.long_name = "forecast time"
    startdate_str = datetime.strftime(startdate, "%Y-%m-%d %H:%M:%S")
    var_time.units = "seconds since %s" % startdate_str
    
    if num_ens_members == 1:
        var_F = ncf.createVariable(var_name, np.float32, 
                                   dimensions=("time", "y", "x"), zlib=True, 
                                   complevel=9)
    else:
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
    exporter["timestep"]  = timestep
    exporter["metadata"]  = metadata
    exporter["incremental"] = incremental
    exporter["num_timesteps"] = num_timesteps
    exporter["num_ens_members"] = num_ens_members
    exporter["shape"] = shape
    
    return exporter

def export_nowcast(F, exporter):
    """Write a nowcast dataset into a file.
    
    Parameters
    ----------
    exporter : dict
        An exporter object created with any initialization method impelmented 
        in this module.
    F : array_like
        The dataset to write. The required shape depends on the choice of the 
        'incremental' parameter the exporter was initialized with:
        
        +-------------------+-----------------------------------------------------+
        |    incremental    |                    required shape                   |
        +===================+=====================================================+
        |    None           |  (num_ens_members,num_timesteps,shape[0],shape[1])  |
        +-------------------+-----------------------------------------------------+
        |    'timestep'     |  (num_ens_members,shape[0],shape[1])                |
        +-------------------+-----------------------------------------------------+
        |    'member'       |  (num_timesteps,shape[0],shape[1])                  |
        +-------------------+-----------------------------------------------------+
    """
    if not netcdf4_imported:
        raise Exception("netCDF4 not imported")
    
    if exporter["incremental"] == None and \
       F.shape != (exporter["num_ens_members"], exporter["num_timesteps"], 
                   exporter["shape"][0], exporter["shape"][1]):
        raise ValueError("F has invalid shape")
    elif exporter["incremental"] == "timestep" and \
       F.shape != (exporter["num_ens_members"], exporter["shape"][0], 
                   exporter["shape"][1]):
        raise ValueError("F has invalid shape")
    elif exporter["incremental"] == "member" and \
        F.shape != (exporter["num_timesteps"], exporter["shape"][0], 
                    exporter["shape"][1]):
        raise ValueError("F has invalid shape")
    
    if exporter["method"] == "netcdf":
        _export_netcdf(F, exporter)
    else:
        raise ValueError("unknown exporter method %s" % exporter["method"])

def _export_netcdf(F, exporter):
    var_F = exporter["var_F"]
    
    if exporter["incremental"] == None:
        var_F[:] = F
    elif exporter["incremental"] == "timestep":
        var_F[:, var_F.shape[1], :, :] = F
        var_time = exporter["var_time"]
        var_time[len(var_time)-1] = len(var_time) * exporter["timestep"] * 60
    else:
        var_F[var_F.shape[0], :, :, :] = F
        var_ens_num = exporter["var_time"]
        var_ens_num[len(var_ens_num)-1] = len(var_ens_num)

# TODO: Write methods for converting Proj.4 projection definitions into CF grid 
# mapping attributes. Currently this has been implemented for the stereographic 
# projection.
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
    # TODO: implement more projection types here
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
        params["false_easting"]  = float(d["x_0"])
        params["false_northing"] = float(d["y_0"])
    else:
        return None,None,None
    
    return grid_mapping_var_name, grid_mapping_name, params
