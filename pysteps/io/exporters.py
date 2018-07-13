"""Methods for writing forecasts of 2d precipitation fields into various file 
formats."""

import numpy as np
from datetime import datetime
# TODO: Check that the needed modules are imported when trying to use them
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

# TODO: This is a draft version of the exporter. Revise the variable names and 
# the structure of the file if necessary.
def write_nowcast_netCDF(F, filename, startdate, timestep, metadata):
    """Write a forecast or a forecast ensemble into a netCDF file in the CF 1.7 
    format.
    
    Parameters
    ----------
    F : array_like
        Three- or four-dimensional array containing the forecast F(t,x,y) or a 
        forecast ensemble F(i,t,y,x), where i is ensemble member, t is lead time 
        and x and y are grid coordinates. The time index t is assumed to be in 
        ascending order, and the time steps are assumed to be regular with no 
        gaps.
    filename : str
        Name of the output file.
    startdate : datetime.datetime
        Start date of the forecast.
    timestep : int
        Time step of the forecast (minutes).
    metadata : dict
        Metadata dictionary containing the projection,x1,x2,y1,y2 and unit 
        attributes described in the documentation of pysteps.io.importers.
    """
    if not netcdf4_imported:
        raise Exception("netCDF4 not imported")
    if not pyproj_imported:
        raise Exception("pyproj not imported")
    
    if len(F.shape) not in [3, 4]:
        raise ValueError("F has invalid dimensions: must be a three- or four-dimensional array")
    
    ds = netCDF4.Dataset(filename, 'w', format="NETCDF4")
    
    ds.Conventions = "CF-1.7"
    ds.title = "pysteps-generated nowcast"
    ds.institution = "the pySTEPS community (https://pysteps.github.io)"
    ds.source = "pysteps" # TODO: Add pySTEPS version here
    ds.history = ""
    ds.references = ""
    ds.comment = ""
    
    if len(F.shape) == 3:
        time = ds.createDimension("time", size=F.shape[0])
        y = ds.createDimension("y",    size=F.shape[1])
        x = ds.createDimension("x",    size=F.shape[2])
        w = F.shape[2]
        h = F.shape[1]
        num_timesteps = F.shape[0]
    else:
        ens_number  = ds.createDimension("ens_number", size=F.shape[0])
        time = ds.createDimension("time", size=F.shape[1])
        y = ds.createDimension("y",    size=F.shape[2])
        x = ds.createDimension("x",    size=F.shape[3])
        w = F.shape[3]
        h = F.shape[2]
        num_timesteps = F.shape[1]
    
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
    
    # TODO: Always save the dataset into a four-dimensional array?
    if len(F.shape) == 3:
        var_F = ds.createVariable(var_name, np.float32, 
                                  dimensions=("time", "y", "x"), zlib=True, 
                                  complevel=9)
    else:
        var_F = ds.createVariable(var_name, np.float32, 
                                  dimensions=("ens_number", "time", "y", "x"), 
                                  zlib=True, complevel=9)
    var_F[:] = F
    
    xr = np.linspace(metadata["x1"], metadata["x2"], w+1)[:-1]
    xr += 0.5 * (xr[1] - xr[0])
    yr = np.linspace(metadata["y1"], metadata["y2"], h+1)[:-1]
    yr += 0.5 * (yr[1] - yr[0])
    
    var_xc = ds.createVariable("xc", np.float32, dimensions=("x",))
    var_xc[:] = xr
    var_xc.axis = 'X'
    var_xc.standard_name = "projection_x_coordinate"
    var_xc.long_name = "x-coordinate in Cartesian system"
    # TODO: Don't hard-code the unit.
    var_xc.units = 'm'
    
    var_yc = ds.createVariable("yc", np.float32, dimensions=("x",))
    var_yc[:] = yr
    var_yc.axis = 'Y'
    var_yc.standard_name = "projection_y_coordinate"
    var_yc.long_name = "y-coordinate in Cartesian system"
    # TODO: Don't hard-code the unit.
    var_yc.units = 'm'
    
    X,Y = np.meshgrid(xr, yr)
    pr = pyproj.Proj(metadata["projection"])
    lon,lat = pr(X.flatten(), Y.flatten(), inverse=True)
    
    var_lon = ds.createVariable("lon", np.float, dimensions=("y", "x"))
    var_lon[:] = lon
    var_lon.standard_name = "longitude"
    var_lon.long_name     = "longitude coordinate"
    # TODO: Don't hard-code the unit.
    var_lon.units         = "degrees_east"
    
    var_lat = ds.createVariable("lat", np.float, dimensions=("y", "x"))
    var_lat[:] = lat
    var_lat.standard_name = "latitude"
    var_lat.long_name     = "latitude coordinate"
    # TODO: Don't hard-code the unit.
    var_lat.units         = "degrees_north"
    
    # TODO: Extract the map projection from the metadata and save it to the grid 
    # mapping attributes in the NetCDF file.
    # Currently the projection string is saved into the "projection" attribute  
    # at the top level.
    ds.projection = metadata["projection"]
    
    var_time = ds.createVariable("time", np.int, dimensions=("time",))
    var_time[:] = [i*timestep*60 for i in range(1, num_timesteps+1)]
    var_time.long_name = "forecast time"
    startdate_str = datetime.strftime(startdate, "%Y-%m-%d %H:%M:%S")
    var_time.units = "seconds since %s" % startdate_str
    
    if var_standard_name is not None:
        var_F.standard_name = var_standard_name
    var_F.long_name = var_long_name
    var_F.coordinates = "y x"
    var_F.units = var_unit
    
    ds.close()

# TODO: Write a method for converting a Proj.4 projection definition into CF 
# grid mapping attributes.
def _convert_proj4_to_grid_mapping():
    pass
