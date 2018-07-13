"""Methods for writing forecasts of 2d precipitation fields into various file 
formats."""

import numpy as np
try:
  import netCDF4
  netcdf4_exported = True
except ImportError:
  netcdf4_exported = False

# TODO: This is a first draft version of the exporter. Revise the variable names 
# and the structure of the file if necessary.
def write_nowcast_netCDF(F, filename, metadata):
    """Write a forecast or a forecast ensemble into a netCDF file in the CF 1.7 
    format.
    
    Parameters
    ----------
    F : array-like
      Three- or four-dimensional array containing the forecast F(t,x,y) or a 
      forecast ensemble F(i,t,y,x), where i is ensemble member, t is lead time 
      and x and y are grid coordinates.
    filename : str
      Name of the output file.
    """
    if len(F.shape) not in [3, 4]:
        raise ValueError("F has invalid dimensions: must be a three- or four-dimensional array")
    
    ds = netCDF4.Dataset(filename, 'w', format="NETCDF4")
    
    if len(F.shape) == 3:
        time = ds.createDimension("time", size=F.shape[0])
        y    = ds.createDimension("y",    size=F.shape[1])
        x    = ds.createDimension("x",    size=F.shape[2])
    else:
        ens_number  = ds.createDimension("ens_number", size=F.shape[0])
        time = ds.createDimension("time", size=F.shape[1])
        y    = ds.createDimension("y",    size=F.shape[2])
        x    = ds.createDimension("x",    size=F.shape[3])
    
    if metadata["unit"] == "mm/h":
        var_name = "precip_intensity"
    elif metadata["unit"] == "mm":
        var_name = "hourly_precip_accum"
    elif metadata["unit"] == "dBZ":
        var_name = "reflectivity"
    else:
        raise ValueError("unknown unit %s" % metadata["unit"])
    
    if len(F.shape) == 3:
        var_F = ds.createVariable(var_name, np.float32, 
                                  dimensions=("time", "y", "x"), zlib=True, 
                                  complevel=9)
    else:
        var_F = ds.createVariable(var_name, np.float32, 
                                  dimensions=("ens_number", "time", "y", "x"), 
                                  zlib=True, complevel=9)
    var_F[:] = F
    
    ds.close()

