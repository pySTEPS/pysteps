"""Methods for importing nowcast files.

The methods in this module implement the following interface:

  import_xxx(filename, optional arguments)

where xxx is the name (or abbreviation) of the file format and filename is the
name of the input file.

The output of each method is a two-element tuple containing the nowcast array 
and a metadata dictionary.

The metadata dictionary contains the following mandatory key-value pairs:
    projection   PROJ.4-compatible projection definition
    x1           x-coordinate of the lower-left corner of the data raster (meters)
    y1           y-coordinate of the lower-left corner of the data raster (meters)
    x2           x-coordinate of the upper-right corner of the data raster (meters)
    y2           y-coordinate of the upper-right corner of the data raster (meters)
    xpixelsize   grid resolution in x-direction (meters)
    ypixelsize   grid resolution in y-direction (meters)
    yorigin      a string specifying the location of the first element in
                 the data raster w.r.t. y-axis:
                 'upper' = upper border
                 'lower' = lower border
    institution  name of the institution who provides the data
    timestep     time step of the input data (minutes)
    unit         the unit of the data: 'mm/h', 'mm' or 'dBZ'
"""


try:
    import netCDF4
    netcdf4_imported = True
except ImportError:
    netcdf4_imported = False

def import_netcdf_pysteps(filename, **kwargs):
    """the CF 1.7 format"""
    if not netcdf4_imported:
        raise Exception("netCDF4 not imported")
    
    ds = netCDF4.Dataset(filename, 'r')
    
    var_names = list(ds.variables.keys())
    
    if "precip_intensity" in var_names:
        R = ds.variables["precip_intensity"]
    elif "hourly_precip_accum" in var_names:
        R = ds.variables["hourly_precip_accum"]
    elif "reflectivity" in var_names:
        R = ds.variables["reflectivity"]
    else:
        raise Exception("the netCDF file does not contain any supported variable name ('precip_intensity', 'hourly_precip_accum', or 'reflectivity')")
    
    R = R[...].squeeze().astype(float)
    
    projdef = ""
    if "polar_stereographic" in var_names:
        vn = "polar_stereographic"
    
    attr_dict = {}
    for attr_name in ds.variables[vn].ncattrs():
        attr_dict[attr_name] = ds[vn].getncattr(attr_name)
    
    metadata = {}
    proj_str = _convert_grid_mapping_to_proj4(attr_dict)
    # TODO: Read the metadata to the dictionary.
    
    ds.close()
    
    return R,metadata

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
