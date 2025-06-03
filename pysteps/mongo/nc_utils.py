"""
    Refactored IO utilities for pysteps.
"""
import numpy as np
from pyproj import CRS
import netCDF4
from datetime import datetime, timezone
from typing import Optional
import io

def replace_extension(filename: str, new_ext: str) -> str:
    return f"{filename.rsplit('.', 1)[0]}{new_ext}"

def convert_timestamps_to_datetimes(timestamps):
    """Convert POSIX timestamps to datetime objects."""
    return [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]


def write_netcdf(rain: np.ndarray, geo_data: dict, time: int):
    """
    Write rain data as a NetCDF4 memory buffer.

    :param buffer: A BytesIO buffer to store the NetCDF data.
    :param rain: Rainfall data as a NumPy array.
    :param geo_data: Dictionary containing geo-referencing data with keys:
                     'x', 'y', 'projection', and other metadata.
    :param time: POSIX timestamp representing the time dimension.
    :return: The BytesIO buffer containing the NetCDF data.
    """
    x = geo_data['x']
    y = geo_data['y']
    # Default to WGS84 if not provided
    projection = geo_data.get('projection', 'EPSG:4326')

    # Create an in-memory NetCDF dataset
    ds = netCDF4.Dataset('inmemory.nc', mode='w', memory=1024)

    # Define dimensions
    y_dim = ds.createDimension("y", len(y))
    x_dim = ds.createDimension("x", len(x))
    t_dim = ds.createDimension("time", 1)

    # Define coordinate variables
    y_var = ds.createVariable("y", "f4", ("y",))
    x_var = ds.createVariable("x", "f4", ("x",))
    t_var = ds.createVariable("time", "i8", ("time",))

    # Define rain variable
    rain_var = ds.createVariable(
        "rainfall", "i2", ("time", "y", "x"), zlib=True
    )  # int16 with a fill value
    rain_var.scale_factor = 0.1
    rain_var.add_offset = 0.0
    rain_var.units = "mm/h"
    rain_var.long_name = "Rainfall rate"
    rain_var.grid_mapping = "projection"

    # Assign coordinate values
    y_var[:] = y
    y_var.standard_name = "projection_y_coordinate"
    y_var.units = "m"

    x_var[:] = x
    x_var.standard_name = "projection_x_coordinate"
    x_var.units = "m"

    t_var[:] = [time]
    t_var.standard_name = "time"
    t_var.units = "seconds since 1970-01-01T00:00:00Z"

    # Handle NaNs in rain data and assign to variable
    rain[np.isnan(rain)] = -1
    rain_var[0,:, :] = rain

    # Define spatial reference (CRS)
    crs = CRS.from_user_input(projection)
    cf_grid_mapping = crs.to_cf()

    # Create spatial reference variable
    spatial_ref = ds.createVariable("projection", "i4")
    for key, value in cf_grid_mapping.items():
        setattr(spatial_ref, key, value)

    # Add global attributes
    ds.Conventions = "CF-1.7"
    ds.title = "Rainfall data"
    ds.institution = "Weather Radar New Zealand Ltd"
    ds.references = ""
    ds.comment = ""
    return ds.close()

import io
import tempfile
import netCDF4
import os
import numpy as np
from pyproj import CRS

def write_netcdf_io(rain: np.ndarray, geo_data: dict, time: int) -> io.BytesIO:
    """
    Write a NetCDF file to a temporary file, read it into memory, and return a BytesIO buffer.
    """

    x = geo_data['x']
    y = geo_data['y']
    projection = geo_data.get('projection', 'EPSG:4326')

    # Use NamedTemporaryFile to create a temp NetCDF file
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    # Create NetCDF file on disk
    ds = netCDF4.Dataset(tmp_path, mode='w', format='NETCDF4')

    # Define dimensions
    ds.createDimension("y", len(y))
    ds.createDimension("x", len(x))
    ds.createDimension("time", 1)

    # Coordinate variables
    y_var = ds.createVariable("y", "f4", ("y",))
    x_var = ds.createVariable("x", "f4", ("x",))
    t_var = ds.createVariable("time", "i8", ("time",))

    # Rainfall variable
    rain_var = ds.createVariable(
        "rainfall", "i2", ("time", "y", "x"),
        zlib=True, complevel=5, fill_value=-1
    )
    rain_var.scale_factor = 0.1
    rain_var.add_offset = 0.0
    rain_var.units = "mm/h"
    rain_var.long_name = "Rainfall rate"
    rain_var.grid_mapping = "projection"

    # Assign values
    y_var[:] = y
    x_var[:] = x
    t_var[:] = [time]
    rain_var[0, :, :] = np.nan_to_num(rain, nan=-1)

    y_var.standard_name = "projection_y_coordinate"
    y_var.units = "m"
    x_var.standard_name = "projection_x_coordinate"
    x_var.units = "m"
    t_var.standard_name = "time"
    t_var.units = "seconds since 1970-01-01T00:00:00Z"

    # CRS
    crs = CRS.from_user_input(projection)
    cf_grid_mapping = crs.to_cf()
    spatial_ref = ds.createVariable("projection", "i4")
    for key, value in cf_grid_mapping.items():
        setattr(spatial_ref, key, value)

    # Global attributes
    ds.Conventions = "CF-1.7"
    ds.title = "Rainfall data"
    ds.institution = "Weather Radar New Zealand Ltd"
    ds.references = ""
    ds.comment = ""

    ds.close()

    # Now read into memory
    with open(tmp_path, "rb") as f:
        nc_bytes = f.read()

    os.remove(tmp_path)
    return io.BytesIO(nc_bytes)


def generate_geo_data(x, y, projection='EPSG:2193'):
    """Generate geo-referencing data."""
    return {
        "projection": projection,
        "x": x,
        "y": y,
        "x1": np.round(x[0],decimals=0),
        "x2": np.round(x[-1],decimals=0),
        "y1": np.round(y[0],decimals=0),
        "y2": np.round(y[-1],decimals=0),
        "xpixelsize": np.round(x[1] - x[0],decimals=0),
        "ypixelsize": np.round(y[1] - y[0],decimals=0),
        "cartesian_unit": 'm',
        "yorigin": 'lower',
        "unit": 'mm/h',
        "transform": None,
        "threshold": 0.1,
        "zerovalue": 0
    }


def read_nc(buffer: bytes):
    """
    Read netCDF file from a memory buffer and return geo-referencing data and rain rates.

    :param buffer: Byte data of the NetCDF file from GridFS.
    :return: Tuple containing geo-referencing data, valid times, and rain rate array.
    """
    # Convert the byte buffer to a BytesIO object
    byte_stream = io.BytesIO(buffer)

    # Open the NetCDF dataset
    with netCDF4.Dataset('inmemory', mode='r', memory=byte_stream.getvalue()) as ds:
        # Extract geo-referencing data
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        geo_data = generate_geo_data(x, y)

        # Convert timestamps to datetime
        valid_times = convert_timestamps_to_datetimes(ds.variables["time"][:])

        # Extract rain rates
        rain_rate = ds.variables["rainfall"][:]

        # Replace invalid data with NaN and squeeze dimensions
        rain_rate = np.squeeze(rain_rate)
        rain_rate[rain_rate < 0] = np.nan
        valid_times = np.squeeze(valid_times)

    return geo_data, valid_times, rain_rate


def validate_keys(keys, mandatory_keys):
    """Validate the presence of mandatory keys."""
    missing_keys = [key for key in mandatory_keys if key not in keys]
    if missing_keys:
        raise KeyError(f"Missing mandatory keys: {', '.join(missing_keys)}")

def make_nc_name_dt(out_file_name, name, out_product, valid_time, base_time, iens):

    vtime = valid_time
    if vtime.tzinfo is None:
        vtime = vtime.replace(tzinfo=timezone.utc)
    vtime_stamp = vtime.timestamp() 

    if base_time is not None:    
        btime = base_time
        if btime.tzinfo is None:
            btime = btime.replace(tzinfo=timezone.utc)
        btime_stamp = btime.timestamp()
    else:
        btime_stamp = None 
        
    fx_file_name = make_nc_name(
        out_file_name, name, out_product, vtime_stamp, btime_stamp, iens)
    return fx_file_name


def make_nc_name(name_template: str, name: str, prod: str, valid_time: int,
                 base_time: Optional[int] = None, ens: Optional[int] = None) -> str:
    """
    Generate a file name using a template.

    :param name_template: Template for the file name
    :param name: Name of the domain - Mandatory
    :param prod: Name of the product - Mandatory
    :param valid_time: Valid time of the field - Mandatory
    :param run_time: NWP run time - Optional
    :param ens: Ensemble member - Optional
    :return: String with the file name
    """
    result = name_template

    # Set up the valid time
    vtime_info = datetime.fromtimestamp(valid_time, tz=timezone.utc)

    # Set up the NWP base time if available
    btime_info = datetime.fromtimestamp(
        base_time, tz=timezone.utc) if base_time is not None else None

    has_flag = True
    while has_flag:
        # Search for a flag
        flag_posn = result.find("$")
        if flag_posn == -1:
            has_flag = False
        else:
            # Get the field type
            f_type = result[flag_posn + 1]

            try:
                # Add the valid and base times 
                if f_type in ['V', 'B']:
                    # Get the required format string
                    field_start = result.find("{", flag_posn + 1)
                    field_end = result.find("}", flag_posn + 1)
                    if field_start == -1 or field_end == -1:
                        raise ValueError(f"Invalid time format for flag '${
                                         f_type}' in template.")

                    time_format = result[field_start + 1:field_end]
                    if f_type == 'V':
                        date_str = vtime_info.strftime(time_format)
                    elif f_type == 'B' and btime_info:
                        date_str = btime_info.strftime(time_format)
                    else:
                        date_str = ""

                    # Replace the format field with the formatted time
                    result = result[:flag_posn] + \
                        date_str + result[field_end + 1:]
                elif f_type == 'P':
                    result = result[:flag_posn] + prod + result[flag_posn + 2:]
                elif f_type == 'N':
                    result = result[:flag_posn] + name + result[flag_posn + 2:]
                elif f_type == 'E' and ens is not None:
                    result = result[:flag_posn] + \
                        f"{ens:02d}" + result[flag_posn + 2:]
                else:
                    raise ValueError(f"Unknown or unsupported flag '${
                                     f_type}' in template.")
            except Exception as e:
                raise ValueError(f"Error processing flag '${
                                 f_type}': {str(e)}")

    return result