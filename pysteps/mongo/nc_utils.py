"""
Refactored IO utilities for pysteps.
"""

import numpy as np
from pyproj import CRS
import netCDF4
import datetime
from typing import Optional
import io
from pathlib import Path


def replace_extension(filename: str, new_ext: str) -> str:
    return f"{filename.rsplit('.', 1)[0]}{new_ext}"


def convert_timestamps_to_datetimes(timestamps):
    """Convert POSIX timestamps to datetime objects."""
    return [
        datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        for ts in timestamps
    ]


def write_netcdf_file(
    file_path: Path,
    rain: np.ndarray,
    geo_data: dict,
    valid_times: list[datetime.datetime],
    ensembles: list[int] | None,
) -> None:
    """
    Write a set of rainfall grids to a CF-compliant NetCDF file using i2 data and scale_factor.

    Args:
        file_path (Path): Full path to the output file.
        rain (np.ndarray): Rainfall array. Shape is [ensemble, time, y, x] if ensembles is provided,
                           otherwise [time, y, x], with units in mm/h as float.
        geo_data (dict): Geospatial metadata (must include 'x', 'y', and optionally 'projection').
        valid_times (list[datetime.datetime]): List of timezone-aware valid times.
        ensembles (list[int] | None): Optional list of ensemble member IDs.
    """
    # Convert datetime to seconds since epoch
    time_stamps = [vt.timestamp() for vt in valid_times]

    x = geo_data["x"]
    y = geo_data["y"]
    projection = geo_data.get("projection", "EPSG:4326")
    rain_fill_value = -1

    with netCDF4.Dataset(file_path, mode="w", format="NETCDF4") as ds:
        # Define dimensions
        ds.createDimension("y", len(y))
        ds.createDimension("x", len(x))
        ds.createDimension("time", len(valid_times))
        if ensembles is not None:
            ds.createDimension("ensemble", len(ensembles))

        # Define coordinate variables
        x_var = ds.createVariable("x", "f4", ("x",))
        x_var[:] = x
        x_var.standard_name = "projection_x_coordinate"
        x_var.units = "m"

        y_var = ds.createVariable("y", "f4", ("y",))
        y_var[:] = y
        y_var.standard_name = "projection_y_coordinate"
        y_var.units = "m"

        t_var = ds.createVariable("time", "f8", ("time",))
        t_var[:] = time_stamps
        t_var.standard_name = "time"
        t_var.units = "seconds since 1970-01-01T00:00:00Z"
        t_var.calendar = "standard"

        if ensembles is not None:
            e_var = ds.createVariable("ensemble", "i4", ("ensemble",))
            e_var[:] = ensembles
            e_var.standard_name = "ensemble"
            e_var.units = "1"

        # Define the rainfall variable with proper fill_value
        rain_dims = (
            ("time", "y", "x") if ensembles is None else ("ensemble", "time", "y", "x")
        )
        rain_var = ds.createVariable(
            "rainfall",
            "i2",
            rain_dims,
            zlib=True,
            complevel=5,
            fill_value=rain_fill_value,
        )

        # Scale and store rainfall
        rain_scaled = np.where(
            np.isnan(rain), rain_fill_value, np.round(rain * 10).astype(np.int16)
        )
        rain_var[...] = rain_scaled

        # Metadata
        rain_var.scale_factor = 0.1
        rain_var.add_offset = 0.0
        rain_var.units = "mm/h"
        rain_var.long_name = "Rainfall rate"
        rain_var.grid_mapping = "projection"
        rain_var.coordinates = " ".join(rain_dims)

        # CRS
        crs = CRS.from_user_input(projection)
        cf_grid_mapping = crs.to_cf()
        spatial_ref = ds.createVariable("projection", "i4")
        for key, value in cf_grid_mapping.items():
            setattr(spatial_ref, key, value)

        # Global attributes
        ds.Conventions = "CF-1.8"
        ds.title = ""
        ds.institution = ""
        ds.references = ""
        ds.comment = ""


import io
import tempfile
import netCDF4
import os
import numpy as np
from pyproj import CRS


def make_netcdf_buffer(rain: np.ndarray, geo_data: dict, time: int) -> io.BytesIO:
    """
    Make the BytesIO netcdf object that is needed for writing to GridFS database
    Args:
        rain (np.ndarray): array of rain rates in mm/h as float
        geo_data (dict): spatial metadata
        time (int): seconds since 1970-01-01T00:00:00Z

    Returns:
        io.BytesIO: _description_
    """
    x = geo_data["x"]
    y = geo_data["y"]
    projection = geo_data.get("projection", "EPSG:4326")

    # Use NamedTemporaryFile to create a temp NetCDF file
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    # Create NetCDF file on disk
    ds = netCDF4.Dataset(tmp_path, mode="w", format="NETCDF4")

    # Define dimensions
    ds.createDimension("y", len(y))
    ds.createDimension("x", len(x))
    ds.createDimension("time", 1)

    # Coordinate variables
    y_var = ds.createVariable("y", "f4", ("y",))
    x_var = ds.createVariable("x", "f4", ("x",))
    t_var = ds.createVariable("time", "i8", ("time",))

    # Rainfall variable,
    # Expects a float input array and the packing to i2 is done by the netCDF4 library
    rain_var = ds.createVariable(
        "rainfall", "i2", ("time", "y", "x"), zlib=True, complevel=5, fill_value=-1
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
    ds.Conventions = "CF-1.8"
    ds.title = "Rainfall data"
    ds.institution = ""
    ds.references = ""
    ds.comment = ""

    ds.close()

    # Now read into memory
    with open(tmp_path, "rb") as f:
        nc_bytes = f.read()

    os.remove(tmp_path)
    return io.BytesIO(nc_bytes)


def generate_geo_dict(domain: dict) -> dict:
    """
    Generate the pysteps geo-spatial metadata from a domain dictionary.

    Args:
        domain (dict): pysteps_param domain dictionary

    Returns:
        dict: pysteps geo-data dictionary, or {} if required keys are missing
    """
    required_keys = {"n_cols", "n_rows", "p_size", "start_x", "start_y"}
    missing = required_keys - domain.keys()
    if missing:
        # Missing keys, return empty dict
        return {}

    ncols = domain.get("n_cols")
    nrows = domain.get("n_rows")
    psize = domain.get("p_size")
    start_x = domain.get("start_x")
    start_y = domain.get("start_y")

    x = [start_x + i * psize for i in range(ncols)]  # type: ignore
    y = [start_y + i * psize for i in range(nrows)]  # type: ignore

    out_geo = {}
    out_geo["x"] = x
    out_geo["y"] = y
    out_geo["xpixelsize"] = psize
    out_geo["ypixelsize"] = psize
    out_geo["x1"] = start_x
    out_geo["y1"] = start_y
    out_geo["x2"] = start_x + (ncols - 1) * psize  # type: ignore
    out_geo["y2"] = start_y + (nrows - 1) * psize  # type: ignore
    out_geo["projection"] = domain["projection"]["epsg"]
    out_geo["cartesian_unit"] = ("m",)
    out_geo["yorigin"] = ("lower",)
    out_geo["unit"] = "mm/h"
    out_geo["threshold"] = 0
    out_geo["transform"] = None

    return out_geo


def read_nc(buffer: bytes):
    """
    Read netCDF file from a memory buffer and return geo-referencing data and rain rates.

    :param buffer: Byte data of the NetCDF file from GridFS.
    :return: Tuple containing geo-referencing data, valid times, and rain rate array.
    """
    # Convert the byte buffer to a BytesIO object
    byte_stream = io.BytesIO(buffer)

    # Open the NetCDF dataset
    with netCDF4.Dataset("inmemory", mode="r", memory=byte_stream.getvalue()) as ds:

        # Extract geo-referencing data
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]

        domain = {}
        domain["ncols"] = len(x)
        domain["nrows"] = len(y)
        domain["psize"] = abs(x[1] - x[0])
        domain["start_x"] = x[0]
        domain["start_y"] = y[0]
        geo_data = generate_geo_dict(domain)

        # Convert timestamps to datetime
        valid_times = convert_timestamps_to_datetimes(ds.variables["time"][:])

        # Extract rain rates
        rain_rate = ds.variables["rainfall"][:]

        # Replace invalid data with NaN and squeeze dimensions of np.ndarray
        rain_rate = np.squeeze(rain_rate)
        rain_rate[rain_rate < 0] = np.nan

    return geo_data, valid_times, rain_rate


def validate_keys(keys, mandatory_keys):
    """Validate the presence of mandatory keys."""
    missing_keys = [key for key in mandatory_keys if key not in keys]
    if missing_keys:
        raise KeyError(f"Missing mandatory keys: {', '.join(missing_keys)}")


def make_nc_name(
    domain: str,
    prod: str,
    valid_time: datetime.datetime,
    base_time: Optional[datetime.datetime] = None,
    ens: Optional[int] = None,
    name_template: Optional[str] = None,
) -> str:
    """
    Generate a unique name for a single rain field using a formatting template.

    Default templates:
        Forecast products: "$D_$P_$V{%Y%m%dT%H%M%S}_$B{%Y%m%dT%H%M%S}_$E.nc"
        QPE products: "$D_$P_$V{%Y%m%dT%H%M%S}.nc"

    Where:
        $D = Domain name
        $P = Product name
        $V = Valid time (with strftime format)
        $B = Base time (with strftime format)
        $E = Ensemble number (zero-padded 2-digit)

    Returns:
        str: Unique NetCDF file name.
    """

    if not isinstance(valid_time, datetime.datetime):
        raise TypeError(f"valid_time must be datetime, got {type(valid_time)}")

    if base_time is not None and not isinstance(base_time, datetime.datetime):
        raise TypeError(f"base_time must be datetime or None, got {type(base_time)}")

    # Default template logic
    if name_template is None:
        name_template = "$D_$P_$V{%Y-%m-%dT%H:%M:%S}"
        if base_time is not None:
            name_template += "_$B{%Y-%m-%dT%H:%M:%S}"
        if ens is not None:
            name_template += "_$E"
        name_template += ".nc"

    result = name_template

    # Ensure timezone-aware times
    if valid_time.tzinfo is None:
        valid_time = valid_time.replace(tzinfo=datetime.timezone.utc)
    if base_time is not None and base_time.tzinfo is None:
        base_time = base_time.replace(tzinfo=datetime.timezone.utc)

    # Replace flags
    while "$" in result:
        flag_posn = result.find("$")
        if flag_posn == -1:
            break
        f_type = result[flag_posn + 1]

        try:
            if f_type in ["V", "B"]:
                field_start = result.find("{", flag_posn + 1)
                field_end = result.find("}", flag_posn + 1)
                if field_start == -1 or field_end == -1:
                    raise ValueError(
                        f"Missing braces for format of '${f_type}' in template."
                    )

                fmt = result[field_start + 1 : field_end]
                if f_type == "V":
                    time_str = valid_time.strftime(fmt)
                elif f_type == "B" and base_time is not None:
                    time_str = base_time.strftime(fmt)
                else:
                    time_str = ""

                result = result[:flag_posn] + time_str + result[field_end + 1 :]

            elif f_type == "D":
                result = result[:flag_posn] + domain + result[flag_posn + 2 :]
            elif f_type == "P":
                result = result[:flag_posn] + prod + result[flag_posn + 2 :]
            elif f_type == "E" and ens is not None:
                result = result[:flag_posn] + f"{ens:02d}" + result[flag_posn + 2 :]
            else:
                raise ValueError(
                    f"Unknown or unsupported flag '${f_type}' in template."
                )
        except Exception as e:
            raise ValueError(f"Error processing flag '${f_type}': {e}")

    return result
