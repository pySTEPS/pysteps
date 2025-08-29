import xarray as xr
import pandas as pd
import numpy as np
import datetime
import logging


def generate_geo_dict(domain):
    ncols = domain.get("n_cols")
    nrows = domain.get("n_rows")
    psize = domain.get("p_size")
    start_x = domain.get("start_x")
    start_y = domain.get("start_y")
    x = [start_x + i * psize for i in range(ncols)]
    y = [start_y + i * psize for i in range(nrows)]

    out_geo = {}
    out_geo["x"] = x
    out_geo["y"] = y
    out_geo["xpixelsize"] = psize
    out_geo["ypixelsize"] = psize
    out_geo["x1"] = start_x
    out_geo["y1"] = start_y
    out_geo["x2"] = start_x + (ncols - 1) * psize
    out_geo["y2"] = start_y + (nrows - 1) * psize
    out_geo["projection"] = domain["projection"]["epsg"]
    out_geo["cartesian_unit"] = "m"
    out_geo["yorigin"] = "lower"
    out_geo["unit"] = "mm/h"
    out_geo["threshold"] = 0
    out_geo["transform"] = None

    return out_geo


def generate_geo_dict_xy(x: np.ndarray, y: np.ndarray, epsg: str):
    n_cols = x.size
    n_rows = y.size

    out_geo = {}
    out_geo["xpixelsize"] = (x[-1] - x[0]) / (n_cols - 1)
    out_geo["ypixelsize"] = (y[-1] - y[0]) / (n_rows - 1)
    out_geo["x1"] = x[0]
    out_geo["x2"] = x[-1]
    out_geo["y1"] = y[0]
    out_geo["y2"] = y[-1]
    out_geo["projection"] = epsg
    out_geo["cartesian_unit"] = "m"
    out_geo["yorigin"] = "lower"
    out_geo["unit"] = "mm/h"
    out_geo["threshold"] = 0
    out_geo["transform"] = None

    return out_geo


def read_qpe_netcdf(file_path):
    """
    Read WRNZ QPE NetCDF file and return xarray Dataset of rain rate with:
    - 'rain' variable in [time, yc, xc] order
    - time as timezone-aware UTC datetimes
    - EPSG:2193 (NZTM2000) projection info added using CF conventions
    - Return None on error reading the file

    Assumes that the input file is rain rate in [t,y,x] order
    """

    try:
        ds = xr.open_dataset(file_path, decode_times=True)
        ds.load()

        # Make the times timezone-aware UTC
        time_values = ds["time"].values.astype("datetime64[ns]")
        time_utc = pd.DatetimeIndex(time_values, tz=datetime.UTC)
        ds["time"] = ("time", time_utc)

        # Rename
        ds = ds.rename({"rainfall": "rain"})

        # Define CF-compliant grid mapping for EPSG:2193
        crs = xr.DataArray(
            0,
            attrs={
                "grid_mapping_name": "transverse_mercator",
                "scale_factor_at_central_meridian": 0.9996,
                "longitude_of_central_meridian": 173.0,
                "latitude_of_projection_origin": 0.0,
                "false_easting": 1600000.0,
                "false_northing": 10000000.0,
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257222101,
                "spatial_ref": "EPSG:2193",
            },
            name="NZTM2000",
        )

        ds["NZTM2000"] = crs
        ds["rain"].attrs["grid_mapping"] = "NZTM2000"

        ds = ds[["rain", "NZTM2000"]]
        ds = ds.assign_coords(time=ds["time"], yc=ds["y"], xc=ds["x"])

        return ds

    except (ValueError, OverflowError, TypeError) as e:
        logging.warning(f"Failed to read {file_path}: {e}")
        return None
