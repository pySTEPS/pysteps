"""
Output an nc file with past and forcast ensemble 
"""

from models.mongo_access import get_db, get_config
from models.nc_utils import convert_timestamps_to_datetimes, make_nc_name
from pymongo import MongoClient
import logging
import argparse
import pymongo
from gridfs import GridFSBucket, NoFile
import numpy as np
import datetime
import netCDF4
import pandas as pd
from pathlib import Path
import xarray as xr
from pyproj import CRS
import io

import numpy as np
import netCDF4
from pathlib import Path


def write_rainfall_netcdf(filename: Path, rainfall: np.ndarray,
                          x: np.ndarray, y: np.ndarray,
                          time: list, ensemble: np.ndarray):
    """
    Write rainfall data to NetCDF using low-level netCDF4 interface.
    - rainfall: 4D np.ndarray (ensemble, time, y, x), float32, mm/h with NaNs
    - x, y: 1D arrays of projection coordinates in meters
    - time: list of timezone-aware datetime.datetime objects
    - ensemble: 1D array of ensemble member IDs (int)
    """

    n_ens, n_times, ny, nx = rainfall.shape
    assert len(time) == n_times
    assert len(ensemble) == n_ens

    with netCDF4.Dataset(filename, "w", format="NETCDF4") as ds:
        # Create dimensions
        ds.createDimension("ensemble", n_ens)
        ds.createDimension("time", n_times)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)

        # Coordinate variables
        x_var = ds.createVariable("x", "f4", ("x",))
        y_var = ds.createVariable("y", "f4", ("y",))
        t_var = ds.createVariable("time", "i4", ("time",))
        ens_var = ds.createVariable("ensemble", "i4", ("ensemble",))

        x_var[:] = x
        y_var[:] = y
        ens_var[:] = ensemble
        t_var[:] = netCDF4.date2num(
            time, units="seconds since 1970-01-01T00:00:00", calendar="standard")

        x_var.units = "m"
        x_var.standard_name = "projection_x_coordinate"
        y_var.units = "m"
        y_var.standard_name = "projection_y_coordinate"
        t_var.units = "seconds since 1970-01-01 00:00:00"
        t_var.standard_name = "time"
        ens_var.long_name = "ensemble member"

        # CRS variable (dummy scalar)
        crs_var = ds.createVariable("crs", "i4")
        crs_var.grid_mapping_name = "transverse_mercator"
        crs_var.scale_factor_at_central_meridian = 0.9996
        crs_var.longitude_of_central_meridian = 173.0
        crs_var.latitude_of_projection_origin = 0.0
        crs_var.false_easting = 1600000.0
        crs_var.false_northing = 10000000.0
        crs_var.semi_major_axis = 6378137.0
        crs_var.inverse_flattening = 298.257222101
        crs_var.spatial_ref = "EPSG:2193"

        # Rainfall variable (compressed int16 with scale)
        rain_var = ds.createVariable(
            "rainfall", "i2", ("ensemble", "time", "y", "x"),
            zlib=True, complevel=5, fill_value=-1
        )
        rain_var.scale_factor = 0.1
        rain_var.add_offset = 0.0
        rain_var.units = "mm/h"
        rain_var.long_name = "Rainfall rate"
        rain_var.grid_mapping = "crs"

        rainfall[np.isnan(rainfall)] = -1
        rain_var[:, :, :, :] = rainfall


def is_valid_iso8601(time_str: str) -> bool:
    """Check if the given string is a valid ISO 8601 datetime."""
    try:
        datetime.datetime.fromisoformat(time_str)
        return True
    except ValueError:
        return False


def get_filenames(db: MongoClient, name: str, query: dict):
    meta_coll = db[f"{name}.rain.files"]

    # Fetch matching filenames and metadata in a single query
    fields_projection = {"_id": 1, "filename": 1, "metadata": 1}
    results = meta_coll.find(query, projection=fields_projection).sort(
        "filename", pymongo.ASCENDING)
    files = []
    for doc in results:
        record = {
            "valid_time": doc["metadata"]["valid_time"],
            "base_time": doc["metadata"]["base_time"],
            "ensemble": doc["metadata"]["ensemble"],
            "_id": doc["_id"],
            "filename": doc["filename"]
        }
        files.append(record)

    files_df = pd.DataFrame(files)
    return files_df


def main():
    parser = argparse.ArgumentParser(
        description="Write rainfall fields to a netCDF file")
    parser.add_argument('-n', '--name', required=True,
                        help='Domain name (e.g., AKL)')
    parser.add_argument('-b', '--base_time', type=str, required=True,
                        help='Base time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-d', '--directory', required=True, type=Path,
                        help='Path to output directory for the figures')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Validate start and end time and read them in
    if args.base_time and is_valid_iso8601(args.base_time):
        base_time = datetime.datetime.fromisoformat(str(args.base_time))
        if base_time.tzinfo is None:
            base_time = base_time.replace(tzinfo=datetime.timezone.utc)
    else:
        logging.error(
            "Invalid base time format. Please provide a valid ISO 8601 time string.")
        return

    file_dir = args.directory
    if not file_dir.exists():
        logging.error(f"Invalid output diectory {file_dir}")
        return

    name = args.name
    db = get_db()

    # Get the domain geometry
    config = get_config(db, name)
    nwpblend_config = config["output"]["nwpblend"]
    n_ens = nwpblend_config.get("n_ens_members")
    n_fx = nwpblend_config.get("n_forecasts")
    n_qpe = n_fx
    ts_seconds = config["pysteps"]["timestep"]
    ts = datetime.timedelta(seconds=ts_seconds)

    # Get the file names for the input data
    start_qpe = base_time - n_qpe * ts
    end_qpe = base_time
    query = {
        "metadata.product": "QPE",
        "metadata.valid_time": {"$gte": start_qpe, "$lte": end_qpe}
    }
    qpe_df = get_filenames(db, name, query)

    start_blend = base_time
    end_blend = base_time + n_fx*ts

    query = {
        "metadata.product": "nwpblend",
        "metadata.valid_time": {"$gt": base_time, "$lte": end_blend},
        "metadata.base_time": base_time
    }
    blend_df = get_filenames(db, name, query)

    qpe_fields = []
    qpe_times = []

    bucket_name = f"{name}.rain"
    bucket = GridFSBucket(db, bucket_name=bucket_name)

    for index, row in qpe_df.iterrows():
        filename = row["filename"]
        with bucket.open_download_stream_by_name(filename) as stream:
            buffer = stream.read()
            byte_stream = io.BytesIO(buffer)
            ds = netCDF4.Dataset('inmemory', mode='r',
                                 memory=byte_stream.getvalue())

        # Extract rain rate and handle 3D (time, y, x) or 2D (y, x)
        rain_rate = ds.variables["rainfall"][:]
        if rain_rate.ndim == 3:
            rain_rate = rain_rate[0, :, :]  # Take first time slice if present

        # Get valid time (assuming one timestamp per file)
        time_var = ds.variables["time"][:]
        valid_time = convert_timestamps_to_datetimes(
            time_var)[0]  # e.g., returns a list

        if index == 0:
            y_ref = ds.variables["y"][:]
            x_ref = ds.variables["x"][:]
        else:
            assert np.allclose(ds.variables["y"][:], y_ref)
            assert np.allclose(ds.variables["x"][:], x_ref)

        # Accumulate
        qpe_fields.append(rain_rate)
        qpe_times.append(valid_time)

    # Convert to xarray.DataArray
    qpe_array = xr.DataArray(
        data=np.stack(qpe_fields),  # shape: (time, y, x)
        coords={"time": qpe_times, "y": y_ref, "x": x_ref},
        dims=["time", "y", "x"],
        name="qpe"
    )

    # Ensure sorted and aligned valid_times across all ensemble members
    ensembles = np.sort(blend_df["ensemble"].unique())

    blend_times = np.sort(blend_df["valid_time"].unique())
    # ensures tz-aware datetime64[ns, UTC]
    blend_times = pd.to_datetime(blend_times, utc=True)
    # convert to native datetime.datetime
    blend_times = [dt.to_pydatetime() for dt in blend_times]

    n_ens = len(ensembles)
    n_time = len(blend_times)
    ny, nx = y_ref.shape[0], x_ref.shape[0]

    # Initialize a 4D array (ensemble, time, y, x)
    blend_data = np.full((n_ens, n_time, ny, nx), np.nan, dtype=np.float32)

    # Mapping from value to index
    ensemble_to_idx = {ens: i for i, ens in enumerate(ensembles)}
    time_to_idx = {vt: i for i, vt in enumerate(blend_times)}

    for index, row in blend_df.iterrows():
        filename = row["filename"]
        ensemble = row["ensemble"]
        with bucket.open_download_stream_by_name(filename) as stream:
            buffer = stream.read()
            byte_stream = io.BytesIO(buffer)
            ds = netCDF4.Dataset('inmemory', mode='r',
                                 memory=byte_stream.getvalue())

        rain_rate = ds.variables["rainfall"][:]
        if rain_rate.ndim == 3:
            rain_rate = rain_rate[0, :, :]

        time_var = ds.variables["time"][:]
        valid_time = convert_timestamps_to_datetimes(time_var)[0]

        assert np.allclose(ds.variables["y"][:], y_ref)
        assert np.allclose(ds.variables["x"][:], x_ref)

        # Write into 4D array
        ei = ensemble_to_idx[ensemble]
        ti = time_to_idx[valid_time]
        blend_data[ei, ti, :, :] = rain_rate

    # Build DataArray
    blend_array = xr.DataArray(
        data=blend_data,
        coords={
            "ensemble": ensembles,
            "time": blend_times,
            "y": y_ref,
            "x": x_ref
        },
        dims=["ensemble", "time", "y", "x"],
        name="blend"
    )

    qpe_times = list(qpe_array.coords["time"].values)
    blend_times = list(blend_array.coords["time"].values)
    combined_times = qpe_times + blend_times

    # Convert to tz-aware datetime.datetime
    combined_times = pd.to_datetime(combined_times, utc=True)
    combined_times = [t.to_pydatetime() for t in combined_times]
    qpe_data = qpe_array.values

    # Tile across ensemble:
    n_ens = blend_array.sizes["ensemble"]
    qpe_broadcast = np.tile(qpe_data[None, :, :, :], (n_ens, 1, 1, 1))

    # Stack QPE and forecasts:
    combined_data = np.concatenate([qpe_broadcast, blend_array.values], axis=1)

    # Create combined xarray
    combined_array = xr.DataArray(
        data=combined_data,
        coords={
            "ensemble": blend_array.coords["ensemble"],
            "time": combined_times,
            "y": y_ref,
            "x": x_ref
        },
        dims=["ensemble", "time", "y", "x"],
        name="rainfall"
    )
    template = "$N_$P_$V{%Y%m%d_%H%M%S}.nc"
    tstamp = base_time.timestamp()
    product = "qpe_nwpblend"
    fname = make_nc_name(template, name, product, tstamp, None, None)
    fdir = args.directory
    file_name = fdir / fname
    logging.info(f"Writing data to {file_name}")

    write_rainfall_netcdf(
        filename=file_name,
        rainfall=combined_array.values,
        x=x_ref,
        y=y_ref,
        time=combined_times,
        ensemble=combined_array.coords["ensemble"].values
    )

    return


if __name__ == "__main__":
    main()
