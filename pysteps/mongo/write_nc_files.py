"""
Write rainfall grids to a netCDF file 
"""

from models import read_nc, make_nc_name_dt
from models import get_db, get_config
from pymongo import MongoClient
import logging
import argparse
import gridfs
import pymongo
import numpy as np
import datetime
import os
import netCDF4
from pyproj import CRS
import pandas as pd


def is_valid_iso8601(time_str: str) -> bool:
    """Check if the given string is a valid ISO 8601 datetime."""
    try:
        datetime.datetime.fromisoformat(time_str)
        return True
    except ValueError:
        return False


def get_base_times(db, base_time_query):
    meta_coll = db["AKL.rain.files"]
    base_times = list(meta_coll.distinct(
        "metadata.base_time", base_time_query))
    return base_times


def get_valid_times(db, valid_time_query):
    meta_coll = db["AKL.rain.files"]
    valid_times = list(meta_coll.distinct(
        "metadata.valid_time", valid_time_query))
    return valid_times


def get_rain_fields(db: pymongo.MongoClient, query: dict):
    meta_coll = db["AKL.rain.files"]

    # Fetch matching filenames and metadata in a single query
    fields_projection = {"_id": 1, "filename": 1, "metadata": 1}
    results = meta_coll.find(query, projection=fields_projection).sort(
        "filename", pymongo.ASCENDING)
    files = []
    for doc in results:
        record = {"_id": doc["_id"],
                  "valid_time": doc["metadata"]["valid_time"]}
        files.append(record)
    return files


def load_rain_field(db, file_id):
    """Retrieve a specific rain field NetCDF file from GridFS and return as numpy array"""
    fs = gridfs.GridFS(db, collection='AKL.rain')
    file_obj = fs.get(file_id)
    metadata = file_obj.metadata
    data_bytes = file_obj.read()
    geo_data, valid_time, rain_rate = read_nc(data_bytes)
    if isinstance(valid_time, np.ndarray):
        valid_time = valid_time.tolist()    
    return geo_data, metadata, valid_time, rain_rate


def write_netcdf(file_path: str, rain: np.ndarray, geo_data: dict, times: list[datetime.datetime], ensembles: list[int]) -> None:
    """
    Write a set of rainfall grids to a CF netCDF file
    Args:
        file_path (str): Full path to the output file
        rain (np.ndarray): Rainfall array. Shape is [ensemble, time, y, x] if ensembles is provided, 
        otherwise [time, y, x]
        geo_data (dict): Geospatial information 
        times (list[datetime.datetime]): list of valid times
        ensembles (list[int]): Optional list of valid ensemble numbers
    """
    # Convert the times to seconds since 1970-01-01T00:00:00Z
    time_stamps = []
    for time in times:
        if time.tzinfo is None:
            time = time.replace(tzinfo=datetime.timezone.utc)
        time_stamp = time.timestamp()
        time_stamps.append(time_stamp)

    x = geo_data['x']
    y = geo_data['y']
    projection = geo_data.get('projection', 'EPSG:4326')

    # Create NetCDF file on disk
    with netCDF4.Dataset(file_path, mode='w', format='NETCDF4') as ds:

        # Define dimensions
        ds.createDimension("y", len(y))
        ds.createDimension("x", len(x))
        ds.createDimension("time", len(times))

        # Coordinate variables
        y_var = ds.createVariable("y", "f4", ("y",))
        y_var[:] = y
        y_var.standard_name = "projection_y_coordinate"
        y_var.units = "m"

        x_var = ds.createVariable("x", "f4", ("x",))
        x_var[:] = x
        x_var.standard_name = "projection_x_coordinate"
        x_var.units = "m"

        t_var = ds.createVariable("time", "f8", ("time",))
        t_var[:] = time_stamps
        t_var.standard_name = "time"
        t_var.units = "seconds since 1970-01-01T00:00:00Z"
        t_var.calendar = "standard"

        # Set up the ensemble if we have one
        if ensembles is not None:
            ds.createDimension("ensemble", len(ensembles))
            e_var = ds.createVariable("ensemble", "i4", ("ensemble",))
            e_var[:] = ensembles
            e_var.standard_name = "ensemble"
            e_var.units = "1"

        # Rainfall
        if ensembles is None:
            rain_var = ds.createVariable(
                "rainfall", "i2", ("time", "y", "x"),
                zlib=True, complevel=5, fill_value=-1
            )
            rain_var[:, :, :] = np.nan_to_num(rain, nan=-1)

        else:
            rain_var = ds.createVariable(
                "rainfall", "i2", ("ensemble", "time", "y", "x"),
                zlib=True, complevel=5, fill_value=-1
            )
            rain_var[:, :, :, :] = np.nan_to_num(rain, nan=-1)

        rain_var.scale_factor = 0.1
        rain_var.add_offset = 0.0
        rain_var.units = "mm/h"
        rain_var.long_name = "Rainfall rate"
        rain_var.grid_mapping = "projection"
        rain_var.coordinates = "time y x" if ensembles is None else "ensemble time y x"

        # CRS
        crs = CRS.from_user_input(projection)
        cf_grid_mapping = crs.to_cf()
        spatial_ref = ds.createVariable("projection", "i4")
        for key, value in cf_grid_mapping.items():
            setattr(spatial_ref, key, value)

        # Global attributes
        ds.Conventions = "CF-1.10"
        ds.title = "Rainfall data"
        ds.institution = "Weather Radar New Zealand Ltd"
        ds.references = ""
        ds.comment = ""
    return


def main():
    parser = argparse.ArgumentParser(
        description="Write rainfall fields to a netCDF file")

    parser.add_argument('-s', '--start', type=str, required=True,
                        help='Start time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-e', '--end', type=str, required=True,
                        help='End time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='Name of domain [AKL]')
    parser.add_argument('-p', '--product', type=str, required=True,
                        help='Name of input product [QPE, auckprec]')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Validate start and end time and read them in
    if args.start and is_valid_iso8601(args.start):
        start_time = datetime.datetime.fromisoformat(str(args.start))
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=datetime.timezone.utc)
    else:
        logging.error(
            "Invalid start time format. Please provide a valid ISO 8601 time string.")
        return

    if args.end and is_valid_iso8601(args.end):
        end_time = datetime.datetime.fromisoformat(str(args.end))
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=datetime.timezone.utc)
    else:
        logging.error(
            "Invalid start time format. Please provide a valid ISO 8601 time string.")
        return

    name = str(args.name)
    product = str(args.product)
    valid_products = ["QPE", "auckprec", "qpesim"]
    if product not in valid_products:
        logging.error(
            f"Invalid product. Please provide either {valid_products}.")
        return

    db = get_db()
    meta_coll = db["AKL.rain.files"]

    if product == "QPE":
        file_id_query = {'metadata.product': product,
                         'metadata.valid_time': {"$gte": start_time, "$lte": end_time}}
        file_ids = get_rain_fields(db, file_id_query)

        out_grid = []
        valid_times = []
        geo_out = None
        expected_shape = None        
        for file_id in file_ids:
            geo_data, metadata, nc_times, rain_data = load_rain_field(
                db, file_id["_id"])
            
            if expected_shape is None:
                expected_shape = rain_data.shape
            elif rain_data.shape != expected_shape:
                logging.error(f"Inconsistent rain_data shape: expected {expected_shape}, got {rain_data.shape}")
                return
            
            out_grid.append(rain_data)
            valid_times.append(nc_times)
            if geo_out is None:
                geo_out = geo_data

        # QPE files are named using the start and end valid times
        name_template = "$N_$P_$V{%Y-%m-%dT%H:%M:%S}_$B{%Y-%m-%dT%H:%M:%S}.nc"
        file_name = make_nc_name_dt(
            name_template, name, product, start_time, end_time, None) 
        out_array = np.array(out_grid) 
        
        logging.info(f"Writing {file_name}")        
        write_netcdf(file_name, out_array, geo_out, valid_times, None)

    else:

        # Get the list of base times in the time period 
        base_time_query = {'metadata.product': product,
                        'metadata.base_time': {"$gte": start_time, "$lte": end_time}}
        base_times = list(meta_coll.distinct(
            "metadata.base_time", base_time_query))

        # Loop over the base times that have been found 
        for base_time in base_times:
            
            # Get the sorted list of ensmble members and valid times for this base time 
            ensemble_query = {'metadata.product': product,
                            'metadata.base_time': base_time}
            ensembles = list(meta_coll.distinct(
                "metadata.ensemble", ensemble_query))
            ensembles.sort()
            ne = len(ensembles)
            valid_times = list(meta_coll.distinct("metadata.valid_time", ensemble_query))
            valid_times.sort()
            nt = len(valid_times)
            # Loop over the ensembles and read in the grids 
            out_grid = []
            geo_out = None
            expected_shape = None        
            for ensemble in ensembles:
                
                # Get all the valid times for this ensemble 
                file_id_query = {'metadata.product': product,
                                'metadata.base_time': base_time, 'metadata.ensemble': ensemble}
                # Check that the expected number of fields have been found 
                file_ids = get_rain_fields(db, file_id_query)
                if len(valid_times) != len(file_ids):
                    logging.error(f"{base_time}:Expected {len(valid_times)} found {len(file_ids)} valid times") 

                for file_id in file_ids:
                    geo_data, metadata, nc_times, rain_data = load_rain_field(
                        db, file_id["_id"])
                    
                    if expected_shape is None:
                        expected_shape = rain_data.shape
                    elif rain_data.shape != expected_shape:
                        logging.error(f"Inconsistent rain_data shape: expected {expected_shape}, got {rain_data.shape}")
                        return
                    
                    out_grid.append(rain_data)
                    if geo_out is None:
                        geo_out = geo_data

            # Forecast files are named using their base time
            name_template = "$N_$P_$V{%Y-%m-%dT%H:%M:%S}.nc"
            ny,nx = expected_shape
            file_name = make_nc_name_dt(
                name_template, name, product, base_time, None, None) 
            out_array = np.array(out_grid).reshape(ne,nt,ny,nx) 

            logging.info(f"Writing {file_name}")
            write_netcdf(file_name, out_array, geo_out, valid_times, ensembles) 

    return


if __name__ == "__main__":
    main()
