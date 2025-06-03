"""
Script to decompose and track the input rainfall fields and load them into the database

"""

from models import store_cascade_to_gridfs, replace_extension, read_nc
from models import get_db, get_config 
from pymongo import MongoClient
import logging
import argparse
import gridfs
import pymongo
import numpy as np
import datetime
import os
import sys

from pysteps import motion
from pysteps.utils import transformation
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.cascade.bandpass_filters import filter_gaussian 

from urllib.parse import quote_plus

WAR_THRESHOLD = 0.05  # Select only fields with rain for analysis

def is_valid_iso8601(time_str: str) -> bool:
    """Check if the given string is a valid ISO 8601 datetime."""
    try:
        datetime.datetime.fromisoformat(time_str)
        return True
    except ValueError:
        return False


def process_files(file_names: list[str], db: MongoClient, config: dict):
    timestep = config["pysteps"]["timestep"]
    db_zerovalue = config["pysteps"]["zerovalue"]
    n_levels = config['pysteps']['n_cascade_levels']
    n_rows = config['domain']['n_rows']
    n_cols = config['domain']['n_cols']
    name = config['name']

    oflow_method = motion.get_method("LK")  # Lucas-Kanade method
    bp_filter = filter_gaussian((n_rows, n_cols), n_levels)

    time_delta_tolerance = 120
    min_delta_time = datetime.timedelta(
        seconds=timestep - time_delta_tolerance)
    max_delta_time = datetime.timedelta(
        seconds=timestep + time_delta_tolerance)

    rain_col_name = f"{name}.rain"
    state_col_name = f"{name}.state"
    rain_fs = gridfs.GridFS(db, collection=rain_col_name)
    state_fs = gridfs.GridFS(db, collection=state_col_name)

    # Initialize buffers for batch processing
    prev_time = None
    cur_time = None
    prev_field = None
    cur_field = None
    file_names.sort()

    for file_name in file_names:
        grid_out = rain_fs.find_one({"filename": file_name}) 
        if grid_out is None:
            logging.warning(f"File {file_name} not found in GridFS, skipping.")
            continue

        # Extract metadata safely
        rain_fs_metadata = grid_out.metadata if hasattr(grid_out, "metadata") else {}

        if not rain_fs_metadata:
            logging.warning(f"No metadata found for {file_name}, skipping.")
            continue

        try:

            # Copy relevant metadata from rain_fs (MongoDB) to state_fs
            field_metadata = {
                "filename": replace_extension(grid_out.filename, ".npz"),
                "product": rain_fs_metadata.get("product", "unknown"),
                "domain": rain_fs_metadata.get("domain", "AKL"),
                "ensemble": rain_fs_metadata.get("ensemble", None),
                "base_time": rain_fs_metadata.get("base_time", None),
                "valid_time": rain_fs_metadata.get("valid_time", None),
                "mean": rain_fs_metadata.get("mean", 0),
                "std_dev": rain_fs_metadata.get("std_dev", 0),
                "wetted_area_ratio": rain_fs_metadata.get("wetted_area_ratio", 0)
            }

            # Check if cascade already exists for this file
            filename = replace_extension(grid_out.filename, ".npz")
            existing_file = state_fs.find_one({"filename": filename})
            if existing_file:
                state_fs.delete(existing_file._id)

            # Read the input NetCDF file
            in_buffer = grid_out.read()
            rain_geodata, valid_time, rain_data = read_nc(in_buffer)
            
            # Transform the field to dB if needed
            if rain_geodata.get("transform") is None:
                db_data, db_geodata = transformation.dB_transform(
                    rain_data, rain_geodata, threshold=0.1, zerovalue=db_zerovalue
                )
                db_data[~np.isfinite(db_data)] = db_geodata["zerovalue"]
            else:
                db_data = rain_data.copy()
                db_geodata = rain_geodata.copy()

            # Perform cascade decomposition
            cascade_dict = decomposition_fft(
                db_data, bp_filter, compute_stats=True, normalize=True
            )
            
            # Add the rain field transformation for the cascade 
            cascade_dict["transform"] = "dB"
            cascade_dict["zerovalue"] = db_zerovalue 
            cascade_dict["threshold"] = -10 # Assumes db_transform threshold = 0.1 

            # Compute optical flow
            if prev_time is None:
                prev_time = valid_time
                cur_time = valid_time
                prev_field = db_data
                cur_field = db_data
            else:
                prev_time = cur_time
                prev_field = cur_field
                cur_time = valid_time
                cur_field = db_data

            # Compute motion field if the time difference is in the acceptable range
            V1 = np.zeros((2, n_rows, n_cols))
            tdiff = cur_time - prev_time
            if min_delta_time < tdiff < max_delta_time:
                R = np.array([prev_field, cur_field])
                V1 = oflow_method(R)


            # Store cascade and motion field in GridFS with metadata
            store_cascade_to_gridfs(
                db, name, cascade_dict, V1, field_metadata["filename"], field_metadata)

        except Exception as e:
            logging.error(f"Error processing {grid_out.filename}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Decompose and track rainfall fields")


    parser.add_argument('-s', '--start', type=str, required=True,
                        help='Start time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-e', '--end', type=str, required=True,
                        help='End time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='Name of domain [AKL]')
    parser.add_argument('-p', '--product', type=str, required=True,
                        help='Name of input product [QPE, auckprec]')

    args = parser.parse_args()

    # Include app name (module name) in log output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting cascade generation process")

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
    if product not in ["QPE", "auckprec", "qpesim"]:
        logging.error(
            "Invalid product. Please provide either 'QPE' or 'auckprec'.")
        return
    
    db = get_db() 
    config = get_config(db,name) 
    meta_coll = db[f"{name}.rain.files"]

    # Single pass through the data for qpe product
    if product == "QPE":
        f_filter = {
            "metadata.product": product,
            "metadata.valid_time": {"$gte": start_time, "$lte": end_time},
            "metadata.wetted_area_ratio": {"$gte": WAR_THRESHOLD}
        }

        fields = {"_id": 0, "filename": 1, "metadata.wetted_area_ratio": 1}
        results = meta_coll.find(filter=f_filter, projection=fields).sort(
            "filename", pymongo.ASCENDING)
        if results is None:
            logging.error(
                f"Failed to find {product}data for {start_time} - {end_time}")
            return

        file_names = [doc["filename"] for doc in results]
        logging.info(
            f"Found {len(file_names)} {product} fields to process between {start_time} and {end_time}")
        process_files(file_names, db, config)
    else:
        # Get the list of unique nwp run times in this period in ascending order
        base_time_query = {
            "metadata.product": product,
            "metadata.base_time": {"$gte": start_time, "$lte": end_time}
        }
        base_times = meta_coll.distinct("metadata.base_time", base_time_query)
        if base_times is None:
            logging.error(
                f"Failed to find {product} data for {start_time} - {end_time}")
            return

        base_times.sort()
        logging.info(
            f"Found {len(base_times)} {product} NWP runs to process between {start_time} and {end_time}")

        for base_time in base_times:
            logging.info(f"Processing NWP run {base_time}") 

            # Get the list of unique ensembles found at base_time
            ensembles = meta_coll.distinct(
                "metadata.ensemble", {"metadata.product": product, "metadata.base_time": base_time})

            if not ensembles:
                logging.warning(
                    f"No ensembles found for base_time {base_time}")
                continue  # Skip this base_time if no ensembles exist

            logging.info(
                f"Found {len(ensembles)} ensembles for base_time {base_time}")
            ensembles.sort()

            for ensemble in ensembles:
                # Get all the forecasts for this base_time and ensemble and process

                f_filter = {
                    "metadata.product": product,
                    "metadata.base_time": base_time,
                    "metadata.valid_time": {"$gte": start_time, "$lte": end_time},
                    "metadata.ensemble": ensemble,
                    "metadata.wetted_area_ratio": {"$gte": WAR_THRESHOLD}
                }

                fields = {"_id": 0, "filename": 1,
                          "metadata.wetted_area_ratio": 1}
                results = meta_coll.find(filter=f_filter, projection=fields).sort(
                    "filename", pymongo.ASCENDING)
                file_names = [doc["filename"] for doc in results]

                if len(file_names) > 0:
                    process_files(file_names, db, config)


if __name__ == "__main__":
    main()
