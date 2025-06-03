"""
make_parameters.py 
===================

Script to estimate the following STEPS parameters and place then in a MongoDB collection:
correl: lag 1 and 2 auto correlations for the cascade levels
b1, b2, l1: Slope of isotropic power spectrum above and below the scale l1 for rainfall field
mean, variance, wetted are ratio of the rainfall field
pdist: Sample cumulative probability distribution of rainfall field

"""

from models import read_nc, get_states, compute_field_parameters, get_db
from models import compute_field_stats, correlation_length

import datetime
import numpy as np
import io
import pymongo
import gridfs
import argparse
import logging
from pymongo import MongoClient

from pysteps.utils import transformation
from pysteps import extrapolation

from urllib.parse import quote_plus
import os
import sys

WAR_THRESHOLD = 0.05  # Select only fields with rain for analysis


def is_valid_iso8601(time_str: str) -> bool:
    """Check if the given string is a valid ISO 8601 datetime."""
    try:
        datetime.datetime.fromisoformat(time_str)
        return True
    except ValueError:
        return False


def lagr_auto_cor(data: np.ndarray, oflow: np.ndarray, config: dict):
    """
    Generate the Lagrangian auto correlations for STEPS cascades.

    Args:
        data (np.ndarray): [T, L, M, N] where:
            - T = ar_order + 1 (number of time steps)
            - L = number of cascade levels
            - M, N = spatial dimensions.
        oflow (np.ndarray): [2, M, N] Optical flow vectors.
        config (dict): Configuration dictionary containing:
            - "n_cascade_levels": Number of cascade levels (L).
            - "ar_order": Autoregressive order (1 or 2).
            - "extrapolation_method": Method for extrapolating fields.

    Returns:
        np.ndarray: Autocorrelation coefficients of shape (L, ar_order).
    """

    n_cascade_levels = config["pysteps"]["n_cascade_levels"]
    ar_order = config["pysteps"]["ar_order"]
    e_method = config["pysteps"]["extrapolation_method"]

    if data.shape[0] < (ar_order + 1):
        raise ValueError(
            f"Insufficient time steps. Expected at least {ar_order + 1}, got {data.shape[0]}.")

    extrapolation_method = extrapolation.get_method(e_method)
    autocorrelation_coefficients = np.full(
        (n_cascade_levels, ar_order), np.nan)

    for level in range(n_cascade_levels):
        lag_1 = extrapolation_method(data[-2, level], oflow, 1)[0]
        lag_1 = np.where(np.isfinite(lag_1), lag_1, 0)

        data_t = np.where(np.isfinite(data[-1, level]), data[-1, level], 0)
        if np.std(lag_1) > 1e-1 and np.std(data_t) > 1e-1:
            autocorrelation_coefficients[level, 0] = np.corrcoef(
                lag_1.flatten(), data_t.flatten())[0, 1]

        if ar_order == 2:
            lag_2 = extrapolation_method(data[-3, level], oflow, 1)[0]
            lag_2 = np.where(np.isfinite(lag_2), lag_2, 0)

            lag_1 = extrapolation_method(lag_2, oflow, 1)[0]
            lag_1 = np.where(np.isfinite(lag_1), lag_1, 0)

            if np.std(lag_1) > 1e-1 and np.std(data_t) > 1e-1:
                autocorrelation_coefficients[level, 1] = np.corrcoef(
                    lag_1.flatten(), data_t.flatten())[0, 1]

    return autocorrelation_coefficients


def process_files(file_names, db, config: dict):
    """
    Loop over a list of files and calculate the STEPS parameters.

    Args:
        file_names (list[str]): List of files to process
        data_base (pymongo.MongoClient): MongoDB database
        config (dict): Dictionary with pysteps configuration 

    Returns:
        list[dict]: List of steps parameter dictionaries     
    """
    ar_order = config["pysteps"]["ar_order"]
    timestep = config["pysteps"]["timestep"]
    time_step_mins = config["pysteps"]["timestep"] // 60
    db_zerovalue = config["pysteps"]["zerovalue"]
    db_threshold = config["pysteps"]["threshold"]
    scale_break = config['pysteps']["scale_break"]
    kmperpixel = config['pysteps']["kmperpixel"]
    name = config['name']

    delta_time_step = datetime.timedelta(seconds=timestep)

    rain_col_name = f"{name}.rain"
    rain_fs = gridfs.GridFS(db, collection=rain_col_name)

    params = []
    lag_2, lag_1, lag_0 = None, None, None
    oflow_0 = None

    for file_name in file_names:
        field = rain_fs.find_one({"filename": file_name})
        if field is None:
            continue

        # Set up the field metadata 
        valid_time = field.metadata["valid_time"]
        valid_time = valid_time.replace(tzinfo=datetime.timezone.utc)
        base_time = field.metadata["base_time"]
        if base_time is not None:
            base_time = base_time.replace(tzinfo=datetime.timezone.utc) 

        ensemble = field.metadata["ensemble"]
        product = field.metadata["product"]
        metadata = {
            "field_id": field._id,
            "product": product,
            "base_time": base_time,
            "ensemble": ensemble,
            "valid_time": valid_time,
            "kmperpixel": kmperpixel  # Need this when generating the stochastic fields
        }

        # Read in the rain field
        in_buffer = field.read()
        rain_geodata, _, rain_data = read_nc(in_buffer)
        # Needs to be consistent with db_threshold = -10
        rain_geodata["threshold"] = 0.1
        rain_geodata["zerovalue"] = 0
        rain_stats = compute_field_stats(rain_data, rain_geodata)

        if rain_geodata["transform"] is None:
            db_data, db_geodata = transformation.dB_transform(
                rain_data, rain_geodata, threshold=0.1, zerovalue=db_zerovalue
            )
            db_data[~np.isfinite(db_data)] = db_geodata["zerovalue"]
        else:
            db_data = rain_data.copy()

        db_geodata["threshold"] = db_threshold
        db_geodata["zerovalue"] = db_zerovalue
        db_stats = compute_field_stats(db_data, db_geodata)

        # Compute the power spectrum and prob dist
        steps_params = compute_field_parameters(
            db_data, db_geodata, scale_break)

        # Read in the cascades and calculate Lagr auto correlation
        cascade = {}

        # Only calculate cascade parameters if enough rain
        if rain_stats["nonzero_fraction"] < 0.05:
            steps_params.update({
                "metadata": metadata,
                "rain_stats": rain_stats,
                "dbr_stats": db_stats,
                "cascade": None
            })
            params.append(steps_params)
            continue

        # Fetch states for (t, t-1, t-2)
        query = {
            "metadata.product": product,
            "metadata.valid_time": {"$in": [valid_time, valid_time - delta_time_step, valid_time - 2 * delta_time_step]},
            "metadata.base_time": base_time,
            "metadata.ensemble": ensemble
        }

        states = get_states(db, name, query)

        # Set up the keys for the states
        bkey = "NA" if base_time is None else base_time
        ekey = "NA" if ensemble is None else ensemble
        lag0_inx = (valid_time, bkey, ekey)
        lag1_inx = (valid_time - delta_time_step, bkey, ekey)
        lag2_inx = (valid_time - 2*delta_time_step, bkey, ekey)

        state = states.get(lag0_inx)
        lag_0 = state["cascade"] if state is not None else None
        oflow_0 = state["optical_flow"] if state is not None else None
        state = states.get(lag1_inx)
        lag_1 = state["cascade"] if state is not None else None
        state = states.get(lag2_inx)
        lag_2 = state["cascade"] if state is not None else None

        # set up the cascade level means and stds for valid_time
        stds = lag_0.get("stds") if lag_0 else None
        cascade["stds"] = stds
        means = lag_0.get("means") if lag_0 else None
        cascade["means"] = means

        # Calculate the Lagr auto correl if enough data
        num_valid = sum(x is not None for x in [lag_2, lag_1, lag_0])
        if num_valid == ar_order + 1 and oflow_0 is not None:
            data = np.array([lag_1["cascade_levels"], lag_0["cascade_levels"]]) if ar_order == 1 else np.array(
                [lag_2["cascade_levels"], lag_1["cascade_levels"], lag_0["cascade_levels"]])
            auto_cor = lagr_auto_cor(data, oflow_0, config)

            # calculate the correlation lengths (minutes)
            lag1_list = auto_cor[:, 0].tolist()
            lag2_list = auto_cor[:, 1].tolist() if ar_order == 2 else [
                None] * len(lag1_list)
            corl_list = [
                correlation_length(l1, l2, time_step_mins)
                for l1, l2 in zip(lag1_list, lag2_list)
            ]

            cascade.update({
                "lag1": lag1_list,
                "lag2": lag2_list,
                "corl": corl_list,
                "corl_zero":corl_list[0]
            })
        else:
            cascade.update({
                "lag1": None,
                "lag2": None,
                "corl": None,
                "corl_zero":None
            })

        steps_params.update({
            "metadata": metadata,
            "rain_stats": rain_stats,
            "dbr_stats": db_stats,
            "cascade": cascade
        })
        params.append(steps_params)

    return params


def main():

    parser = argparse.ArgumentParser(
        description="Calculate STEPS parameters")

    parser.add_argument('-s', '--start', type=str, required=True,
                        help='Start time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-e', '--end', type=str, required=True,
                        help='End time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='Name of domain [AKL]')
    parser.add_argument('-p', '--product', type=str, required=True,
                        help='Name of input product [QPE, auckprec, qpesim]')

    args = parser.parse_args()

    # Include app name (module name) in log output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout
    )

    logger = logging.getLogger(__name__)
    logger.info("Calculating STEPS parameters")

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
            "Invalid product. Please provide either 'QPE', 'auckprec', or 'qpesim'.")
        return

    db = get_db()
    config_coll = db["config"]
    record = config_coll.find_one({'config.name': name}, sort=[
                                  ('time', pymongo.DESCENDING)])
    if record is None:
        logging.error(f"Could not find configuration for domain {name}")
        return

    config = record['config']
    meta_coll = db[f"{name}.rain.files"]
    params_coll = db[f"{name}.params"]

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
        steps_params = process_files(file_names, db, config)

        params_coll.delete_many({
            "metadata.product": product,
            "metadata.valid_time": {"$gte": start_time, "$lte": end_time}
        })
        if steps_params:
            params_coll.insert_many(steps_params)
    else:
        # Get the list of unique nwp run times in this period in ascending order
        start_base_time = start_time - datetime.timedelta(hours=12)
        base_time_query = {
            "metadata.product": product,
            "metadata.base_time": {"$gte": start_base_time, "$lte": end_time}
        }
        base_times = meta_coll.distinct("metadata.base_time", base_time_query)
        if base_times is None:
            logging.error(
                f"Failed to find {product} data for {start_time} - {end_time}")
            return

        base_times.sort()

        for base_time in base_times:

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
                    "metadata.valid_time": {"$gte": start_time, "$lte": end_time},
                    "metadata.base_time": base_time,
                    "metadata.ensemble": ensemble,
                    "metadata.wetted_area_ratio": {"$gte": WAR_THRESHOLD}
                }

                fields = {"_id": 0, "filename": 1,
                          "metadata.wetted_area_ratio": 1}
                results = meta_coll.find(filter=f_filter, projection=fields).sort(
                    "filename", pymongo.ASCENDING)
                file_names = [doc["filename"] for doc in results]

                if len(file_names) > 0:
                    steps_params = process_files(
                        file_names, db, config)
                    params_coll.delete_many({
                        "metadata.product": product,
                        "metadata.valid_time": {"$gte": start_time, "$lte": end_time},
                        "metadata.base_time": base_time,
                        "metadata.ensemble": ensemble,
                    })
                    if steps_params:
                        params_coll.insert_many(steps_params)


if __name__ == "__main__":
    main()
