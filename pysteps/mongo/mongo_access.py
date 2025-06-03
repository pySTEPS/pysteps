# Contains: get_db, get_config, get_parameters_df, get_parameters, to_utc_naive
from typing import Dict
import pandas as pd
import datetime
import os
import logging
import pymongo.collection
from pymongo import MongoClient
from urllib.parse import quote_plus
from models.steps_params import StochasticRainParameters
from models.cascade_utils import get_cascade_wavelengths


def get_parameters(query: Dict, param_coll) -> Dict:
    """
    Get the parameters matching the query, indexed by valid_time.

    Args:
        query (dict): MongoDB query dictionary.
        param_coll (pymongo collection): Collection with the parameters.

    Returns:
        dict: Dictionary {valid_time: StochasticRainParameters}
    """
    result = {}
    for doc in param_coll.find(query).sort("metadata.valid_time", pymongo.ASCENDING):
        try:
            param = StochasticRainParameters.from_dict(doc)
            param.calc_corl()
            result[param.valid_time] = param
        except Exception as e:
            print(
                f"Warning: could not parse parameter for valid_time {doc.get('valid_time')}: {e}")
    return result


def get_parameters_df(query: Dict, param_coll: pymongo.collection.Collection) -> pd.DataFrame:
    """
    Retrieve STEPS parameters from the database and return a DataFrame
    indexed by (valid_time, base_time, ensemble), using 'NA' as sentinel for missing values.

    Args:
        query (dict): MongoDB query dictionary.
        param_coll (pymongo.collection.Collection): MongoDB collection.

    Returns:
        pd.DataFrame: Indexed by (valid_time, base_time, ensemble), with a 'param' column.
    """
    records = []

    for doc in param_coll.find(query).sort("metadata.valid_time", pymongo.ASCENDING):
        try:
            metadata = doc.get("metadata", {}) 
            if metadata is None:
                continue 

            if doc["cascade"]["lag1"] is None or  doc["cascade"]["lag2"] is None:
                continue
               
            valid_time = metadata.get("valid_time")
            if valid_time is not None and valid_time.tzinfo is None:
                valid_time = valid_time.replace(tzinfo=datetime.timezone.utc)

            base_time = metadata.get("base_time")
            if base_time is None:
                base_time = "NA"
            elif base_time.tzinfo is None:
                base_time = base_time.replace(tzinfo=datetime.timezone.utc)

            ensemble = metadata.get("ensemble") if metadata.get(
                "ensemble") is not None else "NA"
            param = StochasticRainParameters.from_dict(doc)

            param.calc_corl()
            records.append({
                "valid_time": valid_time,
                "base_time": base_time,
                "ensemble": ensemble,
                "param": param
            })
        except Exception as e:
            print(
                f"Warning: could not parse parameter for {metadata.get('valid_time')}: {e}")

    if not records:
        return pd.DataFrame(columns=["valid_time", "base_time", "ensemble", "param"])

    df = pd.DataFrame(records)
    return df


def get_config(db: pymongo.MongoClient, name: str) -> Dict:
    """_summary_
    Return the most recent configuration setting 
    Args:
        db (pymongo.MongoClient): Project database 

    Returns:
        Dict: Project configuration dictionary
    """

    config_coll = db["config"]
    record = config_coll.find_one({'config.name': name}, sort=[
                                  ('time', pymongo.DESCENDING)])
    if record is None:
        logging.error(f"Could not find configuration for domain {name}")
        return None

    config = record['config']
    return config


def get_db(mongo_port=None):
    MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
    # Use the function argument if provided, otherwise fall back to the environment variable, then default
    MONGO_PORT = mongo_port if mongo_port is not None else int(
        os.getenv("MONGO_PORT", 27017))

    if mongo_port is None:
        logging.info(f"Using MONGO_PORT from env: {MONGO_PORT}")
    else:
        logging.info(f"Using MONGO_PORT from argument: {mongo_port}")

    STEPS_USER = os.getenv("STEPS_USER", "radar")
    STEPS_PWD = os.getenv("STEPS_PWD", "c-bandBox")
    AUTH_DB = "STEPS"
    TARGET_DB = "STEPS"

    conect_string = (
        f"mongodb://{quote_plus(STEPS_USER)}:{quote_plus(STEPS_PWD)}"
        f"@{MONGO_HOST}:{MONGO_PORT}/STEPS?authSource={AUTH_DB}"
    )
    logging.info(f"Connecting to {conect_string}")
    client = MongoClient(conect_string)
    db = client[TARGET_DB]
    return db


def to_utc_naive(dt):
    if dt.tzinfo is not None:
        return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return dt


def get_central_wavelengths(db, name):
    config = get_config(db, name)
    n_levels = config["pysteps"].get("n_cascade_levels")
    domain = config["domain"]
    n_rows = domain.get("n_rows")
    n_cols = domain.get("n_cols")
    p_size = domain.get("p_size")
    p_size_km = p_size / 1000.0
    domain_size_km = max(n_rows, n_cols) * p_size_km

    # Get central wavelengths
    wavelengths_km = get_cascade_wavelengths(
        n_levels, domain_size_km, p_size_km)
    return wavelengths_km

def get_base_time(valid_time, product, name, db):
    # Get the base_time for the nwp run nearest to the valid_time in UTC zone
    # Assume spin-up of 3 hours
    start_base_time = valid_time - datetime.timedelta(hours=27)
    end_base_time = valid_time - datetime.timedelta(hours=3)
    base_time_query = {
        "metadata.product": product,
        "metadata.base_time": {"$gte": start_base_time, "$lte": end_base_time}
    }
    col_name = f"{name}.rain.files"
    nwp_base_times = db[col_name].distinct(
        "metadata.base_time", base_time_query)

    if nwp_base_times is None:
        logging.warning(
            f"Failed to find {product} data for {valid_time}")
        return None

    nwp_base_times.sort(reverse=True)
    base_time = nwp_base_times[0]

    if base_time.tzinfo is None:
        base_time = base_time.replace(tzinfo=datetime.timezone.utc)

    return base_time
