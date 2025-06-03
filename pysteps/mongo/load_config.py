import argparse
import json
import logging
import os
from pathlib import Path
import datetime
from pymongo import MongoClient, errors
from urllib.parse import quote_plus
from models import get_db

# Default pysteps configuration values
DEFAULT_PYSTEPS_CONFIG = {
    "precip_threshold": None,
    "extrapolation_method": "semilagrangian",
    "decomposition_method": "fft",
    "bandpass_filter_method": "gaussian",
    "noise_method": "nonparametric",
    "noise_stddev_adj": None,
    "ar_order": 1,
    "scale_break": None,
    "velocity_perturbation_method": None,
    "conditional": False,
    "probmatching_method": "cdf",
    "mask_method": "incremental",
    "seed": None,
    "num_workers": 1,
    "fft_method": "numpy",
    "domain": "spatial",
    "extrapolation_kwargs": {},
    "filter_kwargs": {},
    "noise_kwargs": {},
    "velocity_perturbation_kwargs": {},
    "mask_kwargs": {},
    "measure_time": False,
    "callback": None,
    "return_output": True
}

valid_product_list = ["qpesim", "auckprec", "nowcast", "nwpblend"]

# Default output configuration
DEFAULT_OUTPUT_CONFIG = { 
    "qpesim":{
        "gridfs_out": True,
        "nc_out": False,
        "out_product": "qpesim",
        "out_dir_name": None,
        "out_file_name": "$N_$P_$V{%Y-%m-%dT%H:%M:%S}_$B{%Y-%m-%dT%H:%M:%S}_$E.nc"
    },
    "auckprec":{ 
        "gridfs_out": True,
        "nc_out": False,
        "out_product": "auckprec",
        "tmp_dir": "$HOME/tmp",
        "out_dir_name": None,
        "out_file_name": "$N_$P_$V{%Y-%m-%dT%H:%M:%S}_$B{%Y-%m-%dT%H:%M:%S}_$E.nc"
    },
    "nowcast":{ 
        "gridfs_out": False,
        "nc_out": False,
        "out_product": "nowcast",
        "out_dir_name": None,
        "out_file_name": "$N_$P_$V{%Y-%m-%dT%H:%M:%S}_$B{%Y-%m-%dT%H:%M:%S}_$E.nc"
    },
    "nwpblend":{ 
        "gridfs_out": True,
        "nc_out": False,
        "out_product": "nwpblend",
        "out_dir_name": None,
        "out_file_name": "$N_$P_$V{%Y-%m-%dT%H:%M:%S}_$B{%Y-%m-%dT%H:%M:%S}_$E.nc"
    }
}

# Default domain configuration
DEFAULT_DOMAIN_CONFIG = {
    "n_rows": None,
    "n_cols": None,
    "p_size": None,
    "start_x": None,
    "start_y": None
}

# Default projection configuration for NZ
DEFAULT_PROJECTION_CONFIG = {
    "epsg": "EPSG:2193",
    "name": "transverse_mercator",
    "central_meridian": 173.0,
    "latitude_of_origin": 0.0,
    "scale_factor": 0.9996,
    "false_easting": 1600000.0,
    "false_northing": 10000000.0
}


def file_exists(file_path: Path) -> bool:
    """Check if the given file path exists."""
    return file_path.is_file()


def load_config(config_path: Path) -> dict:
    """Load the full configuration from a JSON file, applying defaults for missing fields."""
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON file: {config_path}")
        return {}

    name = config.get("name", None)
    if name is None:
        logging.error(
            f"Domain name not found")
        return {}

    # Extract and validate pysteps configuration
    pysteps_config = config.get("pysteps", {})
    if not isinstance(pysteps_config, dict):
        logging.error(
            f"Malformed pysteps configuration in {config_path}, expected a dictionary.")
        return {}

    # Apply default values
    for key, default_value in DEFAULT_PYSTEPS_CONFIG.items():
        if key not in pysteps_config:
            logging.warning(
                f"Missing key '{key}' in pysteps configuration, using default value.")
            pysteps_config[key] = default_value

    # Validate mandatory keys
    required_pysteps_keys = [
        "n_cascade_levels", "timestep", "kmperpixel"
    ]
    for key in required_pysteps_keys:
        if key not in pysteps_config:
            logging.error(
                f"Missing mandatory key '{key}' in pysteps configuration.")
            return {}

    # Extract and validate output configurations
    output_config = config.get("output", {})
    if not isinstance(output_config, dict):
        logging.error(
            f"Malformed output configuration in {config_path}, expected a dictionary."
        )
        return {}

    # Ensure "products" key exists and is a list
    valid_product_list = ["qpesim", "auckprec", "nowcast", "nwpblend"]

    products = output_config.get("products", [])
    if not isinstance(products, list):
        logging.error(
            f"Malformed 'products' key in output configuration, expected a list."
        )
        return {}

    # Dictionary to store parsed output configurations
    parsed_output_config = {}

    # Iterate over each product and extract its configuration
    for product in products:

        if product not in valid_product_list:
            logging.error(
                f"Unexpected product found, '{product}' not in {valid_product_list}."
            )
            continue

        product_config = output_config.get(product, {})

        if not isinstance(product_config, dict):
            logging.error(
                f"Malformed configuration for product '{product}', expected a dictionary."
            )
            continue

        # Merge with defaults
        complete_config = DEFAULT_OUTPUT_CONFIG[product].copy()
        complete_config.update(product_config)

        parsed_output_config[product] = complete_config

    # Extract and validate the domain location configuration
    domain_config = config.get("domain", {})
    if not isinstance(domain_config, dict):
        logging.error(
            f"Malformed domain configuration in {config_path}, expected a dictionary.")
        return {}

    for key, default_value in DEFAULT_DOMAIN_CONFIG.items():
        if key not in domain_config:
            logging.error(f"Missing key '{key}' in domain configuration.")
            return {}

    # Extract and validate the projection configuration - assumes CF fields for Transverse Mercator
    projection_config = config.get("projection", {})
    if not isinstance(projection_config, dict):
        logging.error(
            f"Malformed projection configuration in {config_path}, expected a dictionary.")
        return {}

    for key, default_value in DEFAULT_PROJECTION_CONFIG.items():
        if key not in projection_config:
            logging.warning(
                f"Missing key '{key}' in projection configuration, using default value")
            projection_config[key] = default_value

    # Get the dynamic scaling if present
    dynamic_scaling_config = config.get("dynamic_scaling", {})

    # Only check for required keys if the dictionary is not empty
    if dynamic_scaling_config:
        required_ds_keys = ["central_wave_lengths",
                            "space_time_exponent", "lag2_constants", "lag2_exponents"]
        for key in required_ds_keys:
            if key not in dynamic_scaling_config:
                logging.error(
                    f"Missing mandatory key '{key}' in dynamic_scaling configuration.")
                return {}

    return {
        "name": name,
        "pysteps": pysteps_config,
        "output": parsed_output_config,
        "domain": domain_config,
        "projection": projection_config,
        "dynamic_scaling": dynamic_scaling_config
    }


def insert_config_into_mongodb(config: dict):
    """Insert the configuration into the MongoDB config collection."""
    record = {
        "time": datetime.datetime.now(datetime.timezone.utc),
        "config": config
    }

    try:
        db = get_db()
        collection = db["config"]

        # Insert the record
        result = collection.insert_one(record)
        logging.info(
            f"Configuration inserted successfully. Document ID: {result.inserted_id}")

    except errors.ServerSelectionTimeoutError:
        logging.error(
            "Failed to connect to MongoDB. Check if MongoDB is running and the URI is correct.")
    except errors.PyMongoError as e:
        logging.error(f"MongoDB error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Insert pysteps configuration into MongoDB"
    )
    parser.add_argument('-c', '--config', type=Path,
                        help='Path to configuration file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Validate config file path
    if not args.config or not file_exists(args.config):
        logging.error(f"Configuration file does not exist: {args.config}")
        return

    # Load the full configuration
    config = load_config(args.config)

    if config:
        logging.info("Final loaded configuration:\n%s",
                     json.dumps(config, indent=2))
        insert_config_into_mongodb(config)


if __name__ == "__main__":
    main()
