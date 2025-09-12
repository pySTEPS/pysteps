# Contains: store_cascade_to_gridfs, load_cascade_from_gridfs, load_rain_field, get_rain_fields, get_states
from io import BytesIO
import gridfs
import numpy as np
import pymongo
import copy
import datetime
from typing import Dict, Any, Optional, Union, Tuple 

def store_cascade_to_gridfs(db, name, cascade_dict, oflow, file_name, field_metadata):
    """
    Stores a pysteps cascade decomposition dictionary into MongoDB's GridFS.

    Parameters:
        db (pymongo.database.Database): The MongoDB database object.
        cascade_dict (dict): The pysteps cascade decomposition dictionary.
        oflow (np.ndarray): The optical flow field.
        file_name (str): The (unique) name of the file to be stored.
        field_metadata (dict): Additional metadata related to the field.

    Returns:
        bson.ObjectId: The GridFS file ID.
    """
    assert cascade_dict["domain"] == "spatial", "Only 'spatial' domain is supported."
    state_col_name = f"{name}.state"
    fs = gridfs.GridFS(db, collection=state_col_name)

    # Delete existing file with same filename
    for old_file in fs.find({"filename": file_name}):
        fs.delete(old_file._id)

    # Convert cascade_levels and oflow to a compressed format
    buffer = BytesIO()
    np.savez_compressed(
        buffer, cascade_levels=cascade_dict["cascade_levels"], oflow=oflow)
    buffer.seek(0)

    # Prepare metadata
    metadata = {
        "filename": file_name,
        "domain": cascade_dict["domain"],
        "normalized": cascade_dict["normalized"],
        "transform": cascade_dict.get("transform"),
        "threshold": cascade_dict.get("threshold"),
        "zerovalue": cascade_dict.get("zerovalue")
    }
    metadata.update(field_metadata)  # Merge additional metadata

    # Add optional statistics if available
    if "means" in cascade_dict:
        metadata["means"] = cascade_dict["means"]
    if "stds" in cascade_dict:
        metadata["stds"] = cascade_dict["stds"]

    # Store binary data and metadata atomically in GridFS
    file_id = fs.put(buffer.getvalue(), filename=file_name, metadata=metadata)

    return file_id


def load_cascade_from_gridfs(db, name, file_name):
    """
    Loads a pysteps cascade decomposition dictionary and optical flow from MongoDB's GridFS.

    Parameters:
        db (pymongo.database.Database): The MongoDB database object.
        file_name (str): The name of the file to retrieve.

    Returns:
        tuple: (cascade_dict, oflow, metadata)
    """
    state_col_name = f"{name}.state"
    fs = gridfs.GridFS(db, collection=state_col_name)

    # Retrieve the file from GridFS
    grid_out = fs.find_one({"filename": file_name})
    if grid_out is None:
        raise ValueError(f"No file found with filename: {file_name}")

    # Retrieve metadata
    metadata = grid_out.metadata

    # Read and decompress stored arrays
    buffer = BytesIO(grid_out.read())
    npzfile = np.load(buffer)

    # Reconstruct cascade dictionary including the initial field transformation
    cascade_dict = {
        "cascade_levels": npzfile["cascade_levels"],
        "domain": metadata["domain"],
        "normalized": metadata["normalized"],
        "transform": metadata.get("transform"),
        "threshold": metadata.get("threshold"),
        "zerovalue": metadata.get("zerovalue")
    }

    # Restore optional statistics if they exist
    if "means" in metadata:
        cascade_dict["means"] = metadata["means"]
    if "stds" in metadata:
        cascade_dict["stds"] = metadata["stds"]

    oflow = npzfile["oflow"]  # Optical flow field

    return cascade_dict, oflow, metadata


def load_rain_field(db, name, filename, nc_buf, metadata):

    # Check if the file exists, if yes then delete it
    rain_col_name = f"{name}.rain"

    fs = gridfs.GridFS(db, collection=rain_col_name)

    existing_file = fs.find_one(
        {"filename": filename})
    if existing_file:
        fs.delete(existing_file._id)

    # Upload to GridFS
    fs.put(nc_buf.tobytes(),
           filename=filename, metadata=metadata)


def get_rain_fields(db: pymongo.MongoClient, name: str, query: dict):
    rain_col_name = f"{name}.rain"
    meta_col_name = f"{name}.rain.files"
    fs = gridfs.GridFS(db, collection=rain_col_name)
    meta_coll = db[meta_col_name]

    # Fetch matching filenames and metadata in a single query
    fields_projection = {"_id": 0, "filename": 1, "metadata": 1}
    results = meta_coll.find(query, projection=fields_projection).sort(
        "filename", pymongo.ASCENDING)

    fields = []

    # Process each matching file
    for doc in results:
        filename = doc["filename"]

        # Fetch metadata from GridFS
        grid_out = fs.find_one({"filename": filename})
        if grid_out is None:
            logging.warning(f"File {filename} not found in GridFS, skipping.")
            continue

        rain_fs_metadata = grid_out.metadata if hasattr(
            grid_out, "metadata") else {}

        # Copy relevant metadata
        field_metadata = {
            "filename": filename,
            "product": rain_fs_metadata.get("product", "unknown"),
            "domain": rain_fs_metadata.get("domain", "AKL"),
            "ensemble": rain_fs_metadata.get("ensemble", None),
            "base_time": rain_fs_metadata.get("base_time", None),
            "valid_time": rain_fs_metadata.get("valid_time", None),
            "mean": rain_fs_metadata.get("mean", 0),
            "std_dev": rain_fs_metadata.get("std_dev", 0),
            "wetted_area_ratio": rain_fs_metadata.get("wetted_area_ratio", 0)
        }

        # Stream and decompress data
        buffer = BytesIO(grid_out.read())
        rain_geodata, _, rain_data = read_nc(buffer)  # Fixed variable name

        # Add the georeferencing metadata dictionary
        field_metadata["geo_data"] = rain_geodata

        # Store the final record
        record = {"rain": rain_data.copy(
        ), "metadata": copy.deepcopy(field_metadata)}
        fields.append(record)  # Append the record to the list

    return fields


def get_states(db: pymongo.MongoClient, name: str, query: dict,
               get_cascade: Optional[bool] = True,
               get_optical_flow: Optional[bool] = True
               ) -> Dict[Tuple[Any, Any, Any], Dict[str, Optional[Union[dict, np.ndarray]]]]:
    """
    Retrieve state fields (cascade and/or optical flow) from a GridFS collection,
    indexed by (valid_time, base_time, ensemble).

    Args:
        db (pymongo.MongoClient): Database with the state collections.
        name (str): Name prefix of the state collections.
        query (dict): Mongo query for filtering state files.
        get_cascade (bool, optional): Whether to retrieve cascade state. Defaults to True.
        get_optical_flow (bool, optional): Whether to retrieve optical flow. Defaults to True.

    Returns:
        dict: {(valid_time, base_time, ensemble): {"cascade": dict or None,
                                                   "optical_flow": np.ndarray or None,
                                                   "metadata": dict}}
    """
    state_col_name = f"{name}.state"
    meta_col_name = f"{name}.state.files"
    fs = gridfs.GridFS(db, collection=state_col_name)
    meta_coll = db[meta_col_name]

    fields = {"_id": 0, "filename": 1, "metadata": 1}
    results = meta_coll.find(query, projection=fields).sort("filename", pymongo.ASCENDING)

    states = {}

    for doc in results:
        state_file = doc["filename"]
        metadata_dict = doc.get("metadata", {})

        valid_time = metadata_dict.get("valid_time")
        if valid_time is None:
            logging.warning(f"No valid_time in metadata for file {state_file}, skipping.")
            continue
        if valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=datetime.timezone.utc)

        base_time = metadata_dict.get("base_time", "NA")
        if base_time is not None and base_time.tzinfo is None:
            base_time = base_time.replace(tzinfo=datetime.timezone.utc)

        ensemble = metadata_dict.get("ensemble", "NA")

        # Set missing base_time or ensemble to "NA"
        if base_time is None:
            base_time = "NA"
        if ensemble is None:
            ensemble = "NA"


        grid_out = fs.find_one({"filename": state_file})
        if grid_out is None:
            logging.warning(f"File {state_file} not found in GridFS, skipping.")
            continue

        buffer = BytesIO(grid_out.read())
        npzfile = np.load(buffer)

        cascade_dict = None
        if get_cascade:
            cascade_dict = {
                "cascade_levels": npzfile["cascade_levels"],
                "domain": metadata_dict.get("domain"),
                "normalized": metadata_dict.get("normalized"),
                "transform": metadata_dict.get("transform"),
                "threshold": metadata_dict.get("threshold"),
                "zerovalue": metadata_dict.get("zerovalue"),
                "means": metadata_dict.get("means"),
                "stds": metadata_dict.get("stds"),
            }

        oflow = None
        if get_optical_flow:
            oflow = npzfile["oflow"]

        key = (valid_time, base_time, ensemble)
        states[key] = {
            "cascade": copy.deepcopy(cascade_dict) if cascade_dict is not None else None,
            "optical_flow": oflow.copy() if oflow is not None else None,
            "metadata": copy.deepcopy(metadata_dict)
        }

    return states

