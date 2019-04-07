"""
Testing helper functions
=======================

Collection of helper functions for the testing suite.
"""
from datetime import datetime

import numpy as np
import pytest

import pysteps as stp
from pysteps import io, rcparams


def get_precipitation_fields(num_prev_files=0):
    """Get a precipitation field from the archive to be used as reference."""

    # Selected case
    date = datetime.strptime("201505151630", "%Y%m%d%H%M")
    data_source = rcparams.data_sources["mch"]

    root_path = data_source["root_path"]
    path_fmt = data_source["path_fmt"]
    fn_pattern = data_source["fn_pattern"]
    fn_ext = data_source["fn_ext"]
    importer_name = data_source["importer"]
    importer_kwargs = data_source["importer_kwargs"]

    # Find the input files from the archive
    fns = io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext,
                                  timestep=5, num_prev_files=num_prev_files)

    # Read the radar composites
    importer = io.get_method(importer_name, "importer")
    reference_field, quality, metadata = io.read_timeseries(fns, importer,
                                                            **importer_kwargs)

    del quality  # Not used

    if num_prev_files == 0:
        reference_field = np.squeeze(reference_field)  # Remove time dimension

    # Convert to mm/h
    reference_field, metadata = stp.utils.to_rainrate(reference_field, metadata)

    # Mask invalid values
    reference_field = np.ma.masked_invalid(reference_field)

    # Log-transform the data [dBR]
    reference_field, metadata = stp.utils.dB_transform(reference_field,
                                                       metadata,
                                                       threshold=0.1,
                                                       zerovalue=-15.0)
    return reference_field


def smart_assert(actual_value, expected, tolerance=None):
    """
    Assert by equality for non-numeric values, or by approximation otherwise.

    If the precision keyword is None, assert by equality.
    When the precision is not None, assert that two numeric values
    (or two sets of numbers) are equal to each other within the tolerance.
    """

    if tolerance is None:
        assert actual_value == expected
    else:
        # Compare numbers up to a certain precision
        assert actual_value == pytest.approx(expected, 1e-6)
