"""
Testing helper functions
=======================

Collection of helper functions for the testing suite.
"""
import numpy as np
import pytest
from datetime import datetime

import pysteps as stp
from pysteps import io, rcparams, utils
from pysteps.utils import aggregate_fields_space


def get_precipitation_fields(num_prev_files=0,
                             num_next_files=0,
                             return_raw=False,
                             metadata=False,
                             upscale=None):
    """
    Get a precipitation field from the archive to be used as reference.

    Source: mch
    Reference time: 2015/05/15 1630 UTC

    Parameters
    ----------

    num_prev_files: int
        Number of previous times (files) to return with respect to the
        reference time.

    num_next_files: int
        Number of future times (files) to return with respect to the
        reference time.

    return_raw: bool
        Do not preprocess the precipitation fields. False by default.
        The pre-processing steps are: 1) Convert to mm/h,
        2) Mask invalid values, 3) Log-transform the data [dBR].

    metadata : bool
        If True, also return file metadata.

    upscale: float or None
        Upscale fields in space during the pre-processing steps.
        If it is None, the precipitation field is not
        modified.
        If it is a float, represents the length of the space window that is
        used to upscale the fields.


    Returns
    -------
    reference_field : array

    metadata : dict


    """
    pytest.importorskip('PIL')
    # Selected case
    date = datetime.strptime("201505151630", "%Y%m%d%H%M")
    data_source = rcparams.data_sources["mch"]

    root_path = data_source["root_path"]
    path_fmt = data_source["path_fmt"]
    fn_pattern = data_source["fn_pattern"]
    fn_ext = data_source["fn_ext"]
    importer_name = data_source["importer"]
    importer_kwargs = data_source["importer_kwargs"]
    timestep = data_source["timestep"]

    # Find the input files from the archive
    fns = io.archive.find_by_date(date,
                                  root_path,
                                  path_fmt,
                                  fn_pattern,
                                  fn_ext,
                                  timestep=timestep,
                                  num_prev_files=num_prev_files,
                                  num_next_files=num_next_files)

    # Read the radar composites
    importer = io.get_method(importer_name, "importer")
    reference_field, __, ref_metadata = io.read_timeseries(fns, importer,
                                                           **importer_kwargs)

    if not return_raw:

        if (num_prev_files == 0) and (num_next_files == 0):
            # Remove time dimension
            reference_field = np.squeeze(reference_field)

        # Convert to mm/h
        reference_field, ref_metadata = stp.utils.to_rainrate(reference_field,
                                                              ref_metadata)

        # Upscale data to 2 km
        reference_field, ref_metadata = aggregate_fields_space(reference_field,
                                                               ref_metadata,
                                                               upscale)

        # Mask invalid values
        reference_field = np.ma.masked_invalid(reference_field)

        # Log-transform the data [dBR]
        reference_field, ref_metadata = stp.utils.dB_transform(reference_field,
                                                               ref_metadata,
                                                               threshold=0.1,
                                                               zerovalue=-15.0)

        # Set missing values with the fill value
        reference_field.data[reference_field.mask] = -15.0

    if metadata:
        return reference_field, ref_metadata

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
        assert actual_value == pytest.approx(expected,
                                             rel=tolerance,
                                             abs=tolerance,
                                             nan_ok=True,
                                             )
