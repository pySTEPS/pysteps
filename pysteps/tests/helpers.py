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
from pysteps.utils import aggregate_fields_space

_reference_dates = dict()
_reference_dates["bom"] = datetime(2018, 6, 16, 10, 0)
_reference_dates["fmi"] = datetime(2016, 9, 28, 16, 0)
_reference_dates["knmi"] = datetime(2010, 8, 26, 0, 0)
_reference_dates["mch"] = datetime(2015, 5, 15, 16, 30)
_reference_dates["opera"] = datetime(2018, 8, 24, 18, 0)
_reference_dates["saf"] = datetime(2018, 6, 1, 7, 0)
_reference_dates["mrms"] = datetime(2019, 6, 10, 0, 0)


def get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=False,
    metadata=False,
    upscale=None,
    source="mch",
    log_transform=True,
    clip=None,
    **importer_kwargs,
):
    """
    Get a precipitation field from the archive to be used as reference.

    Source: bom
    Reference time: 2018/06/16 10000 UTC

    Source: fmi
    Reference time: 2016/09/28 1600 UTC

    Source: knmi
    Reference time: 2010/08/26 0000 UTC

    Source: mch
    Reference time: 2015/05/15 1630 UTC

    Source: opera
    Reference time: 2018/08/24 1800 UTC

    Source: saf
    Reference time: 2018/06/01 0700 UTC

    Source: mrms
    Reference time: 2019/06/10 0000 UTC

    Parameters
    ----------

    num_prev_files: int, optional
        Number of previous times (files) to return with respect to the
        reference time.

    num_next_files: int, optional
        Number of future times (files) to return with respect to the
        reference time.

    return_raw: bool, optional
        Do not preprocess the precipitation fields. False by default.
        The pre-processing steps are: 1) Convert to mm/h,
        2) Mask invalid values, 3) Log-transform the data [dBR].

    metadata: bool, optional
        If True, also return file metadata.

    clip: scalars (left, right, bottom, top), optional
        The extent of the bounding box in data coordinates to be used to clip
        the data.

    upscale: float or None, optional
        Upscale fields in space during the pre-processing steps.
        If it is None, the precipitation field is not modified.
        If it is a float, represents the length of the space window that is
        used to upscale the fields.

    source: {"bom", "fmi" , "knmi", "mch", "opera", "saf", "mrms"}, optional
        Name of the data source to be used.

    log_transform: bool
        Whether to transform the output to dB.

    Other Parameters
    ----------------

    importer_kwargs : dict
        Additional keyword arguments passed to the importer.

    Returns
    -------
    reference_field : array

    metadata : dict
    """

    if source == "bom":
        pytest.importorskip("netCDF4")

    if source == "fmi":
        pytest.importorskip("pyproj")

    if source == "knmi":
        pytest.importorskip("h5py")

    if source == "mch":
        pytest.importorskip("PIL")

    if source == "opera":
        pytest.importorskip("h5py")

    if source == "saf":
        pytest.importorskip("netCDF4")

    if source == "mrms":
        pytest.importorskip("pygrib")

    try:
        date = _reference_dates[source]
    except KeyError:
        raise ValueError(
            f"Unknown source name '{source}'\n"
            "The available data sources are: "
            f"{str(list(_reference_dates.keys()))}"
        )

    data_source = rcparams.data_sources[source]
    root_path = data_source["root_path"]
    path_fmt = data_source["path_fmt"]
    fn_pattern = data_source["fn_pattern"]
    fn_ext = data_source["fn_ext"]
    importer_name = data_source["importer"]
    _importer_kwargs = data_source["importer_kwargs"].copy()
    _importer_kwargs.update(**importer_kwargs)
    timestep = data_source["timestep"]

    # Find the input files from the archive
    fns = io.archive.find_by_date(
        date,
        root_path,
        path_fmt,
        fn_pattern,
        fn_ext,
        timestep=timestep,
        num_prev_files=num_prev_files,
        num_next_files=num_next_files,
    )

    # Read the radar composites
    importer = io.get_method(importer_name, "importer")

    reference_field, __, ref_metadata = io.read_timeseries(
        fns, importer, **_importer_kwargs
    )

    if not return_raw:

        if (num_prev_files == 0) and (num_next_files == 0):
            # Remove time dimension
            reference_field = np.squeeze(reference_field)

        # Convert to mm/h
        reference_field, ref_metadata = stp.utils.to_rainrate(
            reference_field, ref_metadata
        )

        # Clip domain
        reference_field, ref_metadata = stp.utils.clip_domain(
            reference_field, ref_metadata, clip
        )

        # Upscale data
        reference_field, ref_metadata = aggregate_fields_space(
            reference_field, ref_metadata, upscale
        )

        # Mask invalid values
        reference_field = np.ma.masked_invalid(reference_field)

        if log_transform:
            # Log-transform the data [dBR]
            reference_field, ref_metadata = stp.utils.dB_transform(
                reference_field, ref_metadata, threshold=0.1, zerovalue=-15.0
            )

            # Set missing values with the fill value
            np.ma.set_fill_value(reference_field, -15.0)
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
        assert actual_value == pytest.approx(
            expected, rel=tolerance, abs=tolerance, nan_ok=True
        )


def get_invalid_mask(input_array, fillna=np.nan):
    """
    Return a bool array indicating the invalid values in ``input_array``.

    If the input array is a MaskedArray, its mask will be returned.
    Otherwise, it returns an array with the ``input_array == fillna``
    element-wise comparison.
    """
    if isinstance(input_array, np.ma.MaskedArray):
        invalid_mask = np.ma.getmaskarray(input_array)
    else:
        if fillna is np.nan:
            invalid_mask = ~np.isfinite(input_array)
        else:
            invalid_mask = input_array == fillna

    return invalid_mask
