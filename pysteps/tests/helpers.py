"""
Testing helper functions
=======================

Collection of helper functions for the testing suite.
"""
from datetime import datetime

import numpy as np
import pytest

from pysteps import io, utils, rcparams
from pysteps.decorators import _xarray2legacy

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
    source="mch",
    convert_to=None,
    transform_to=None,
    filled=False,
    coarsen=None,
    clip=None,
    importer_kwargs=None,
    **kwargs,
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
    source: {"bom", "fmi" , "knmi", "mch", "opera", "saf", "mrms"}
        Name of the data source to be used.
    convert_to: {"mm/h", "mm", "Z", "dBZ"}, optional
        Convert to given units.
    transform_to: {"boxcox", "log", "db", "nq", "sqrt"}, optional
        Tranform data.
    filled: bool
        Whether to fill all the missing values.
    clip: scalars (left, right, bottom, top), optional
        The extent of the bounding box in data coordinates to be used to clip
        the data.
    coarsen: float, optional
        Upscale fields in space during the pre-processing steps.
        If it is None, the precipitation field is not modified.
        If it is a float, represents the the window size that is used to
        upscale the fields.
    importer_kwargs : dict, optional
        Additional keyword arguments passed to the importer.

    Returns
    -------
    data_array : xr.DataArray
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

    if importer_kwargs is None:
        importer_kwargs = {}

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
    reference_field = io.read_timeseries(fns, importer, **_importer_kwargs)

    # Squeeze single fields
    if (num_prev_files == 0) and (num_next_files == 0):
        reference_field = reference_field.squeeze("t")

    # Clip domain
    if clip:
        reference_field = reference_field.sel(
            x=slice(clip[0], clip[1]),
            y=slice(clip[2], clip[3]),
        )

    # Fill nans
    if filled:
        units = reference_field.attrs.get("unit")
        reference_field = reference_field.pysteps.to_rainrate()
        reference_field = reference_field.fillna(0)
        converter = utils.get_method(units)
        reference_field = converter(reference_field)

    # Coarsen data
    if coarsen:
        units = reference_field.attrs.get("unit")
        reference_field = reference_field.pysteps.to_rainrate()
        reference_field = reference_field.coarsen(
            x=coarsen, y=coarsen, boundary="trim"
        ).mean()
        converter = utils.get_method(units)
        reference_field = converter(reference_field)

    # Convert
    converter = utils.get_method(convert_to)
    reference_field = converter(reference_field)

    # Transform
    transformer = utils.get_method(transform_to)
    reference_field = transformer(reference_field)

    if "legacy" in kwargs or "metadata" in kwargs:
        reference_field, _, ref_metadata = _xarray2legacy(reference_field)
        if kwargs.get("metadata", False):
            return reference_field, ref_metadata
        else:
            return reference_field

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
