# -*- coding: utf-8 -*-

import pytest
from numpy.testing import assert_array_almost_equal
from pysteps.tests.helpers import smart_assert, get_precipitation_fields

precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=True,
    source="mrms",
    log_transform=False,
    window_size=1,
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]


def test_io_import_mrms_grib():
    """Test the importer for NSSL data."""
    assert precip_dataarray.shape == (1, 3500, 7000)
    assert precip_dataarray.dtype == "single"


expected_proj = "+proj=longlat  +ellps=IAU76"

# list of (variable,expected,tolerance) tuples
test_attrs = [
    (precip_dataset.attrs["projection"], expected_proj, None),
    (
        precip_dataset.attrs["institution"],
        "NOAA National Severe Storms Laboratory",
        None,
    ),
    (precip_dataarray.attrs["units"], "mm/h", None),
    (precip_dataarray.attrs["zerovalue"], 0.0, 1e-6),
    (precip_dataarray.attrs["threshold"], 0.1, 1e-10),
    (precip_dataset.x.isel(x=0).values, -129.995, 1e-10),
    (precip_dataset.y.isel(y=0).values, 20.005001, 1e-10),
    (precip_dataset.x.isel(x=-1).values, -60.005002, 1e-10),
    (precip_dataset.y.isel(y=-1).values, 54.995, 1e-10),
    (precip_dataset.x.attrs["stepsize"], 0.01, 1e-4),
    (precip_dataset.y.attrs["stepsize"], 0.01, 1e-4),
    (precip_dataset.x.attrs["units"], "degrees", None),
    (precip_dataset.y.attrs["units"], "degrees", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs)
def test_io_import_mrms_grib_dataset_attrs(variable, expected, tolerance):
    """Test the importer MRMS_GRIB."""
    smart_assert(variable, expected, tolerance)


def test_io_import_mrms_grib_dataset_extent():
    """Test the importer MRMS_GRIB."""

    precip_dataset_smaller = get_precipitation_fields(
        num_prev_files=0,
        num_next_files=0,
        return_raw=True,
        source="mrms",
        log_transform=False,
        extent=(230, 300, 20, 55),
        window_size=1,
    )

    precip_var_smaller = precip_dataset_smaller.attrs["precip_var"]
    precip_dataarray_smaller = precip_dataset_smaller[precip_var_smaller]
    smart_assert(precip_dataarray_smaller.shape, (1, 3500, 7000), None)
    assert_array_almost_equal(precip_dataarray.values, precip_dataarray_smaller.values)

    precip_dataset_even_smaller = get_precipitation_fields(
        num_prev_files=0,
        num_next_files=0,
        return_raw=True,
        source="mrms",
        log_transform=False,
        extent=(250, 260, 30, 35),
        window_size=1,
    )

    precip_var_even_smaller = precip_dataset_even_smaller.attrs["precip_var"]
    precip_dataarray_even_smaller = precip_dataset_even_smaller[precip_var_even_smaller]
    smart_assert(precip_dataarray_even_smaller.shape, (1, 500, 1000), None)
    # XR: we had to change the selection of the original field since these is a flip happening in the way the data is read in.
    # XR: We had two ways to solve this: precip_dataarray[:,::-1, :][:, 2000:2500, 2000:3000][:,::-1, :] or switch the 2000:2500 to
    # I think this is logical as both the extend selected data and the reference data are flipped. That is why we need the double flip
    assert_array_almost_equal(
        precip_dataarray.values[:, 1000:1500, 2000:3000],
        precip_dataarray_even_smaller.values,
    )

    precip_dataset_double = get_precipitation_fields(
        num_prev_files=0,
        num_next_files=0,
        return_raw=True,
        source="mrms",
        log_transform=False,
        extent=(250, 260, 30, 35),
        window_size=1,
        dtype="double",
    )

    precip_var_double = precip_dataset_double.attrs["precip_var"]
    precip_dataarray_double = precip_dataset_double[precip_var_double]
    smart_assert(precip_dataarray_double.dtype, "double", None)

    precip_dataset_single = get_precipitation_fields(
        num_prev_files=0,
        num_next_files=0,
        return_raw=True,
        source="mrms",
        log_transform=False,
        extent=(250, 260, 30, 35),
        window_size=1,
        dtype="single",
    )

    precip_var_single = precip_dataset_single.attrs["precip_var"]
    precip_dataarray_single = precip_dataset_single[precip_var_single]
    smart_assert(precip_dataarray_single.dtype, "single", None)
