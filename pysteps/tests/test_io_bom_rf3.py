# -*- coding: utf-8 -*-

import os

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert, get_precipitation_fields

precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=True,
    source="bom",
    log_transform=False,
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]


def test_io_import_bom_shape():
    """Test the shape of the read file."""
    assert precip_dataarray.shape == (1, 512, 512)


# Test import_bom_rf3 function
expected_proj = (
    "+proj=aea  +lon_0=144.752 +lat_0=-37.852 " "+lat_1=-18.000 +lat_2=-36.000"
)

test_metadata_bom = [
    (precip_dataset.attrs["projection"], expected_proj, None),
    (
        precip_dataset.attrs["institution"],
        "Commonwealth of Australia, Bureau of Meteorology",
        None,
    ),
    (precip_dataset.x.isel(x=0).values, -127750.0, 1e-5),
    (precip_dataset.y.isel(y=0).values, -127250.0, 1e-5),
    (precip_dataset.x.isel(x=-1).values, 127250.0, 1e-5),
    (precip_dataset.y.isel(y=-1).values, 127750.0, 1e-5),
    (precip_dataset.x.attrs["stepsize"], 500.0, 1e-4),
    (precip_dataset.y.attrs["stepsize"], 500.0, 1e-4),
    (precip_dataset.x.attrs["units"], "m", None),
    (precip_dataset.y.attrs["units"], "m", None),
    (precip_dataarray.attrs["accutime"], 6, 1e-4),
    (precip_dataarray.attrs["zerovalue"], 0.0, 1e-4),
    (precip_dataarray.attrs["units"], "mm", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_metadata_bom)
def test_io_import_bom_rf3_metadata(variable, expected, tolerance):
    """Test the importer Bom RF3."""
    smart_assert(variable, expected, tolerance)
