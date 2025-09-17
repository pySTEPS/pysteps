# -*- coding: utf-8 -*-

import pytest

from pysteps.tests.helpers import get_precipitation_fields, smart_assert

# Test for RADOLAN RY product
precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=True,
    source="dwd",
    log_transform=False,
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]


def test_io_import_dwd_hdf5_ry_shape():
    """Test the importer DWD HDF5."""
    assert precip_dataarray.shape == (1, 1200, 1100)


# Test_metadata
# Expected projection definition
expected_proj = (
    "+proj=stere +lat_0=90 +lat_ts=60 "
    "+lon_0=10 +a=6378137 +b=6356752.3142451802 "
    "+no_defs +x_0=543196.83521776402 "
    "+y_0=3622588.8619310018 +units=m"
)

# List of (variable,expected,tolerance) tuples
test_ry_attrs = [
    (precip_dataset.attrs["projection"], expected_proj, None),
    (float(precip_dataset.lon.isel(x=0, y=-1).values), 3.57220017, 1e-8),
    (float(precip_dataset.lat.isel(x=0, y=-1).values), 45.70099971, 1e-8),
    (float(precip_dataset.lon.isel(x=-1, y=0).values), 18.72270377, 1e-8),
    (float(precip_dataset.lat.isel(x=-1, y=0).values), 55.84175857, 1e-8),
    (precip_dataset.x.isel(x=0).values, 0.0, 1e-3),
    (precip_dataset.y.isel(y=-1).values, -1199000.0, 1e-6),
    (precip_dataset.x.isel(x=-1).values, 1099000.0, 1e-6),
    (precip_dataset.y.isel(y=0).values, 0.0, 1e-3),
    (precip_dataset.x.attrs["stepsize"], 1000.0, 1e-10),
    (precip_dataset.y.attrs["stepsize"], -1000.0, 1e-10),
    (precip_dataset.x.attrs["units"], "m", None),
    (precip_dataset.y.attrs["units"], "m", None),
    (
        precip_dataset.attrs["institution"],
        "ORG:78,CTY:616,CMT:Deutscher Wetterdienst radolan@dwd.de",
        None,
    ),
    (precip_dataarray.attrs["accutime"], 5.0, 1e-10),
    (precip_dataset.time.attrs["stepsize"], 300, 1e-10),
    (precip_dataarray.attrs["units"], "mm/h", None),
    (precip_dataarray.attrs["zerovalue"], 0.0, 1e-6),
    (precip_dataarray.attrs["threshold"], 0.12, 1e-6),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_ry_attrs)
def test_io_import_dwd_hdf5_ry_metadata(variable, expected, tolerance):
    """Test the importer OPERA HDF5."""
    smart_assert(variable, expected, tolerance)
