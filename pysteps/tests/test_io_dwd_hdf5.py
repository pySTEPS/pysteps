# -*- coding: utf-8 -*-

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert, get_precipitation_fields

pytest.importorskip("h5py")

# Test for RADOLAN RY product

precip_ry, metadata_ry = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=False,
    metadata=True,
    source="dwd",
    log_transform=False,
    importer_kwargs=dict(qty="RATE"),
)


def test_io_import_dwd_hdf5_ry_shape():
    """Test the importer DWD HDF5."""
    assert precip_ry.shape == (1200, 1100)


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
    ("projection", expected_proj, None),
    ("ll_lon", 3.566994635, 1e-10),
    ("ll_lat", 45.69642538, 1e-10),
    ("ur_lon", 18.73161645, 1e-10),
    ("ur_lat", 55.84543856, 1e-10),
    ("x1", -500.0, 1e-6),
    ("y1", -1199500.0, 1e-6),
    ("x2", 1099500.0, 1e-6),
    ("y2", 500.0, 1e-6),
    ("xpixelsize", 1000.0, 1e-10),
    ("xpixelsize", 1000.0, 1e-10),
    ("cartesian_unit", "m", None),
    ("yorigin", "upper", None),
    ("institution", "ORG:78,CTY:616,CMT:Deutscher Wetterdienst radolan@dwd.de", None),
    ("accutime", 5.0, 1e-10),
    ("unit", "mm/h", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 1e-6),
    ("threshold", 0.12, 1e-6),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_ry_attrs)
def test_io_import_dwd_hdf5_ry_metadata(variable, expected, tolerance):
    """Test the importer OPERA HDF5."""
    smart_assert(metadata_ry[variable], expected, tolerance)
