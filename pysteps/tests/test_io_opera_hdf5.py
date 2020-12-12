# -*- coding: utf-8 -*-

import os

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("h5py")


def test_io_import_opera_hdf5_shape():
    """Test the importer OPERA HDF5."""

    root_path = pysteps.rcparams.data_sources["opera"]["root_path"]
    filename = os.path.join(root_path, "20180824", "T_PAAH21_C_EUOC_20180824180000.hdf")
    R, _, _ = pysteps.io.import_opera_hdf5(filename)
    assert R.shape == (2200, 1900)


# test_metadata: list of (variable,expected, tolerance) tuples

expected_proj1 = (
    "+proj=laea +lat_0=55.0 +lon_0=10.0 "
    "+x_0=1950000.0 "
    "+y_0=-2100000.0 "
    "+units=m +ellps=WGS84"
)

test_metadata = [
    ("projection", expected_proj1, None),
    ("ll_lon", -10.434576838640398, 1e-10),
    ("ll_lat", 31.746215319325056, 1e-10),
    ("ur_lon", 57.81196475014995, 1e-10),
    ("ur_lat", 67.62103710275053, 1e-10),
    ("x1", -0.0004161088727414608, 1e-6),
    ("y1", -4400000.001057557, 1e-10),
    ("x2", 3800000.0004256153, 1e-10),
    ("y2", -0.0004262728616595268, 1e-6),
    ("xpixelsize", 2000.0, 1e-10),
    ("xpixelsize", 2000.0, 1e-10),
    ("cartesian_unit", "m", None),
    ("accutime", 15.0, 1e-10),
    ("yorigin", "upper", None),
    ("unit", "mm/h", None),
    ("institution", "Odyssey datacentre", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 1e-10),
    ("threshold", 0.01, 1e-10),
]


@pytest.mark.parametrize("variable,expected,tolerance", test_metadata)
def test_io_import_opera_hdf5_metadata(variable, expected, tolerance):
    """Test the importer OPERA HDF5."""
    root_path = pysteps.rcparams.data_sources["opera"]["root_path"]
    filename = os.path.join(root_path, "20180824", "T_PAAH21_C_EUOC_20180824180000.hdf")
    _, _, metadata = pysteps.io.import_opera_hdf5(filename)
    print(metadata)
    # assert metadata[variable] == expected
    smart_assert(metadata[variable], expected, tolerance)
