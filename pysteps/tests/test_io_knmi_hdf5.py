# -*- coding: utf-8 -*-

import os

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("h5py")


def test_io_import_knmi_hdf5_shape():
    """Test the importer KNMI HDF5."""

    root_path = pysteps.rcparams.data_sources["knmi"]["root_path"]

    filename = os.path.join(root_path, "2010/08", "RAD_NL25_RAP_5min_201008260000.h5")

    precip, __, __ = pysteps.io.import_knmi_hdf5(filename)

    assert precip.shape == (765, 700)


# test_metadata: list of (variable,expected, tolerance) tuples

expected_proj1 = (
    "+proj=stere +lat_0=90 +lon_0=0.0 "
    "+lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 "
    "+y_0=0"
)

test_metadata = [
    ("projection", expected_proj1, None),
    ("x1", 0.0, 1e-10),
    ("y1", -4415.038179210632, 1e-10),
    ("x2", 699.9842646331593, 1e-10),
    ("y2", -3649.950360247753, 1e-10),
    ("xpixelsize", 1.0, 1e-10),
    ("xpixelsize", 1.0, 1e-10),
    ("cartesian_unit", "km", None),
    ("accutime", 5.0, 1e-10),
    ("yorigin", "upper", None),
    ("unit", "mm", None),
    ("institution", "KNMI - Royal Netherlands Meteorological Institute", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 1e-10),
    ("threshold", 0.01, 1e-10),
    ("zr_a", 200.0, None),
    ("zr_b", 1.6, None),
]


@pytest.mark.parametrize("variable,expected,tolerance", test_metadata)
def test_io_import_knmi_hdf5_metadata(variable, expected, tolerance):
    """Test the importer KNMI HDF5."""
    root_path = pysteps.rcparams.data_sources["knmi"]["root_path"]

    filename = os.path.join(root_path, "2010/08", "RAD_NL25_RAP_5min_201008260000.h5")

    __, __, metadata = pysteps.io.import_knmi_hdf5(filename)

    smart_assert(metadata[variable], expected, tolerance)
