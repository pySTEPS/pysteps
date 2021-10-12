# -*- coding: utf-8 -*-

import os

import xarray as xr
import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("h5py")


root_path = pysteps.rcparams.data_sources["knmi"]["root_path"]
filename = os.path.join(root_path, "2010/08", "RAD_NL25_RAP_5min_201008260000.h5")
data_array = pysteps.io.import_knmi_hdf5(filename)


def test_io_import_knmi_hdf5_shape():
    """Test the importer KNMI HDF5."""
    assert isinstance(data_array, xr.DataArray)
    assert data_array.shape == (765, 700)


# test_metadata: list of (variable,expected, tolerance) tuples

expected_proj = (
    "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378137 +b=6356752 +x_0=0 +y_0=0"
)

# list of (variable,expected,tolerance) tuples
test_attrs = [
    ("projection", expected_proj, None),
    ("institution", "KNMI - Royal Netherlands Meteorological Institute", None),
    ("accutime", 5.0, 1e-10),
    ("unit", "mm", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 1e-10),
    ("threshold", 0.01, 1e-10),
    ("zr_a", 200.0, 0.1),
    ("zr_b", 1.6, 0.1),
]


@pytest.mark.parametrize("variable,expected,tolerance", test_attrs)
def test_io_import_knmi_hdf5_metadata(variable, expected, tolerance):
    """Test the importer KNMI HDF5."""
    smart_assert(data_array.attrs[variable], expected, tolerance)
