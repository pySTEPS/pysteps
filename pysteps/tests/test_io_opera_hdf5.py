# -*- coding: utf-8 -*-

import os

import xarray as xr
import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("h5py")


root_path = pysteps.rcparams.data_sources["opera"]["root_path"]
filename = os.path.join(root_path, "20180824", "T_PAAH21_C_EUOC_20180824180000.hdf")
data_array = pysteps.io.import_opera_hdf5(filename)


def test_io_import_opera_hdf5_shape():
    """Test the importer OPERA HDF5."""
    assert isinstance(data_array, xr.DataArray)
    assert data_array.shape == (2200, 1900)


# test_metadata: list of (variable,expected, tolerance) tuples

expected_proj = (
    "+proj=laea +lat_0=55.0 +lon_0=10.0 "
    "+x_0=1950000.0 "
    "+y_0=-2100000.0 "
    "+units=m +ellps=WGS84"
)

# list of (variable,expected,tolerance) tuples
test_attrs = [
    ("projection", expected_proj, None),
    ("institution", "Odyssey datacentre", None),
    ("accutime", 15.0, 1e-10),
    ("unit", "mm/h", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 1e-10),
    ("threshold", 0.01, 1e-10),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs)
def test_io_import_mch_gif_dataset_attrs(variable, expected, tolerance):
    """Test the importer OPERA HDF5."""
    smart_assert(data_array.attrs[variable], expected, tolerance)
