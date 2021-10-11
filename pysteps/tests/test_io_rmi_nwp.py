# -*- coding: utf-8 -*-

# Test the importer for NWP data from the Royal Meteorological Institute of
# Belgium (ALARO/AROME data in netCDF output).

import os
import numpy as np

import xarray as xr
import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("netCDF4")

root_path = pysteps.rcparams.data_sources["rmi_nwp"]["root_path"]
rel_path = os.path.join("2021", "07", "04")
filename = os.path.join(root_path, rel_path, "ao13_2021070412_native_5min.nc")
data_array_xr = pysteps.io.import_rmi_nwp_xr(filename)

expected_proj = "+proj=lcc +lon_0=4.55 +lat_1=50.8 +lat_2=50.8 +a=6371229 +es=0 +lat_0=50.8 +x_0=365950 +y_0=-365950"


def test_io_import_rmi_nwp_xarray():
    """Test the importer RMI NWP."""
    assert isinstance(data_array_xr, xr.DataArray)


def test_io_import_rmi_nwp_xarray_shape():
    """Test the importer RMI NWP shape."""
    assert isinstance(data_array_xr, xr.DataArray)
    assert data_array_xr.shape == (12, 564, 564)


test_attrs_xr = [
    ("projection", expected_proj, None),
    ("institution", "Royal Meteorological Institute of Belgium", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 0.1),
    ("unit", "mm", None),
    ("accutime", np.timedelta64(5, "m"), None),
    ("zr_a", None, None),
    ("zr_b", None, None),
    ("xpixelsize", 1300.0, 0.1),
    ("ypixelsize", 1300.0, 0.1),
    ("units", "mm", None),
    ("yorigin", "upper", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr)
def test_io_import_rmi_nwp_xarray_attrs(variable, expected, tolerance):
    """Test the importer RMI NWP."""
    smart_assert(data_array_xr.attrs[variable], expected, tolerance)


test_attrs_xr_coord_x = [
    ("units", "m", None),
    ("x1", 0, 0.1),
    ("x2", 731900.0, 0.1),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr_coord_x)
def test_io_import_rmi_nwp_xarray_attrs_coordx(variable, expected, tolerance):
    """Test the importer RMI NWP."""
    smart_assert(data_array_xr.x.attrs[variable], expected, tolerance)


test_attrs_xr_coord_y = [
    ("units", "m", None),
    ("y1", -731900.0, 0.1),
    ("y2", 0.0, 0.1),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr_coord_y)
def test_io_import_rmi_nwp_xarray_attrs_coordy(variable, expected, tolerance):
    """Test the importer RMI NWP."""
    smart_assert(data_array_xr.y.attrs[variable], expected, tolerance)
