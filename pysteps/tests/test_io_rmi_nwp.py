# -*- coding: utf-8 -*-

import os
import numpy as np

import xarray as xr
import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("netCDF4")

root_path = pysteps.rcparams.data_sources["rmi_nwp"]["root_path"]
rel_path = os.path.join("2021", "07", "04")
filename = os.path.join(root_path, rel_path, "ao13_2021070412_radar512_5min.nc")
print(filename)
data_array_xr = pysteps.io.import_rmi_nwp_xr(filename)

expected_proj = "+proj=stere +lon_0=4.368 +lat_0=90 +lon_ts=0 +lat_ts=50 +ellps=sphere +x_0=356406 +y_0=3698905"


def test_io_import_rmi_nwp_xarray():
    """Test the importer RMI NWP."""
    assert isinstance(data_array_xr, xr.DataArray)


def test_io_import_rmi_nwp_xarray_shape():
    """Test the importer RMI NWP shape."""
    assert isinstance(data_array_xr, xr.DataArray)
    assert data_array_xr.shape == (12, 512, 512)


test_attrs_xr = [
    ("projection", expected_proj, None),
    ("institution", "Royal Meteorological Institute of Belgium", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 0.1),
    ("unit", "mm", None),
    ("accutime", np.timedelta64(5, "m"), None),
    ("zr_a", None, None),
    ("zr_b", None, None),
    ("xpixelsize", 1058.0, 0.1),
    ("ypixelsize", 1058.0, 0.1),
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
    ("x2", 540638.0, 0.1),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr_coord_x)
def test_io_import_rmi_nwp_xarray_attrs_coordx(variable, expected, tolerance):
    """Test the importer RMI NWP."""
    smart_assert(data_array_xr.x.attrs[variable], expected, tolerance)


test_attrs_xr_coord_y = [
    ("units", "m", None),
    ("y1", -540638.0, 0.1),
    ("y2", 0.0, 0.1),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr_coord_y)
def test_io_import_rmi_nwp_xarray_attrs_coordy(variable, expected, tolerance):
    """Test the importer RMI NWP."""
    smart_assert(data_array_xr.y.attrs[variable], expected, tolerance)
