# -*- coding: utf-8 -*-

import os
import numpy as np

import xarray as xr
import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("netCDF4")

root_path = pysteps.rcparams.data_sources["bom_nwp"]["root_path"]
rel_path = os.path.join("2020", "10", "31")
filename = os.path.join(root_path, rel_path, "20201031_0000_regrid_short.nc")
data_array_xr = pysteps.io.import_bom_nwp_xr(filename)

expected_proj = "+proj=aea  +lon_0=153.240 +lat_0=-27.718 +lat_1=-26.200 +lat_2=-29.300"


def test_io_import_bom_nwp_xarray():
    """Test the importer Bom NWP."""
    assert isinstance(data_array_xr, xr.DataArray)


def test_io_import_bom_nwp_xarray_shape():
    """Test the importer Bom NWP shape."""
    assert isinstance(data_array_xr, xr.DataArray)
    assert data_array_xr.shape == (144, 512, 512)


test_attrs_xr = [
    ("projection", expected_proj, None),
    ("institution", "Commonwealth of Australia, Bureau of Meteorology", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 0.1),
    ("unit", "mm", None),
    ("accutime", np.timedelta64(10, "m"), None),
    ("zr_a", None, None),
    ("zr_b", None, None),
    ("xpixelsize", 500.0, 0.1),
    ("ypixelsize", 500.0, 0.1),
    ("units", "mm", None),
    ("yorigin", "upper", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr)
def test_io_import_bom_nwp_xarray_attrs(variable, expected, tolerance):
    """Test the importer Bom RF3."""
    smart_assert(data_array_xr.attrs[variable], expected, tolerance)


test_attrs_xr_coord_x = [
    ("units", "m", None),
    ("x1", -127750.0, 0.1),
    ("x2", 127750.0, 0.1),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr_coord_x)
def test_io_import_bom_nwp_xarray_attrs_coordx(variable, expected, tolerance):
    """Test the importer Bom NWP."""
    smart_assert(data_array_xr.x.attrs[variable], expected, tolerance)


test_attrs_xr_coord_y = [
    ("units", "m", None),
    ("y1", -127750.0, 0.1),
    ("y2", 127750.0, 0.1),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr_coord_y)
def test_io_import_bom_nwp_xarray_attrs_coordy(variable, expected, tolerance):
    """Test the importer Bom NWP."""
    smart_assert(data_array_xr.y.attrs[variable], expected, tolerance)
