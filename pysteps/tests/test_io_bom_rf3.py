# -*- coding: utf-8 -*-

import os
import numpy as np

import xarray as xr
import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("netCDF4")

root_path = pysteps.rcparams.data_sources["bom"]["root_path"]
rel_path = os.path.join("prcp-cscn", "2", "2018", "06", "16")
filename = os.path.join(root_path, rel_path, "2_20180616_100000.prcp-cscn.nc")
data_array = pysteps.io.import_bom_rf3(filename)


def test_io_import_bom_rf3_shape():
    """Test the importer Bom RF3."""
    assert isinstance(data_array, xr.DataArray)
    assert data_array.shape == (512, 512)


expected_proj = (
    "+proj=aea  +lon_0=144.752 +lat_0=-37.852 " "+lat_1=-18.000 +lat_2=-36.000"
)

test_attrs = [
    ("projection", expected_proj, None),
    ("institution", "Commonwealth of Australia, Bureau of Meteorology", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 0.1),
    ("unit", "mm", None),
    ("accutime", 6, 0.1),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs)
def test_io_import_bom_rf3_dataset_attrs(variable, expected, tolerance):
    """Test the importer Bom RF3."""
    smart_assert(data_array.attrs[variable], expected, tolerance)


# test_geodata: list of (variable,expected,tolerance) tuples
test_geodata_bom = [
    ("projection", expected_proj, None),
    ("unit", "mm", None),
    ("accutime", 6, 0.1),
    ("x1", -128000.0, 0.1),
    ("x2", 127500.0, 0.1),
    ("y1", -127500.0, 0.1),
    ("y2", 128000.0, 0.1),
    ("xpixelsize", 500.0, 0.1),
    ("ypixelsize", 500.0, 0.1),
    ("cartesian_unit", "m", None),
    ("yorigin", "upper", None),
    ("institution", "Commonwealth of Australia, Bureau of Meteorology", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_geodata_bom)
def test_io_import_bom_rf3_geodata(variable, expected, tolerance):
    """Test the importer Bom RF3."""
    root_path = pysteps.rcparams.data_sources["bom"]["root_path"]
    rel_path = os.path.join("prcp-cscn", "2", "2018", "06", "16")
    filename = os.path.join(root_path, rel_path, "2_20180616_100000.prcp-cscn.nc")
    geodata = pysteps.io.importers._import_bom_rf3_geodata(filename)
    smart_assert(geodata[variable], expected, tolerance)


# TEST XARRAY IMPLEMENTATION
data_array_xr = pysteps.io.import_bom_rf3_xr(filename)


def test_io_import_bom_rf3_xarray():
    """Test the importer Bom RF3."""
    assert isinstance(data_array_xr, xr.DataArray)


def test_io_import_bom_rf3_xarray_shape():
    """Test the importer Bom RF3."""
    assert isinstance(data_array_xr, xr.DataArray)
    assert data_array_xr.shape == (1, 512, 512)


test_attrs_xr = [
    ("projection", expected_proj, None),
    ("institution", "Commonwealth of Australia, Bureau of Meteorology", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 0.1),
    ("unit", "mm", None),
    ("accutime", np.timedelta64(6, "m"), None),
    ("zr_a", None, None),
    ("zr_b", None, None),
    ("xpixelsize", 500.0, 0.1),
    ("ypixelsize", 500.0, 0.1),
    ("units", "mm", None),
    ("yorigin", "upper", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr)
def test_io_import_bom_rf3_xarray_attrs(variable, expected, tolerance):
    """Test the importer Bom RF3."""
    smart_assert(data_array_xr.attrs[variable], expected, tolerance)


test_attrs_xr_coord_x = [
    ("units", "m", None),
    ("x1", -128000.0, 0.1),
    ("x2", 127500.0, 0.1),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr_coord_x)
def test_io_import_bom_rf3_xarray_attrs_coordx(variable, expected, tolerance):
    """Test the importer Bom RF3."""
    smart_assert(data_array_xr.x.attrs[variable], expected, tolerance)


test_attrs_xr_coord_y = [
    ("units", "m", None),
    ("y1", -127500.0, 0.1),
    ("y2", 128000.0, 0.1),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr_coord_y)
def test_io_import_bom_rf3_xarray_attrs_coordy(variable, expected, tolerance):
    """Test the importer Bom RF3."""
    smart_assert(data_array_xr.y.attrs[variable], expected, tolerance)
