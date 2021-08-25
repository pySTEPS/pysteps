# -*- coding: utf-8 -*-

import os
import numpy as np

import xarray as xr
import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("netCDF4")

root_path = pysteps.rcparams.data_sources["knmi_nwp"]["root_path"]
rel_path = os.path.join("2018", "09", "05")
filename = os.path.join(root_path, rel_path, "20180905_0600_Pforecast_Harmonie.nc")
data_array_xr = pysteps.io.import_knmi_nwp_xr(filename)

expected_proj = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"


def test_io_import_knmi_nwp_xarray():
    """Test the KNMI NWP Harmonie importer."""
    assert isinstance(data_array_xr, xr.DataArray)


def test_io_import_knmi_nwp_xarray_shape():
    """Test the importer KNMI NWP shape."""
    assert isinstance(data_array_xr, xr.DataArray)
    assert data_array_xr.shape == (49, 300, 300)


test_attrs_xr = [
    ("projection", expected_proj, None),
    ("institution", "  Royal Netherlands Meteorological Institute (KNMI)  ", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 0.1),
    ("unit", "mm", None),
    ("accutime", np.timedelta64(60, "m"), None),
    ("zr_a", None, None),
    ("zr_b", None, None),
    ("xpixelsize", 0.037, 0.0001),
    ("ypixelsize", 0.023, 0.0001),
    ("units", "mm", None),
    ("yorigin", "lower", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr)
def test_io_import_knmi_nwp_xarray_attrs(variable, expected, tolerance):
    """Test the KNMI Harmonie NWP importer."""
    smart_assert(data_array_xr.attrs[variable], expected, tolerance)


test_attrs_xr_coord_x = [
    ("units", "degrees_east", None),
    ("x1", 0.0, 0.0001),
    ("x2", 11.063, 0.0001),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr_coord_x)
def test_io_import_knmi_nwp_xarray_attrs_coordx(variable, expected, tolerance):
    """Test the grid of the KNMI Harmonie NWP importer in the x-direction."""
    smart_assert(data_array_xr.x.attrs[variable], expected, tolerance)


test_attrs_xr_coord_y = [
    ("units", "degrees_north", None),
    ("y1", 49.0, 0.0001),
    ("y2", 55.877, 0.0001),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs_xr_coord_y)
def test_io_import_knmi_nwp_xarray_attrs_coordy(variable, expected, tolerance):
    """Test the grid of the KNMI Harmonie NWP importer in the y-direction."""
    smart_assert(data_array_xr.y.attrs[variable], expected, tolerance)
