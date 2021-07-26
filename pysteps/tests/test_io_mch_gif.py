# -*- coding: utf-8 -*-

import os

import pytest
import xarray as xr

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("PIL")

root_path = pysteps.rcparams.data_sources["mch"]["root_path"]
filename = os.path.join(root_path, "20170131", "AQC170310945F_00005.801.gif")
precip_ds = pysteps.io.import_mch_gif(filename, "AQC", 5.0)
print(precip_ds)

precip_ds.precipitation.plot()
import matplotlib.pyplot as plt

plt.tight_layout()
plt.savefig("test_mch.png")


def test_io_import_mch_gif_shape():
    """Test the importer MCH GIF."""
    assert isinstance(precip_ds, xr.Dataset)
    assert "precipitation" in precip_ds
    assert precip_ds.precipitation.shape == (640, 710)


# list of (variable,expected,tolerance) tuples
test_dataset_attrs = [
    ("crs", "EPSG:21781", None),
    ("institution", "MeteoSwiss (NMC Switzerland)", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_dataset_attrs)
def test_io_import_mch_gif_dataset_attrs(variable, expected, tolerance):
    """Test the importer MCH GIF."""
    smart_assert(precip_ds.attrs[variable], expected, tolerance)


# list of (variable,expected,tolerance) tuples
test_array_attrs = [
    ("standard_name", "rainfall_rate", None),
    ("long_name", "Precipitation intensity", None),
    ("units", "mm h-1", None),
    ("radar_product", "AQC", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_array_attrs)
def test_io_import_mch_gif_array_attrs(variable, expected, tolerance):
    """Test the importer MCH GIF."""
    smart_assert(precip_ds.precipitation.attrs[variable], expected, tolerance)


expected_proj = (
    "+proj=somerc  +lon_0=7.43958333333333 +lat_0=46.9524055555556 "
    "+k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel "
    "+towgs84=674.374,15.056,405.346,0,0,0,0 "
    "+units=m +no_defs"
)

# test_geodata: list of (variable,expected,tolerance) tuples
test_geodata = [
    ("projection", expected_proj, None),
    ("x1", 255000.0, 0.1),
    ("y1", -160000.0, 0.1),
    ("x2", 965000.0, 0.1),
    ("y2", 480000.0, 0.1),
    ("xpixelsize", 1000.0, 0.1),
    ("ypixelsize", 1000.0, 0.1),
    ("cartesian_unit", "m", None),
    ("yorigin", "upper", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_geodata)
def test_io_import_mch_geodata(variable, expected, tolerance):
    """Test the importer MCH geodata."""
    geodata = pysteps.io.importers._import_mch_geodata()
    smart_assert(geodata[variable], expected, tolerance)
