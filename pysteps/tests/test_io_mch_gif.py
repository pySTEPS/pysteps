# -*- coding: utf-8 -*-

import os

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("PIL")

root_path = pysteps.rcparams.data_sources["mch"]["root_path"]
filename = os.path.join(root_path, "20170131", "AQC170310945F_00005.801.gif")
precip, _, metadata = pysteps.io.import_mch_gif(filename, "AQC", "mm", 5.0)


def test_io_import_mch_gif_shape():
    """Test the importer MCH GIF."""
    assert precip.shape == (640, 710)


expected_proj = (
    "+proj=somerc  +lon_0=7.43958333333333 "
    "+lat_0=46.9524055555556 +k_0=1 "
    "+x_0=600000 +y_0=200000 +ellps=bessel "
    "+towgs84=674.374,15.056,405.346,0,0,0,0 "
    "+units=m +no_defs"
)

# list of (variable,expected,tolerance) tuples
test_attrs = [
    ("projection", expected_proj, None),
    ("institution", "MeteoSwiss", None),
    ("accutime", 5.0, 0.1),
    ("unit", "mm", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 0.1),
    ("threshold", 0.0009628129986471908, 1e-19),
    ("zr_a", 316.0, 0.1),
    ("zr_b", 1.5, 0.1),
    ("x1", 255000.0, 0.1),
    ("y1", -160000.0, 0.1),
    ("x2", 965000.0, 0.1),
    ("y2", 480000.0, 0.1),
    ("xpixelsize", 1000.0, 0.1),
    ("ypixelsize", 1000.0, 0.1),
    ("cartesian_unit", "m", None),
    ("yorigin", "upper", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs)
def test_io_import_mch_gif_dataset_attrs(variable, expected, tolerance):
    """Test the importer MCH GIF."""
    smart_assert(metadata[variable], expected, tolerance)


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
