# -*- coding: utf-8 -*-

import os

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("pyproj")


root_path = pysteps.rcparams.data_sources["fmi"]["root_path"]
filename = os.path.join(
    root_path,
    "20160928",
    "201609281600_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
)
precip, _, metadata = pysteps.io.import_fmi_pgm(filename, gzipped=True)


def test_io_import_fmi_pgm_shape():
    """Test the importer FMI PGM."""
    assert precip.shape == (1226, 760)


expected_proj = (
    "+proj=stere  +lon_0=25E +lat_0=90N "
    "+lat_ts=60 +a=6371288 +x_0=380886.310 "
    "+y_0=3395677.920 +no_defs"
)

test_attrs = [
    ("projection", expected_proj, None),
    ("institution", "Finnish Meteorological Institute", None),
    # ("composite_area", ["FIN"]),
    # ("projection_name", ["SUOMI1"]),
    # ("radar", ["LUO", "1", "26.9008", "67.1386"]),
    # ("obstime", ["201609281600"]),
    # ("producttype", ["CAPPI"]),
    # ("productname", ["LOWEST"]),
    # ("param", ["CorrectedReflectivity"]),
    # ("metersperpixel_x", ["999.674053"]),
    # ("metersperpixel_y", ["999.62859"]),
    # ("projection", ["radar", "{"]),
    # ("type", ["stereographic"]),
    # ("centrallongitude", ["25"]),
    # ("centrallatitude", ["90"]),
    # ("truelatitude", ["60"]),
    # ("bottomleft", ["18.600000", "57.930000"]),
    # ("topright", ["34.903000", "69.005000"]),
    # ("missingval", 255),
    ("accutime", 5.0, 0.1),
    ("unit", "dBZ", None),
    ("transform", "dB", None),
    ("zerovalue", -32.0, 0.1),
    ("threshold", -31.5, 0.1),
    ("zr_a", 223.0, 0.1),
    ("zr_b", 1.53, 0.1),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs)
def test_io_import_mch_gif_dataset_attrs(variable, expected, tolerance):
    """Test the importer FMI PMG."""
    smart_assert(metadata[variable], expected, tolerance)


# test_geodata: list of (variable,expected,tolerance) tuples
test_geodata = [
    ("projection", expected_proj, None),
    ("x1", 0.0049823258887045085, 1e-20),
    ("x2", 759752.2852757066, 1e-10),
    ("y1", 0.009731985162943602, 1e-18),
    ("y2", 1225544.6588913496, 1e-10),
    ("xpixelsize", 999.674053, 1e-6),
    ("ypixelsize", 999.62859, 1e-5),
    ("cartesian_unit", "m", None),
    ("yorigin", "upper", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_geodata)
def test_io_import_fmi_pgm_geodata(variable, expected, tolerance):
    """Test the importer FMI pgm."""
    root_path = pysteps.rcparams.data_sources["fmi"]["root_path"]
    filename = os.path.join(
        root_path,
        "20160928",
        "201609281600_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz",
    )
    metadata = pysteps.io.importers._import_fmi_pgm_metadata(filename, gzipped=True)
    geodata = pysteps.io.importers._import_fmi_pgm_geodata(metadata)

    smart_assert(geodata[variable], expected, tolerance)
