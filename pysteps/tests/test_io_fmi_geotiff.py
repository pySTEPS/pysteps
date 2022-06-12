import os

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("pyproj")
pytest.importorskip("osgeo")

root_path = pysteps.rcparams.data_sources["fmi_geotiff"]["root_path"]
filename = os.path.join(
    root_path,
    "20160928",
    "201609281600_FINUTM.tif",
)
precip, _, metadata = pysteps.io.import_fmi_geotiff(filename)


def test_io_import_fmi_geotiff_shape():
    """Test the shape of the read file."""
    assert precip.shape == (7316, 4963)


expected_proj = (
    "+proj=utm +zone=35 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
)

# test_geodata: list of (variable,expected,tolerance) tuples
test_geodata = [
    ("projection", expected_proj, None),
    ("x1", -196593.0043142295908183, 1e-10),
    ("x2", 1044176.9413554778, 1e-10),
    ("y1", 6255329.6988206729292870, 1e-10),
    ("y2", 8084432.005259146, 1e-10),
    ("xpixelsize", 250.0040188736061566, 1e-6),
    ("ypixelsize", 250.0139839309011904, 1e-6),
    ("cartesian_unit", "m", None),
    ("yorigin", "upper", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_geodata)
def test_io_import_fmi_pgm_geodata(variable, expected, tolerance):
    """Test the GeoTIFF and metadata reading."""
    smart_assert(metadata[variable], expected, tolerance)
