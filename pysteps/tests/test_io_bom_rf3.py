# -*- coding: utf-8 -*-

import os

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip('netCDF4')

def test_io_import_bom_rf3_shape():
    """Test the importer Bom RF3."""
    root_path = pysteps.rcparams.data_sources["bom"]["root_path"]
    rel_path = os.path.join("prcp-cscn", "2", "2018", "06", "16")
    filename = os.path.join(root_path, rel_path,
                            "2_20180616_100000.prcp-cscn.nc")
    precip, _, _ = pysteps.io.import_bom_rf3(filename)
    assert precip.shape == (512, 512)


expected_proj1 = ('+proj=aea  +lon_0=144.752 +lat_0=-37.852 '
                  '+lat_1=-18.000 +lat_2=-36.000')

# test_metadata_bom: list of (variable,expected,tolerance) tuples
test_metadata_bom = [
    ('transform', None, None),
    ('zerovalue', 0.0, 0.1),
    ('projection', expected_proj1, None),
    ('unit', 'mm', None),
    ('accutime', 6, 0.1),
    ('x1', -128000.0, 0.1),
    ('x2', 127500.0, 0.1),
    ('y1', -127500.0, 0.1),
    ('y2', 128000.0, 0.1),
    ('xpixelsize', 500.0, 0.1),
    ('ypixelsize', 500.0, 0.1),
    ('yorigin', 'upper', None),
    ('institution', 'Commonwealth of Australia, Bureau of Meteorology', None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_metadata_bom)
def test_io_import_bom_rf3_metadata(variable, expected, tolerance):
    """Test the importer Bom RF3."""
    root_path = pysteps.rcparams.data_sources["bom"]["root_path"]
    rel_path = os.path.join("prcp-cscn", "2", "2018", "06", "16")
    filename = os.path.join(root_path,
                            rel_path, "2_20180616_100000.prcp-cscn.nc")
    _, _, metadata = pysteps.io.import_bom_rf3(filename)
    smart_assert(metadata[variable], expected, tolerance)


expected_proj2 = ('+proj=aea  +lon_0=144.752 +lat_0=-37.852 '
                  '+lat_1=-18.000 +lat_2=-36.000')
# test_geodata: list of (variable,expected,tolerance) tuples
test_geodata_bom = [
    ('projection', expected_proj2, None),
    ('unit', 'mm', None),
    ('accutime', 6, 0.1),
    ('x1', -128000.0, 0.1),
    ('x2', 127500.0, 0.1),
    ('y1', -127500.0, 0.1),
    ('y2', 128000.0, 0.1),
    ('xpixelsize', 500.0, 0.1),
    ('ypixelsize', 500.0, 0.1),
    ('yorigin', 'upper', None),
    ('institution', 'Commonwealth of Australia, Bureau of Meteorology', None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_geodata_bom)
def test_io_import_bom_rf3_geodata(variable, expected, tolerance):
    """Test the importer Bom RF3."""
    root_path = pysteps.rcparams.data_sources["bom"]["root_path"]
    rel_path = os.path.join("prcp-cscn", "2", "2018", "06", "16")
    filename = os.path.join(root_path, rel_path,
                            "2_20180616_100000.prcp-cscn.nc")
    geodata = pysteps.io.importers._import_bom_rf3_geodata(filename)
    smart_assert(geodata[variable], expected, tolerance)
