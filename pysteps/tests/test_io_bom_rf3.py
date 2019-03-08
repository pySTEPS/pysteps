# -*- coding: utf-8 -*-

import pytest
import os
import pysteps


def test_io_import_bom_rf3_shape():
    """Test the importer Bom RF3."""
    name = 'sample_data_bom_rf3.nc'
    filename = os.path.join(os.path.dirname(__file__), 'data', name)
    R, _, _ = pysteps.io.import_bom_rf3(filename)
    assert R.shape == (512, 512)


test_metadata_bom = [
    ('transform', None),
    ('zerovalue', 0.0),
    ('projection',
     '+proj=aea  +lon_0=144.752 +lat_0=-37.852 +lat_1=-18.000 +lat_2=-36.000'),
    ('unit', 'mm'),
    ('accutime', 6),
    ('x1', -128000.0),
    ('x2', 127500.0),
    ('y1', -127500.0),
    ('y2', 128000.0),
    ('xpixelsize', 500.0),
    ('ypixelsize', 500.0),
    ('yorigin', 'upper'),
    ('institution', 'Commonwealth of Australia, Bureau of Meteorology'),
]


@pytest.mark.parametrize("variable,expected", test_metadata_bom)
def test_io_import_bom_rf3_metadata(variable, expected):
    """Test the importer Bom RF3."""
    name = 'sample_data_bom_rf3.nc'
    filename = os.path.join(os.path.dirname(__file__), 'data', name)
    _, _, metadata = pysteps.io.import_bom_rf3(filename)
    assert metadata[variable] == expected


test_geodata_bom = [
    ('projection',
     '+proj=aea  +lon_0=144.752 +lat_0=-37.852 +lat_1=-18.000 +lat_2=-36.000'),
    ('unit', 'mm'),
    ('accutime', 6),
    ('x1', -128000.0),
    ('x2', 127500.0),
    ('y1', -127500.0),
    ('y2', 128000.0),
    ('xpixelsize', 500.0),
    ('ypixelsize', 500.0),
    ('yorigin', 'upper'),
    ('institution', 'Commonwealth of Australia, Bureau of Meteorology'),
]


@pytest.mark.parametrize("variable,expected", test_geodata_bom)
def test_io_import_bom_rf3_geodata(variable, expected):
    """Test the importer Bom RF3."""
    name = 'sample_data_bom_rf3.nc'
    filename = os.path.join(os.path.dirname(__file__), 'data', name)
    geodata = pysteps.io.importers._import_bom_rf3_geodata(filename)
    assert geodata[variable] == expected
