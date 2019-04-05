# -*- coding: utf-8 -*-

import pytest
import os
import pysteps


def test_io_import_bom_rf3_shape():
    """Test the importer Bom RF3."""
    root_path = pysteps.rcparams.data_sources["bom"]["root_path"]
    rel_path = os.path.join("radar", "bom", "prcp-cscn", "2", "2018", "06", "16")
    filename = os.path.join(root_path, rel_path, "2_20180616_100000.prcp-cscn.nc")
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
    rootpath = pysteps.io.get_pysteps_data_rootpath()
    relpath = os.path.join("radar", "bom", "prcp-cscn", "2", "2018", "06", "16")
    filename = os.path.join(rootpath, relpath, "2_20180616_100000.prcp-cscn.nc")
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
    rootpath = pysteps.io.get_pysteps_data_rootpath()
    relpath = os.path.join("radar", "bom", "prcp-cscn", "2", "2018", "06", "16")
    filename = os.path.join(rootpath, relpath, "2_20180616_100000.prcp-cscn.nc")
    geodata = pysteps.io.importers._import_bom_rf3_geodata(filename)
    assert geodata[variable] == expected
