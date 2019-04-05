# -*- coding: utf-8 -*-

import pytest
import os
import pysteps


def test_io_import_mch_gif_shape():
    """Test the importer MCH GIF."""
    root_path = pysteps.rcparams.data_sources["mch"]["root_path"]
    filename = os.path.join(root_path, "20170131",
                            "AQC170310945F_00005.801.gif")
    R, _, metadata = pysteps.io.import_mch_gif(filename,'AQC','dB',5)
    print (R.shape)
    print (metadata)
    assert R.shape == (640, 710)

test_metadata = [
    ('projection',
    '+proj=somerc  +lon_0=7.43958333333333 +lat_0=46.9524055555556 +k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs'),
    ('x1', 255000.0),
    ('y1', -160000.0),
    ('x2', 965000.0),
    ('y2', 480000.0), 
    ('xpixelsize', 1000.0), 
    ('ypixelsize', 1000.0), 
    ('yorigin', 'upper'),
    ('accutime', 5), 
    ('unit', 'dB'), 
    ('transform', None), 
    ('zerovalue', 0.0), 
    ('threshold', 0.0009628129986471908), 
    ('institution', 'MeteoSwiss'), 
    ('product', 'AQC'),
    ]


@pytest.mark.parametrize("variable,expected", test_metadata)
def test_io_import_mch_gif_metadata(variable, expected):
    """Test the importer FMI PMG."""
    root_path = pysteps.rcparams.data_sources["mch"]["root_path"]
    filename = os.path.join(root_path, "20170131",
                            "AQC170310945F_00005.801.gif")
    _, _, metadata = pysteps.io.import_mch_gif(filename,'AQC','dB',5)
    print(metadata)
    assert metadata[variable] == expected


test_geodata = [
    ('projection',
    '+proj=somerc  +lon_0=7.43958333333333 +lat_0=46.9524055555556 +k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs'),
    ('x1', 255000.0),
    ('y1', -160000.0),
    ('x2', 965000.0),
    ('y2', 480000.0), 
    ('xpixelsize', 1000.0), 
    ('ypixelsize', 1000.0), 
    ('yorigin', 'upper'),
    ]



@pytest.mark.parametrize("variable,expected", test_geodata)
def test_io_import_mch_geodata(variable, expected):
    """Test the importer MCH."""
    geodata = pysteps.io.importers._import_mch_geodata()
    print(geodata)
    assert geodata[variable] == expected
