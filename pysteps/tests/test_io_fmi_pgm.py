# -*- coding: utf-8 -*-

import pytest
import os
import pysteps


def test_io_import_fmi_pgm_shape():
    """Test the importer FMI PGM."""
    root_path = pysteps.rcparams.data_sources["fmi"]["root_path"]
    filename = os.path.join(root_path, "20160928",
        "201609281600_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz")
    R, _, _ = pysteps.io.import_fmi_pgm(filename, gzipped=True)
    assert R.shape == (1226, 760)


test_metadata = [
    ('composite_area', ['FIN']),
    ('projection_name', ['SUOMI1']),
    ('radar', ['LUO', '1', '26.9008', '67.1386']),
    ('obstime', ['201609281600']),
    ('producttype', ['CAPPI']),
    ('productname', ['LOWEST']),
    ('param', ['CorrectedReflectivity']),
    ('metersperpixel_x', ['999.674053']), 
    ('metersperpixel_y', ['999.62859']), 
    ('projection', ['radar', '{']), 
    ('type', ['stereographic']), 
    ('centrallongitude', ['25']), 
    ('centrallatitude', ['90']), 
    ('truelatitude', ['60']), 
    ('bottomleft', ['18.600000', '57.930000']), 
    ('topright', ['34.903000', '69.005000']), 
    ('missingval', 255),
    ]


@pytest.mark.parametrize("variable,expected", test_metadata)
def test_io_import_fmi_pmg_metadata(variable, expected):
    """Test the importer FMI PMG."""
    root_path = pysteps.rcparams.data_sources["fmi"]["root_path"]
    filename = os.path.join(root_path, "20160928",
        "201609281600_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz")
    metadata = pysteps.io.importers._import_fmi_pgm_metadata(filename, gzipped=True)
    print(metadata)
    assert metadata[variable] == expected


test_geodata = [
    ('projection',
     '+proj=stere  +lon_0=25E +lat_0=90N +lat_ts=60 +a=6371288 +x_0=380886.310 +y_0=3395677.920 +no_defs'),
    ('x1', 0.0049823258887045085),
    ('x2', 759752.2852757066),
    ('y1', 0.009731985162943602),
    ('y2', 1225544.6588913496),
    ('xpixelsize', 999.674053),
    ('ypixelsize', 999.62859),
    ('yorigin', 'upper'),
]



@pytest.mark.parametrize("variable,expected", test_geodata)
def test_io_import_fmi_pgm_geodata(variable, expected):
    """Test the importer FMI pgm."""
    root_path = pysteps.rcparams.data_sources["fmi"]["root_path"]
    filename = os.path.join(root_path, "20160928",
        "201609281600_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz")
    metadata = pysteps.io.importers._import_fmi_pgm_metadata(filename, gzipped=True)
    geodata = pysteps.io.importers._import_fmi_pgm_geodata(metadata)
    print(geodata)
    assert geodata[variable] == expected
