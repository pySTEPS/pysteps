# -*- coding: utf-8 -*-

import os

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip('netCDF4')


def test_io_import_saf_crri_shape():
    """Test the importer SAF CRRI."""
    root_path = pysteps.rcparams.data_sources["crri"]["root_path"]
    rel_path = "20180601/CRR"
    filename = os.path.join(root_path, rel_path,
                            "S_NWC_CRR_MSG4_Europe-VISIR_20180601T070000Z.nc")
    precip = pysteps.io.importers._import_crri_eu_data(filename)
    assert precip.shape == (2200, 1019)


expected_proj = ("+proj=geos +a=6378137.000000 +b=6356752.300000 "
                 "+lon_0=0.000000 +h=35785863.000000")

# test_metadata_crri: list of (variable,expected,tolerance) tuples
test_metadata_crri = [
    ('projection', expected_proj, None),
    ('unit', "mm/h", None),
    ('accutime', None, None),
    ('x1', -3300000.0, 0.1),
    ('x2', 3297000.0, 0.1),
    ('y1', 2514000.0, 0.1),
    ('y2', 5568000.0, 0.1),
    ('xpixelsize', 3000.0, 0.1),
    ('ypixelsize', 3000.0, 0.1),
    ('yorigin', 'upper', None),
    ('institution', "SAF AEMET", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_metadata_crri)
def test_io_import_saf_crri_geodata(variable, expected, tolerance):
    """Test the importer SAF CRRI."""
    root_path = pysteps.rcparams.data_sources["crri"]["root_path"]
    rel_path = "20180601/CRR"
    filename = os.path.join(root_path, rel_path,
                            "S_NWC_CRR_MSG4_Europe-VISIR_20180601T070000Z.nc")
    geodata = pysteps.io.importers._import_crri_eu_geodata(filename)
    smart_assert(geodata[variable], expected, tolerance)
