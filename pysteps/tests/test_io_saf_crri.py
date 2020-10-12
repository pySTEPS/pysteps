# -*- coding: utf-8 -*-

import os

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("netCDF4")


expected_proj = (
    "+proj=geos +a=6378137.000000 +b=6356752.300000 "
    "+lon_0=0.000000 +h=35785863.000000"
)
test_geodata_crri = [
    ("projection", expected_proj, None),
    ("x1", -3301500.0, 0.1),
    ("x2", 3298500.0, 0.1),
    ("y1", 2512500.0, 0.1),
    ("y2", 5569500.0, 0.1),
    ("xpixelsize", 3000.0, 0.1),
    ("ypixelsize", 3000.0, 0.1),
    ("cartesian_unit", "m", None),
    ("yorigin", "upper", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_geodata_crri)
def test_io_import_saf_crri_geodata(variable, expected, tolerance):
    """Test the importer SAF CRRI."""
    root_path = pysteps.rcparams.data_sources["saf"]["root_path"]
    rel_path = "20180601/CRR"
    filename = os.path.join(
        root_path, rel_path, "S_NWC_CRR_MSG4_Europe-VISIR_20180601T070000Z.nc"
    )
    geodata = pysteps.io.importers._import_saf_crri_geodata(filename)
    print(variable, geodata[variable], expected, tolerance)
    smart_assert(geodata[variable], expected, tolerance)


test_metadata_crri = [
    ("transform", None, None),
    ("zerovalue", 0.0, 0.1),
    ("unit", "mm/h", None),
    ("accutime", None, None),
    ("institution", "Agencia Estatal de Meteorología (AEMET)", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_metadata_crri)
def test_io_import_saf_crri_metadata(variable, expected, tolerance):
    """Test the importer SAF CRRI."""
    root_path = pysteps.rcparams.data_sources["saf"]["root_path"]
    rel_path = "20180601/CRR"
    filename = os.path.join(
        root_path, rel_path, "S_NWC_CRR_MSG4_Europe-VISIR_20180601T070000Z.nc"
    )
    _, _, metadata = pysteps.io.import_saf_crri(filename)
    print(variable, metadata[variable], expected, tolerance)
    smart_assert(metadata[variable], expected, tolerance)


test_extent_crri = [
    (None, (-3301500.0, 3298500.0, 2512500.0, 5569500.0), (1019, 2200), None),
    (
        (-1980000.0, 1977000.0, 2514000.0, 4818000.0),
        (-1978500.0, 1975500.0, 2515500.0, 4816500.0),
        (767, 1318),
        None,
    ),
]


@pytest.mark.parametrize(
    "extent, expected_extent, expected_shape, tolerance", test_extent_crri
)
def test_io_import_saf_crri_extent(extent, expected_extent, expected_shape, tolerance):
    """Test the importer SAF CRRI."""
    root_path = pysteps.rcparams.data_sources["saf"]["root_path"]
    rel_path = "20180601/CRR"
    filename = os.path.join(
        root_path, rel_path, "S_NWC_CRR_MSG4_Europe-VISIR_20180601T070000Z.nc"
    )
    precip, _, metadata = pysteps.io.import_saf_crri(filename, extent=extent)
    extent_out = (metadata["x1"], metadata["x2"], metadata["y1"], metadata["y2"])

    print(extent, extent_out, expected_extent)
    smart_assert(extent_out, expected_extent, tolerance)
    print(precip.shape, expected_shape)
    smart_assert(precip.shape, expected_shape, tolerance)


if __name__ == "__main__":

    for i, args in enumerate(test_geodata_crri):
        test_io_import_saf_crri_geodata(args[0], args[1], args[2])

    for i, args in enumerate(test_metadata_crri):
        test_io_import_saf_crri_metadata(args[0], args[1], args[2])

    for i, args in enumerate(test_extent_crri):
        test_io_import_saf_crri_extent(args[0], args[1], args[2], args[3])
