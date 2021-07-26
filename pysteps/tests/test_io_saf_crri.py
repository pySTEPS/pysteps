# -*- coding: utf-8 -*-

import os

import numpy as np
import xarray as xr
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
    smart_assert(geodata[variable], expected, tolerance)


root_path = pysteps.rcparams.data_sources["saf"]["root_path"]
rel_path = "20180601/CRR"
filename = os.path.join(
    root_path, rel_path, "S_NWC_CRR_MSG4_Europe-VISIR_20180601T070000Z.nc"
)
precip_ds = pysteps.io.import_saf_crri(filename)

# list of (variable,expected,tolerance) tuples
test_dataset_attrs = [
    ("projection", expected_proj, None),
    ("institution", "Agencia Estatal de Meteorología (AEMET)", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_dataset_attrs)
def test_io_import_mch_gif_dataset_attrs(variable, expected, tolerance):
    """Test the importer SAF CRRI."""
    smart_assert(precip_ds.attrs[variable], expected, tolerance)


test_array_attrs = [
    ("transform", None, None),
    ("zerovalue", 0.0, 0.1),
    ("unit", "mm/h", None),
    ("accutime", None, None),
]

@pytest.mark.parametrize("variable, expected, tolerance", test_array_attrs)
def test_io_import_saf_crri_metadata(variable, expected, tolerance):
    """Test the importer SAF CRRI."""
    smart_assert(precip_ds.precipitation.attrs[variable], expected, tolerance)


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
    precip_ds = pysteps.io.import_saf_crri(filename, extent=extent)

    xgridsize = np.diff(precip_ds.x)[0]
    ygridsize = np.diff(precip_ds.y)[0]
    extent_out = (
        precip_ds.x.min() - xgridsize / 2,
        precip_ds.x.max() + xgridsize / 2,
        precip_ds.y.min() - ygridsize / 2,
        precip_ds.y.max() + ygridsize / 2,
    )
    print([int(out) for out in extent_out])
    smart_assert(extent_out, expected_extent, tolerance)

    smart_assert(precip_ds.precipitation.shape, expected_shape, tolerance)
