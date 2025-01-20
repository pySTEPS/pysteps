# -*- coding: utf-8 -*-

import os

import pytest

import pysteps
from pysteps.tests.helpers import smart_assert

pytest.importorskip("h5py")

# tests for three OPERA products:
# Odyssey rain rate composite (production discontinued on October 30th 2024)
# CIRRUS max. reflectivity composites
# NIMBUS rain rate composites

root_path = pysteps.rcparams.data_sources["opera"]["root_path"]

filename = os.path.join(root_path, "20180824", "T_PAAH21_C_EUOC_20180824180000.hdf")
precip_odyssey, _, metadata_odyssey = pysteps.io.import_opera_hdf5(filename, qty="RATE")

filename = os.path.join(
    root_path, "20241126", "CIRRUS", "T_PABV21_C_EUOC_20241126010000.hdf"
)
precip_cirrus, _, metadata_cirrus = pysteps.io.import_opera_hdf5(filename, qty="DBZH")

filename = os.path.join(
    root_path, "20241126", "NIMBUS", "T_PAAH22_C_EUOC_20241126010000.hdf"
)
precip_nimbus_rain_rate, _, metadata_nimbus_rain_rate = pysteps.io.import_opera_hdf5(
    filename, qty="RATE"
)

filename = os.path.join(
    root_path, "20241126", "NIMBUS", "T_PASH22_C_EUOC_20241126010000.hdf"
)
precip_nimbus_rain_accum, _, metadata_nimbus_rain_accum = pysteps.io.import_opera_hdf5(
    filename, qty="ACRR"
)


def test_io_import_opera_hdf5_odyssey_shape():
    """Test the importer OPERA HDF5."""
    assert precip_odyssey.shape == (2200, 1900)


def test_io_import_opera_hdf5_cirrus_shape():
    """Test the importer OPERA HDF5."""
    assert precip_cirrus.shape == (4400, 3800)


def test_io_import_opera_hdf5_nimbus_rain_rate_shape():
    """Test the importer OPERA HDF5."""
    assert precip_nimbus_rain_rate.shape == (2200, 1900)


def test_io_import_opera_hdf5_nimbus_rain_accum_shape():
    """Test the importer OPERA HDF5."""
    assert precip_nimbus_rain_accum.shape == (2200, 1900)


# test_metadata: list of (variable,expected, tolerance) tuples
expected_proj = (
    "+proj=laea +lat_0=55.0 +lon_0=10.0 "
    "+x_0=1950000.0 "
    "+y_0=-2100000.0 "
    "+units=m +ellps=WGS84"
)

# list of (variable,expected,tolerance) tuples
test_odyssey_attrs = [
    ("projection", expected_proj, None),
    ("ll_lon", -10.434576838640398, 1e-10),
    ("ll_lat", 31.746215319325056, 1e-10),
    ("ur_lon", 57.81196475014995, 1e-10),
    ("ur_lat", 67.62103710275053, 1e-10),
    ("x1", -0.0004161088727414608, 1e-6),
    ("y1", -4400000.001057557, 1e-10),
    ("x2", 3800000.0004256153, 1e-10),
    ("y2", -0.0004262728616595268, 1e-6),
    ("xpixelsize", 2000.0, 1e-10),
    ("xpixelsize", 2000.0, 1e-10),
    ("cartesian_unit", "m", None),
    ("accutime", 15.0, 1e-10),
    ("yorigin", "upper", None),
    ("unit", "mm/h", None),
    ("institution", "Odyssey datacentre", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 1e-10),
    ("threshold", 0.01, 1e-10),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_odyssey_attrs)
def test_io_import_opera_hdf5_odyssey_dataset_attrs(variable, expected, tolerance):
    """Test the importer OPERA HDF5."""
    smart_assert(metadata_odyssey[variable], expected, tolerance)


# list of (variable,expected,tolerance) tuples
test_cirrus_attrs = [
    ("projection", expected_proj, None),
    ("ll_lon", -10.4345768386404, 1e-10),
    ("ll_lat", 31.7462153182675, 1e-10),
    ("ur_lon", 57.8119647501499, 1e-10),
    ("ur_lat", 67.6210371071631, 1e-10),
    ("x1", -0.00027143326587975025, 1e-6),
    ("y1", -4400000.00116988, 1e-10),
    ("x2", 3800000.0000817003, 1e-10),
    ("y2", -8.761277422308922e-05, 1e-6),
    ("xpixelsize", 1000.0, 1e-10),
    ("ypixelsize", 1000.0, 1e-10),
    ("cartesian_unit", "m", None),
    ("accutime", 15.0, 1e-10),
    ("yorigin", "upper", None),
    ("unit", "dBZ", None),
    ("institution", "Odyssey datacentre", None),
    ("transform", "dB", None),
    ("zerovalue", -32.0, 1e-10),
    ("threshold", -31.5, 1e-10),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_cirrus_attrs)
def test_io_import_opera_hdf5_cirrus_dataset_attrs(variable, expected, tolerance):
    """Test OPERA HDF5 importer: max. reflectivity composites from CIRRUS."""
    smart_assert(metadata_cirrus[variable], expected, tolerance)


# list of (variable,expected,tolerance) tuples
test_nimbus_rain_rate_attrs = [
    ("projection", expected_proj, None),
    ("ll_lon", -10.434599999137568, 1e-10),
    ("ll_lat", 31.74619995126678, 1e-10),
    ("ur_lon", 57.8119032106317, 1e-10),
    ("ur_lat", 67.62104536996274, 1e-10),
    ("x1", -2.5302714337594807, 1e-6),
    ("y1", -4400001.031169886, 1e-10),
    ("x2", 3799997.4700817037, 1e-10),
    ("y2", -1.0300876162946224, 1e-6),
    ("xpixelsize", 2000.0, 1e-10),
    ("ypixelsize", 2000.0, 1e-10),
    ("cartesian_unit", "m", None),
    ("accutime", 15.0, 1e-10),
    ("yorigin", "upper", None),
    ("unit", "mm/h", None),
    ("institution", "Odyssey datacentre", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 1e-10),
    ("threshold", 0.01, 1e-10),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_nimbus_rain_rate_attrs)
def test_io_import_opera_hdf5_nimbus_rain_rate_dataset_attrs(
    variable, expected, tolerance
):
    """Test OPERA HDF5 importer: rain rate composites from NIMBUS."""
    smart_assert(metadata_nimbus_rain_rate[variable], expected, tolerance)


# list of (variable,expected,tolerance) tuples
test_nimbus_rain_accum_attrs = [
    ("projection", expected_proj, None),
    ("ll_lon", -10.434599999137568, 1e-10),
    ("ll_lat", 31.74619995126678, 1e-10),
    ("ur_lon", 57.8119032106317, 1e-10),
    ("ur_lat", 67.62104536996274, 1e-10),
    ("x1", -2.5302714337594807, 1e-6),
    ("y1", -4400001.031169886, 1e-10),
    ("x2", 3799997.4700817037, 1e-10),
    ("y2", -1.0300876162946224, 1e-6),
    ("xpixelsize", 2000.0, 1e-10),
    ("ypixelsize", 2000.0, 1e-10),
    ("cartesian_unit", "m", None),
    ("accutime", 15.0, 1e-10),
    ("yorigin", "upper", None),
    ("unit", "mm", None),
    ("institution", "Odyssey datacentre", None),
    ("transform", None, None),
    ("zerovalue", 0.0, 1e-10),
    ("threshold", 0.01, 1e-10),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_nimbus_rain_accum_attrs)
def test_io_import_opera_hdf5_nimbus_rain_accum_dataset_attrs(
    variable, expected, tolerance
):
    """Test OPERA HDF5 importer: rain accumulation composites from NIMBUS."""
    smart_assert(metadata_nimbus_rain_accum[variable], expected, tolerance)
