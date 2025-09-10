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

# XR: since the pysteps.datasets module does not support all the OPERA data sources below, we dont use get_precipitation_fields
root_path = pysteps.rcparams.data_sources["opera"]["root_path"]

filename = os.path.join(root_path, "20180824", "T_PAAH21_C_EUOC_20180824180000.hdf")
precip_odyssey = pysteps.io.import_opera_hdf5(filename, qty="RATE")
precip_var = precip_odyssey.attrs["precip_var"]
precip_odyssey_dataarray = precip_odyssey[precip_var]


filename = os.path.join(
    root_path, "20241126", "CIRRUS", "T_PABV21_C_EUOC_20241126010000.hdf"
)
precip_cirrus = pysteps.io.import_opera_hdf5(filename, qty="DBZH")
precip_var = precip_cirrus.attrs["precip_var"]
precip_cirrus_dataarray = precip_cirrus[precip_var]

filename = os.path.join(
    root_path, "20241126", "NIMBUS", "T_PAAH22_C_EUOC_20241126010000.hdf"
)
precip_nimbus_rain_rate = pysteps.io.import_opera_hdf5(filename, qty="RATE")
precip_var = precip_nimbus_rain_rate.attrs["precip_var"]
precip_nimbus_rain_rate_dataarray = precip_nimbus_rain_rate[precip_var]

filename = os.path.join(
    root_path, "20241126", "NIMBUS", "T_PASH22_C_EUOC_20241126010000.hdf"
)
precip_nimbus_rain_accum = pysteps.io.import_opera_hdf5(filename, qty="ACRR")
precip_var = precip_nimbus_rain_accum.attrs["precip_var"]
precip_nimbus_rain_accum_dataarray = precip_nimbus_rain_accum[precip_var]


def test_io_import_opera_hdf5_odyssey_shape():
    """Test the importer OPERA HDF5."""
    assert precip_odyssey_dataarray.shape == (2200, 1900)


def test_io_import_opera_hdf5_cirrus_shape():
    """Test the importer OPERA HDF5."""
    assert precip_cirrus_dataarray.shape == (4400, 3800)


def test_io_import_opera_hdf5_nimbus_rain_rate_shape():
    """Test the importer OPERA HDF5."""
    assert precip_nimbus_rain_rate_dataarray.shape == (2200, 1900)


def test_io_import_opera_hdf5_nimbus_rain_accum_shape():
    """Test the importer OPERA HDF5."""
    assert precip_nimbus_rain_accum_dataarray.shape == (2200, 1900)


# test_metadata: list of (variable,expected, tolerance) tuples
expected_proj = (
    "+proj=laea +lat_0=55.0 +lon_0=10.0 "
    "+x_0=1950000.0 "
    "+y_0=-2100000.0 "
    "+units=m +ellps=WGS84"
)

# list of (variable,expected,tolerance) tuples
test_odyssey_attrs = [
    (precip_odyssey.attrs["projection"], expected_proj, None),
    (float(precip_odyssey.lon.isel(x=0, y=0).values), -10.4268122372, 1e-10),
    (float(precip_odyssey.lat.isel(x=0, y=0).values), 31.7575305091, 1e-10),
    (float(precip_odyssey.lon.isel(x=-1, y=-1).values), 57.7778944303, 1e-10),
    (float(precip_odyssey.lat.isel(x=-1, y=-1).values), 67.6204665961, 1e-10),
    (precip_odyssey.x.isel(x=0).values, 999.999583891, 1e-6),
    (precip_odyssey.y.isel(y=0).values, -4399000.00106, 1e-10),
    (precip_odyssey.x.isel(x=-1).values, 3799000.00043, 1e-10),
    (precip_odyssey.y.isel(y=-1).values, -1000.00042627, 1e-6),
    (precip_odyssey.x.attrs["stepsize"], 2000.0, 1e-10),
    (precip_odyssey.y.attrs["stepsize"], 2000.0, 1e-10),
    (precip_odyssey.x.attrs["units"], "m", None),
    (precip_odyssey.y.attrs["units"], "m", None),
    (precip_odyssey_dataarray.attrs["accutime"], 15.0, 1e-10),
    # ("yorigin", "upper", None),
    (precip_odyssey_dataarray.attrs["units"], "mm/h", None),
    (precip_odyssey.attrs["institution"], "Odyssey datacentre", None),
    (precip_odyssey_dataarray.attrs["transform"], None, None),
    (precip_odyssey_dataarray.attrs["zerovalue"], 0.0, 1e-6),
    (precip_odyssey_dataarray.attrs["threshold"], 0.01, 1e-6),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_odyssey_attrs)
def test_io_import_opera_hdf5_odyssey_dataset_attrs(variable, expected, tolerance):
    """Test the importer OPERA HDF5."""
    smart_assert(variable, expected, tolerance)


# list of (variable,expected,tolerance) tuples
test_cirrus_attrs = [
    (precip_cirrus.attrs["projection"], expected_proj, None),
    (float(precip_cirrus.lon.isel(x=0, y=0).values), -10.4306947565, 1e-10),
    (float(precip_cirrus.lat.isel(x=0, y=0).values), 31.7518730135, 1e-10),
    (float(precip_cirrus.lon.isel(x=-1, y=-1).values), 57.7949292793, 1e-10),
    (float(precip_cirrus.lat.isel(x=-1, y=-1).values), 67.6207527344, 1e-10),
    (precip_cirrus.x.isel(x=0).values, 499.99972864, 1e-6),
    (precip_cirrus.y.isel(y=0).values, -4399500.00116976, 1e-10),
    (precip_cirrus.x.isel(x=-1).values, 3799500.00025612, 1e-10),
    (precip_cirrus.y.isel(y=-1).values, -500.00008774, 1e-6),
    (precip_cirrus.x.attrs["stepsize"], 1000.0, 1e-10),
    (precip_cirrus.y.attrs["stepsize"], 1000.0, 1e-10),
    (precip_cirrus.x.attrs["units"], "m", None),
    (precip_cirrus.y.attrs["units"], "m", None),
    (precip_cirrus_dataarray.attrs["accutime"], 15.0, 1e-10),
    # ("yorigin", "upper", None),
    (precip_cirrus_dataarray.attrs["units"], "dBZ", None),
    (precip_cirrus.attrs["institution"], "Odyssey datacentre", None),
    (precip_cirrus_dataarray.attrs["transform"], "dB", None),
    (precip_cirrus_dataarray.attrs["zerovalue"], -32.0, 1e-10),
    (precip_cirrus_dataarray.attrs["threshold"], -31.5, 1e-10),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_cirrus_attrs)
def test_io_import_opera_hdf5_cirrus_dataset_attrs(variable, expected, tolerance):
    """Test OPERA HDF5 importer: max. reflectivity composites from CIRRUS."""
    smart_assert(variable, expected, tolerance)


# list of (variable,expected,tolerance) tuples
test_nimbus_rain_rate_attrs = [
    (precip_nimbus_rain_rate.attrs["projection"], expected_proj, None),
    (float(precip_nimbus_rain_rate.lon.isel(x=0, y=0).values), -10.4268354001, 1e-10),
    (float(precip_nimbus_rain_rate.lat.isel(x=0, y=0).values), 31.7575151437, 1e-10),
    (float(precip_nimbus_rain_rate.lon.isel(x=-1, y=-1).values), 57.7778328845, 1e-10),
    (float(precip_nimbus_rain_rate.lat.isel(x=-1, y=-1).values), 67.6204748496, 1e-10),
    (precip_nimbus_rain_rate.x.isel(x=0).values, 997.46972871, 1e-6),
    (precip_nimbus_rain_rate.y.isel(y=0).values, -4399001.03116964, 1e-10),
    (precip_nimbus_rain_rate.x.isel(x=-1).values, 3798997.47025605, 1e-10),
    (precip_nimbus_rain_rate.y.isel(y=-1).values, -1001.03008786, 1e-6),
    (precip_nimbus_rain_rate.x.attrs["stepsize"], 2000.0, 1e-10),
    (precip_nimbus_rain_rate.y.attrs["stepsize"], 2000.0, 1e-10),
    (precip_nimbus_rain_rate.x.attrs["units"], "m", None),
    (precip_nimbus_rain_rate.y.attrs["units"], "m", None),
    (precip_nimbus_rain_rate_dataarray.attrs["accutime"], 15.0, 1e-10),
    (precip_nimbus_rain_rate_dataarray.attrs["units"], "mm/h", None),
    (precip_nimbus_rain_rate.attrs["institution"], "Odyssey datacentre", None),
    (precip_nimbus_rain_rate_dataarray.attrs["transform"], None, None),
    (precip_nimbus_rain_rate_dataarray.attrs["zerovalue"], 0.0, 1e-10),
    (precip_nimbus_rain_rate_dataarray.attrs["threshold"], 0.01, 1e-10),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_nimbus_rain_rate_attrs)
def test_io_import_opera_hdf5_nimbus_rain_rate_dataset_attrs(
    variable, expected, tolerance
):
    """Test OPERA HDF5 importer: rain rate composites from NIMBUS."""
    smart_assert(variable, expected, tolerance)


# list of (variable,expected,tolerance) tuples
test_nimbus_rain_accum_attrs = [
    (precip_nimbus_rain_accum.attrs["projection"], expected_proj, None),
    (float(precip_nimbus_rain_accum.lon.isel(x=0, y=0).values), -10.4268354001, 1e-10),
    (float(precip_nimbus_rain_accum.lat.isel(x=0, y=0).values), 31.7575151437, 1e-10),
    (float(precip_nimbus_rain_accum.lon.isel(x=-1, y=-1).values), 57.7778328845, 1e-10),
    (float(precip_nimbus_rain_accum.lat.isel(x=-1, y=-1).values), 67.6204748496, 1e-10),
    (precip_nimbus_rain_accum.x.isel(x=0).values, 997.46972871, 1e-6),
    (precip_nimbus_rain_accum.y.isel(y=0).values, -4399001.03116964, 1e-10),
    (precip_nimbus_rain_accum.x.isel(x=-1).values, 3798997.47025605, 1e-10),
    (precip_nimbus_rain_accum.y.isel(y=-1).values, -1001.03008786, 1e-6),
    (precip_nimbus_rain_accum.x.attrs["stepsize"], 2000.0, 1e-10),
    (precip_nimbus_rain_accum.y.attrs["stepsize"], 2000.0, 1e-10),
    (precip_nimbus_rain_accum.x.attrs["units"], "m", None),
    (precip_nimbus_rain_accum.y.attrs["units"], "m", None),
    (precip_nimbus_rain_accum_dataarray.attrs["accutime"], 15.0, 1e-10),
    (precip_nimbus_rain_accum_dataarray.attrs["units"], "mm", None),
    (precip_nimbus_rain_accum.attrs["institution"], "Odyssey datacentre", None),
    (precip_nimbus_rain_accum_dataarray.attrs["transform"], None, None),
    (precip_nimbus_rain_accum_dataarray.attrs["zerovalue"], 0.0, 1e-10),
    (precip_nimbus_rain_accum_dataarray.attrs["threshold"], 0.01, 1e-10),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_nimbus_rain_accum_attrs)
def test_io_import_opera_hdf5_nimbus_rain_accum_dataset_attrs(
    variable, expected, tolerance
):
    """Test OPERA HDF5 importer: rain accumulation composites from NIMBUS."""
    smart_assert(variable, expected, tolerance)
