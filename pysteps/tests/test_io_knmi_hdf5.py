# -*- coding: utf-8 -*-

import pytest

from pysteps.tests.helpers import get_precipitation_fields, smart_assert

precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=True,
    source="knmi",
    log_transform=False,
    qty="ACRR",
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]


def test_io_import_knmi_hdf5_shape():
    """Test the importer KNMI HDF5."""
    assert precip_dataarray.shape == (1, 765, 700)


# test_metadata: list of (variable,expected, tolerance) tuples

expected_proj = "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0"

# list of (variable,expected,tolerance) tuples
test_attrs = [
    (precip_dataset.attrs["projection"], expected_proj, None),
    (precip_dataset.x.isel(x=0).values, 0.5, 1e-10),
    (precip_dataset.y.isel(y=-1).values, -4414.5, 1e-10),
    (precip_dataset.x.isel(x=-1).values, 699.5, 1e-10),
    (precip_dataset.y.isel(y=0).values, -3650.5, 1e-10),
    (precip_dataset.x.attrs["stepsize"], 1.0, 1e-10),
    (precip_dataset.y.attrs["stepsize"], -1.0, 1e-10),
    (precip_dataarray.attrs["accutime"], 5.0, 1e-10),
    (precip_dataset.time.attrs["stepsize"], 300, 1e-10),
    (precip_dataarray.attrs["units"], "mm", None),
    (precip_dataset.x.attrs["units"], "km", None),
    (precip_dataset.y.attrs["units"], "km", None),
    (
        precip_dataset.attrs["institution"],
        "KNMI - Royal Netherlands Meteorological Institute",
        None,
    ),
    (precip_dataarray.attrs["zerovalue"], 0.0, 1e-6),
    (precip_dataarray.attrs["threshold"], 0.01, 1e-6),
    (precip_dataarray.attrs["zr_a"], 200.0, None),
    (precip_dataarray.attrs["zr_b"], 1.6, None),
]


@pytest.mark.parametrize("variable,expected,tolerance", test_attrs)
def test_io_import_knmi_hdf5_metadata(variable, expected, tolerance):
    """Test the importer KNMI HDF5."""
    smart_assert(variable, expected, tolerance)
