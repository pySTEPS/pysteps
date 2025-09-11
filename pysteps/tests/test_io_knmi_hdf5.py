# -*- coding: utf-8 -*-

import pytest

from pysteps.tests.helpers import smart_assert, get_precipitation_fields

precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=True,
    metadata=True,
    source="knmi",
    log_transform=False,
    importer_kwargs=dict(qty="ACRR"),
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]


def test_io_import_knmi_hdf5_shape():
    """Test the importer KNMI HDF5."""
    assert precip_dataarray.shape == (1, 765, 700)


# test_metadata: list of (variable,expected, tolerance) tuples

expected_proj = (
    "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378137 +b=6356752 +x_0=0 +y_0=0"
)

# list of (variable,expected,tolerance) tuples
test_attrs = [
    (precip_dataset.attrs["projection"], expected_proj, None),
    (precip_dataset.x.isel(x=0).values, 499.98876045, 1e-10),
    (precip_dataset.y.isel(y=0).values, -4414538.12181262, 1e-10),
    (precip_dataset.x.isel(x=-1).values, 699484.27587271, 1e-10),
    (precip_dataset.y.isel(y=-1).values, -3650450.41764577, 1e-10),
    (precip_dataset.x.attrs["stepsize"], 1000.0, 1e-10),
    (precip_dataset.y.attrs["stepsize"], 1000.0, 1e-10),
    (precip_dataarray.attrs["accutime"], 5.0, 1e-10),
    (precip_dataset.time.attrs["stepsize"], 5.0, 1e-10),
    (precip_dataarray.attrs["units"], "mm/h", None),
    (precip_dataset.x.attrs["units"], "m", None),
    (precip_dataset.y.attrs["units"], "m", None),
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
