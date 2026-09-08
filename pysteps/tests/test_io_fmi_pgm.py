# -*- coding: utf-8 -*-
import pytest

from pysteps.tests.helpers import smart_assert, get_precipitation_fields

precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=True,
    source="fmi",
    log_transform=False,
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]


def test_io_import_fmi_pgm_shape():
    """Test the importer FMI PGM."""
    assert precip_dataarray.shape == (1, 1226, 760)


expected_proj = (
    "+proj=stere  +lon_0=25E +lat_0=90N "
    "+lat_ts=60 +a=6371288 +x_0=380886.310 "
    "+y_0=3395677.920 +no_defs"
)


test_attrs = [
    (precip_dataset.attrs["projection"], expected_proj, None),
    (precip_dataset.attrs["institution"], "Finnish Meteorological Institute", None),
    (precip_dataarray.attrs["accutime"], 5.0, 1e-10),
    (precip_dataarray.attrs["units"], "dBZ", None),
    (precip_dataarray.attrs["transform"], "dB", None),
    (precip_dataarray.attrs["zerovalue"], -32.0, 1e-6),
    (precip_dataarray.attrs["threshold"], -31.5, 1e-6),
    (precip_dataarray.attrs["zr_a"], 223.0, 1e-6),
    (precip_dataarray.attrs["zr_b"], 1.53, 1e-6),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs)
def test_io_import_mch_gif_dataset_attrs(variable, expected, tolerance):
    """Test the importer FMI PMG."""
    smart_assert(variable, expected, tolerance)


# test_geodata: list of (variable,expected,tolerance) tuples
test_geodata = [
    (precip_dataset.x.isel(x=0).values, 499.84200883, 1e-10),
    (precip_dataset.y.isel(y=0).values, 499.8240261, 1e-10),
    (precip_dataset.x.isel(x=-1).values, 759252.4482492, 1e-10),
    (precip_dataset.y.isel(y=-1).values, 1225044.84459724, 1e-10),
    (precip_dataset.x.attrs["stepsize"], 999.674053, 1e-8),
    (precip_dataset.y.attrs["stepsize"], 999.62859, 1e-8),
    (precip_dataset.x.attrs["units"], "m", None),
    (precip_dataset.y.attrs["units"], "m", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_geodata)
def test_io_import_fmi_pgm_geodata(variable, expected, tolerance):
    smart_assert(variable, expected, tolerance)
