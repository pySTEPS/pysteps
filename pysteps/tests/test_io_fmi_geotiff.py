# -*- coding: utf-8 -*-

import pytest

from pysteps.tests.helpers import get_precipitation_fields, smart_assert

precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    source="fmi_geotiff",
    log_transform=False,
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]


def test_io_import_fmi_geotiff_shape():
    """Test the shape of the read file."""
    assert precip_dataarray.shape == (1, 7316, 4963)


expected_proj = (
    "+proj=utm +zone=35 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
)

# test_geodata: list of (variable,expected,tolerance) tuples
test_geodata = [
    (precip_dataset.attrs["projection"], expected_proj, None),
    (precip_dataset.attrs["institution"], "Finnish Meteorological Institute", None),
    (precip_dataset.x.isel(x=0).values, -196468.00230479, 1e-10),
    (precip_dataset.y.isel(y=-1).values, 6255454.70581264, 1e-10),
    (precip_dataset.x.isel(x=-1).values, 1044051.93934604, 1e-10),
    (precip_dataset.y.isel(y=0).values, 8084306.99826718, 1e-10),
    (precip_dataset.x.attrs["stepsize"], 250.0040188736061566, 1e-10),
    (precip_dataset.y.attrs["stepsize"], -250.0139839309011904, 1e-10),
    (precip_dataset.x.attrs["units"], "m", None),
    (precip_dataset.y.attrs["units"], "m", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_geodata)
def test_io_import_fmi_pgm_geodata(variable, expected, tolerance):
    """Test the GeoTIFF and metadata reading."""
    smart_assert(variable, expected, tolerance)
