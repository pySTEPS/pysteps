# -*- coding: utf-8 -*-

import pytest

from pysteps.tests.helpers import smart_assert, get_precipitation_fields

precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=True,
    source="saf",
    log_transform=False,
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]

test_extent_crri = [
    (None, (-3300000.0, 3297000.0, 2514000.0, 5568000.0), (1, 1019, 2200), None),
    (
        (-1980000.0, 1977000.0, 2514000.0, 4818000.0),
        (-1977000.0, 1974000.0, 2517000.0, 4815000.0),
        (1, 767, 1318),
        None,
    ),
]


@pytest.mark.parametrize(
    "extent, expected_extent, expected_shape, tolerance", test_extent_crri
)
def test_io_import_saf_crri_extent(extent, expected_extent, expected_shape, tolerance):
    """Test the importer SAF CRRI."""

    precip_dataset_reduced_domain = get_precipitation_fields(
        num_prev_files=0,
        num_next_files=0,
        return_raw=True,
        source="saf",
        log_transform=False,
        extent=extent,
    )
    precip_var = precip_dataset_reduced_domain.attrs["precip_var"]
    precip_dataarray_reduced_domain = precip_dataset_reduced_domain[precip_var]
    x_min = float(precip_dataset_reduced_domain.x.isel(x=0).values)
    x_max = float(precip_dataset_reduced_domain.x.isel(x=-1).values)
    y_min = float(precip_dataset_reduced_domain.y.isel(y=0).values)
    y_max = float(precip_dataset_reduced_domain.y.isel(y=-1).values)
    extent_out = (x_min, x_max, y_min, y_max)
    smart_assert(extent_out, expected_extent, tolerance)
    smart_assert(precip_dataarray_reduced_domain.shape, expected_shape, tolerance)


expected_proj = (
    "+proj=geos +a=6378137.000000 +b=6356752.300000 "
    "+lon_0=0.000000 +h=35785863.000000"
)

test_geodata_crri = [
    (precip_dataset.attrs["projection"], expected_proj, None),
    (precip_dataset.x.isel(x=0).values, -3300000.0, 1e-6),
    (precip_dataset.y.isel(y=0).values, 2514000.0, 1e-6),
    (precip_dataset.x.isel(x=-1).values, 3297000.0, 1e-6),
    (precip_dataset.y.isel(y=-1).values, 5568000.0, 1e-9),
    (precip_dataset.x.attrs["stepsize"], 3000.0, 1e-10),
    (precip_dataset.y.attrs["stepsize"], 3000.0, 1e-10),
    (precip_dataset.x.attrs["units"], "m", None),
    (precip_dataset.y.attrs["units"], "m", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_geodata_crri)
def test_io_import_saf_crri_geodata(variable, expected, tolerance):
    smart_assert(variable, expected, tolerance)


# list of (variable,expected,tolerance) tuples
test_attrs = [
    (precip_dataset.attrs["projection"], expected_proj, None),
    (
        precip_dataset.attrs["institution"],
        "Agencia Estatal de Meteorología (AEMET)",
        None,
    ),
    (precip_dataarray.attrs["accutime"], None, None),
    (precip_dataarray.attrs["units"], "mm/h", None),
    (precip_dataarray.attrs["zerovalue"], 0.0, 1e-6),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs)
def test_io_import_saf_crri_attrs(variable, expected, tolerance):
    """Test the importer SAF CRRI."""
    smart_assert(variable, expected, tolerance)
