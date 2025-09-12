# -*- coding: utf-8 -*-

import pytest

from pysteps.tests.helpers import smart_assert, get_precipitation_fields

precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=True,
    metadata=True,
    source="mch",
    log_transform=False,
    importer_kwargs=dict(qty="AQC"),
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]


def test_io_import_mch_gif_shape():
    """Test the importer MCH GIF."""
    assert precip_dataarray.shape == (1, 640, 710)


expected_proj = (
    "+proj=somerc  +lon_0=7.43958333333333 "
    "+lat_0=46.9524055555556 +k_0=1 "
    "+x_0=600000 +y_0=200000 +ellps=bessel "
    "+towgs84=674.374,15.056,405.346,0,0,0,0 "
    "+units=m +no_defs"
)

# list of (variable,expected,tolerance) tuples
test_attrs = [
    (precip_dataset.attrs["projection"], expected_proj, None),
    (precip_dataset.attrs["institution"], "MeteoSwiss", None),
    (precip_dataarray.attrs["accutime"], 5.0, 1e-10),
    (precip_dataset.time.attrs["stepsize"], 300, 1e-10),
    (precip_dataarray.attrs["units"], "mm", None),
    (precip_dataarray.attrs["transform"], None, None),
    (precip_dataarray.attrs["zerovalue"], 0.0, 1e-6),
    (precip_dataarray.attrs["threshold"], 0.0008258007600496956, 1e-19),
    (precip_dataarray.attrs["zr_a"], 316.0, None),
    (precip_dataarray.attrs["zr_b"], 1.5, None),
    (precip_dataset.x.isel(x=0).values, 255500.0, 1e-10),
    (precip_dataset.y.isel(y=0).values, -159500.0, 1e-10),
    (precip_dataset.x.isel(x=-1).values, 964500.0, 1e-10),
    (precip_dataset.y.isel(y=-1).values, 479500.0, 1e-10),
    (precip_dataset.x.attrs["stepsize"], 1000.0, 0.1),
    (precip_dataset.y.attrs["stepsize"], 1000.0, 0.1),
    (precip_dataset.x.attrs["units"], "m", None),
    (precip_dataset.y.attrs["units"], "m", None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_attrs)
def test_io_import_mch_gif_dataset_attrs(variable, expected, tolerance):
    """Test the importer MCH GIF."""
    smart_assert(variable, expected, tolerance)
