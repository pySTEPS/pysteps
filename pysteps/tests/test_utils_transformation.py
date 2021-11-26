# -*- coding: utf-8 -*-

import numpy as np
import pytest
import xarray as xr


xr.set_options(keep_attrs=True)


data_array = xr.DataArray(
    np.random.randn(10, 10),
    coords={"x": np.arange(10), "y": np.arange(10)},
    dims=("y", "x"),
    attrs={"unit": "dummy"},
)
data_array = data_array.where(data_array > -1)
data_array = xr.where(data_array < 0, 0, data_array)


test_args = [(1.0, 0.01), (0.0, 0.01)]


@pytest.mark.parametrize("boxcox_lambda, offset", test_args)
def test_boxcox_transform(boxcox_lambda, offset):

    data_transformed = data_array.pysteps.boxcox_transform(boxcox_lambda, offset)
    assert data_transformed.attrs.get("boxcox_lambda") == boxcox_lambda
    assert data_transformed.attrs.get("offset") == offset
    assert data_transformed.attrs.get("transform") == "BoxCox"
    assert data_transformed.attrs.get("unit") == "dummy"

    data_back = data_transformed.pysteps.boxcox_transform(inverse=True)
    assert data_back.attrs.get("transform") is None
    assert data_back.attrs.get("unit") == "dummy"
    assert "offset" not in data_back.attrs
    xr.testing.assert_allclose(data_back, data_array)


def test_db_transform():
    offset = 0.01
    data_transformed = data_array.pysteps.db_transform(offset)
    assert data_transformed.attrs.get("offset") == offset
    assert data_transformed.attrs.get("transform") == "dB"

    data_back = data_transformed.pysteps.db_transform(inverse=True)
    assert data_back.attrs.get("transform") is None
    assert data_back.attrs.get("unit") == "dummy"
    assert "offset" not in data_back.attrs
    xr.testing.assert_allclose(data_back, data_array)


def test_nq_transform():
    nq_a = 0.0
    data_transformed = data_array.pysteps.nq_transform(nq_a)
    assert data_transformed.attrs.get("transform") == "NQ"

    data_back = data_transformed.pysteps.nq_transform(template=data_array, inverse=True)
    assert data_back.attrs.get("transform") is None
    assert data_back.attrs.get("unit") == "dummy"
    xr.testing.assert_equal(np.isnan(data_back), np.isnan(data_array))
    # we do not expect an exact match since the extremes are lost
    xr.testing.assert_allclose(data_back, data_array, rtol=1, atol=0.1)


def test_sqrt_transform():
    data_transformed = data_array.pysteps.sqrt_transform()
    assert data_transformed.attrs.get("transform") == "sqrt"

    data_back = data_transformed.pysteps.sqrt_transform(inverse=True)
    assert data_back.attrs.get("transform") is None
    assert data_back.attrs.get("unit") == "dummy"
    xr.testing.assert_allclose(data_back, data_array)
