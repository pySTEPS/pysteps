# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import pytest

from pysteps.utils import get_method


interp_methods = (
    "idwinterp2d",
    "rbfinterp2d",
)


@pytest.mark.parametrize("interp_method", interp_methods)
def test_interp_univariate(interp_method):
    interp = get_method(interp_method)

    # inputs
    coord = np.random.rand(2, 10)
    coord[0, 0] = np.nan  # robust to missing coordinate ?
    input_array = np.random.rand(10)
    input_array[1] = np.nan  # robust to missing values ?
    sparse_data = xr.DataArray(
        input_array.astype(np.float32),
        dims="sample",
        coords={"x": ("sample", coord[0]), "y": ("sample", coord[1])},
        attrs=dict(units="units"),
    )
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 8)

    # input a DataArray
    output = interp(sparse_data, xgrid, ygrid)
    assert isinstance(output, xr.DataArray)
    assert output.ndim == 2
    assert output.shape == (ygrid.size, xgrid.size)
    assert np.isfinite(output).all()
    assert output.dtype == sparse_data.dtype
    assert output.attrs == sparse_data.attrs
    assert np.allclose(output.x, xgrid)
    assert np.allclose(output.y, ygrid)

    # input a Dataset
    sparse_data = sparse_data.to_dataset(name="var1")
    output = interp(sparse_data, xgrid, ygrid)
    assert isinstance(output, xr.Dataset)
    assert output.var1.ndim == 2
    assert output.var1.shape == (ygrid.size, xgrid.size)
    assert output.var1.dtype == sparse_data.var1.dtype
    assert np.isfinite(output.var1).all()
    assert output.attrs == sparse_data.attrs
    assert np.allclose(output.x, xgrid)
    assert np.allclose(output.y, ygrid)


@pytest.mark.parametrize("interp_method", interp_methods)
def test_interp_multivariate(interp_method):
    interp = get_method(interp_method)

    # inputs
    coord = np.random.rand(2, 10)
    coord[0, 0] = np.nan  # robust to missing coordinate ?
    input_array = np.random.rand(10, 2)
    input_array[1, 1] = np.nan  # robust to missing values ?
    sparse_data = xr.DataArray(
        input_array.astype(np.float32),
        dims=("sample", "variable"),
        coords={
            "x": ("sample", coord[0]),
            "y": ("sample", coord[1]),
            "variable": ("variable", ["var1", "var2"]),
        },
        attrs=dict(units="units"),
    )
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 8)

    # input a DataArray
    output = interp(sparse_data, xgrid, ygrid)
    assert isinstance(output, xr.DataArray)
    assert output.ndim == 3
    assert output.sizes["variable"] == 2
    assert output.isel(variable=0).shape == (ygrid.size, xgrid.size)
    assert np.isfinite(output).all()
    assert output.dtype == sparse_data.dtype
    assert output.attrs == sparse_data.attrs
    assert np.allclose(output.x, xgrid)
    assert np.allclose(output.y, ygrid)

    # input a Dataset
    sparse_data = sparse_data.to_dataset(dim="variable")
    output = interp(sparse_data, xgrid, ygrid)
    assert isinstance(output, xr.Dataset)
    assert "var1" in sparse_data.data_vars
    assert "var2" in sparse_data.data_vars
    assert output.var1.ndim == 2
    assert output.var1.shape == (ygrid.size, xgrid.size)
    assert output.var1.dtype == sparse_data.var1.dtype
    assert np.isfinite(output.var1).all()
    assert output.attrs == sparse_data.attrs
    assert np.allclose(output.x, xgrid)
    assert np.allclose(output.y, ygrid)


@pytest.mark.parametrize("interp_method", interp_methods)
def test_wrong_inputs(interp_method):
    interp = get_method(interp_method)

    # inputs
    coord = np.random.rand(2, 10)
    input_array = np.random.rand(10, 2)
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 8)
    sparse_data = xr.DataArray(
        input_array.astype(np.float32),
        dims=("sample", "variable"),
        coords={
            "x": ("sample", coord[0]),
            "y": ("sample", coord[1]),
            "variable": ("variable", ["var1", "var2"]),
        },
        attrs=dict(units="units"),
    )

    # not an xarray object
    with pytest.raises(ValueError):
        interp(input_array, xgrid, ygrid)

    # missing 'sample' dimension
    with pytest.raises(ValueError):
        interp(sparse_data.rename({"sample": "dummy"}), xgrid, ygrid)

    # too many dimensions in the input values
    with pytest.raises(ValueError):
        interp(sparse_data.expand_dims("dummy"), xgrid, ygrid)

    # missing x or y coordinates
    with pytest.raises(ValueError):
        interp(sparse_data.drop_vars("x"), xgrid, ygrid)
        interp(sparse_data.drop_vars("y"), xgrid, ygrid)

    # missing values in target coordinates
    with pytest.raises(ValueError):
        xgridn, ygridn = xgrid.copy(), ygrid.copy()
        xgridn[0] = np.nan
        ygridn[0] = np.nan
        interp(sparse_data, xgridn, ygrid)
        interp(sparse_data, xgrid, ygridn)


@pytest.mark.parametrize("interp_method", interp_methods)
def test_one_sample_input(interp_method):
    interp = get_method(interp_method)

    # inputs
    coord = np.random.rand(2, 1)
    input_array = np.array([1, 2])[None, :]
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 8)
    sparse_data = xr.DataArray(
        input_array.astype(np.float32),
        dims=("sample", "variable"),
        coords={
            "x": ("sample", coord[0]),
            "y": ("sample", coord[1]),
            "variable": ("variable", ["var1", "var2"]),
        },
        attrs=dict(units="units"),
    )

    # one sample returns uniform grids
    output = interp(sparse_data, xgrid, ygrid)
    assert isinstance(output, xr.DataArray)
    assert output.ndim == 3
    assert output.sizes["variable"] == 2
    assert output.isel(variable=0).shape == (ygrid.size, xgrid.size)
    assert np.isfinite(output).all()
    assert output.dtype == sparse_data.dtype
    assert output.attrs == sparse_data.attrs
    assert np.allclose(output.x, xgrid)
    assert np.allclose(output.y, ygrid)
    assert output[0, ...].max() == output[0, ...].min() == 1
    assert output[1, ...].max() == output[1, ...].min() == 2


@pytest.mark.parametrize("interp_method", interp_methods)
def test_uniform_input(interp_method):
    interp = get_method(interp_method)

    coord = np.random.rand(2, 10)
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 8)

    # same value across all variables
    input_array = np.ones((10, 2))
    sparse_data = xr.DataArray(
        input_array.astype(np.float32),
        dims=("sample", "variable"),
        coords={
            "x": ("sample", coord[0]),
            "y": ("sample", coord[1]),
            "variable": ("variable", ["var1", "var2"]),
        },
        attrs=dict(units="units"),
    )
    output = interp(sparse_data, xgrid, ygrid)
    assert np.isfinite(output).all()
    assert output.max() == output.min() == input_array.ravel()[0]

    # TODO: test same value in one variable only


def test_idwinterp2d_k1():
    interp = get_method("idwinterp2d")

    # inputs
    coord = np.random.rand(2, 10)
    input_array = np.random.rand(10, 2)
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 10)
    sparse_data = xr.DataArray(
        input_array.astype(np.float32),
        dims=("sample", "variable"),
        coords={
            "x": ("sample", coord[0]),
            "y": ("sample", coord[1]),
            "variable": ("variable", ["var1", "var2"]),
        },
        attrs=dict(units="units"),
    )

    output = interp(sparse_data, xgrid, ygrid, k=1)
    assert isinstance(output, xr.DataArray)
    assert output.ndim == 3
    assert output.shape[0] == 2
    assert output.shape[1:] == (ygrid.size, xgrid.size)
    assert np.isfinite(output).all()


def test_idwinterp2d_kNone():
    interp = get_method("idwinterp2d")

    # inputs
    coord = np.random.rand(2, 10)
    input_array = np.random.rand(10, 2)
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 10)
    sparse_data = xr.DataArray(
        input_array.astype(np.float32),
        dims=("sample", "variable"),
        coords={
            "x": ("sample", coord[0]),
            "y": ("sample", coord[1]),
            "variable": ("variable", ["var1", "var2"]),
        },
        attrs=dict(units="units"),
    )

    output = interp(sparse_data, xgrid, ygrid, k=None)
    assert isinstance(output, xr.DataArray)
    assert output.ndim == 3
    assert output.shape[0] == 2
    assert output.shape[1:] == (ygrid.size, xgrid.size)
    assert np.isfinite(output).all()
