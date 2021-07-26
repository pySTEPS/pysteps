# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pysteps.utils import get_method


interp_methods = (
    "idwinterp2d",
    "rbfinterp2d",
)


@pytest.mark.parametrize("interp_method", interp_methods)
def test_interp_univariate(interp_method):
    coord = np.random.rand(10, 2)
    input_array = np.random.rand(10)
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 10)

    interp = get_method(interp_method)
    output = interp(coord, input_array, xgrid, ygrid)

    assert isinstance(output, np.ndarray)
    assert output.ndim == 2
    assert output.shape == (ygrid.size, xgrid.size)
    assert np.isfinite(output).all()


@pytest.mark.parametrize("interp_method", interp_methods)
def test_interp_multivariate(interp_method):
    coord = np.random.rand(10, 2)
    input_array = np.random.rand(10, 2)
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 10)

    interp = get_method(interp_method)
    output = interp(coord, input_array, xgrid, ygrid)

    assert isinstance(output, np.ndarray)
    assert output.ndim == 3
    assert output.shape[0] == 2
    assert output.shape[1:] == (ygrid.size, xgrid.size)
    assert np.isfinite(output).all()


@pytest.mark.parametrize("interp_method", interp_methods)
def test_wrong_inputs(interp_method):
    coord = np.random.rand(10, 2)
    input_array = np.random.rand(10, 2)
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 10)

    interp = get_method(interp_method)

    # nan in the input values
    with pytest.raises(ValueError):
        input_with_nans = input_array.copy()
        input_with_nans[0, 0] = np.nan
        interp(coord, input_with_nans, xgrid, ygrid)

    # nan in the input coordinates
    with pytest.raises(ValueError):
        coord_with_nans = coord.copy()
        coord_with_nans[0, 0] = np.nan
        interp(coord_with_nans, input_array, xgrid, ygrid)

    # too many dimensions in the input values
    with pytest.raises(ValueError):
        interp(coord, np.random.rand(10, 2, 1), xgrid, ygrid)

    # wrong dimension size in the input coordinates
    with pytest.raises(ValueError):
        interp(np.random.rand(10, 1), input_array, xgrid, ygrid)

    # wrong number of dimensions in the input coordinates
    with pytest.raises(ValueError):
        interp(np.random.rand(10, 2, 1), input_array, xgrid, ygrid)

    # wrong number of coordinates
    with pytest.raises(ValueError):
        interp(np.random.rand(9, 2), input_array, xgrid, ygrid)


@pytest.mark.parametrize("interp_method", interp_methods)
def test_one_sample_input(interp_method):
    coord = np.random.rand(1, 2)
    input_array = np.array([1, 2])[None, :]
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 10)

    interp = get_method(interp_method)

    # one sample returns uniform grids
    output = interp(coord, input_array, xgrid, ygrid)
    assert np.isfinite(output).all()
    assert output[0, ...].max() == output[0, ...].min() == 1
    assert output[1, ...].max() == output[1, ...].min() == 2


@pytest.mark.parametrize("interp_method", interp_methods)
def test_uniform_input(interp_method):
    coord = np.random.rand(10, 2)
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 10)

    interp = get_method(interp_method)

    # same value across all variables
    input_array = np.ones((10, 2))
    output = interp(coord, input_array, xgrid, ygrid)
    assert np.isfinite(output).all()
    assert output.max() == output.min() == input_array.ravel()[0]

    # # same value in one variable only
    # input_array = np.vstack((np.ones(10), np.random.rand(10))).T
    # output = interp(coord, input_array, xgrid, ygrid)
    # assert output[0,].max() == output[0,].min() == input_array[0,0]


def test_idwinterp2d_k1():
    coord = np.random.rand(10, 2)
    input_array = np.random.rand(10, 2)
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 10)

    interp = get_method("idwinterp2d")
    output = interp(coord, input_array, xgrid, ygrid, k=1)

    assert isinstance(output, np.ndarray)
    assert output.ndim == 3
    assert output.shape[0] == 2
    assert output.shape[1:] == (ygrid.size, xgrid.size)
    assert np.isfinite(output).all()


def test_idwinterp2d_kNone():
    coord = np.random.rand(10, 2)
    input_array = np.random.rand(10, 2)
    xgrid, ygrid = np.linspace(0, 1, 10), np.linspace(0, 1, 10)

    interp = get_method("idwinterp2d")
    output = interp(coord, input_array, xgrid, ygrid, k=None)

    assert isinstance(output, np.ndarray)
    assert output.ndim == 3
    assert output.shape[0] == 2
    assert output.shape[1:] == (ygrid.size, xgrid.size)
    assert np.isfinite(output).all()
