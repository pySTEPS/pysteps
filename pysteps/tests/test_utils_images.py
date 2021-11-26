import numpy as np
import xarray as xr

from pysteps.utils import images


def test_morph_opening_nans():
    """"""
    input_array = np.zeros((20, 20))
    input_array[12:14, 12:14] = np.nan
    input_array = xr.DataArray(input_array, dims=("y", "x"))
    output_array = images.morph_opening(input_array, 0, 3)
    assert isinstance(output_array, xr.DataArray)
    assert output_array.sizes == input_array.sizes
    assert np.isnan(input_array).sum() == 4
    assert np.isnan(output_array).sum() == 4

def test_morph_opening_2d():
    """"""
    input_array = np.zeros((20, 20))
    input_array[1:11, 1:11] = 1
    input_array[12:14, 12:14] = 1
    input_array = xr.DataArray(input_array, dims=("y", "x"))
    output_array = images.morph_opening(input_array, 0, np.ones((3, 3)))
    assert isinstance(output_array, xr.DataArray)
    assert output_array.sizes == input_array.sizes
    assert int(input_array.sum()) == 104
    assert int(output_array.sum()) == 100

def test_morph_opening_3d():
    """"""
    input_array = np.zeros((3, 20, 20))
    input_array[0, 1:11, 1:11] = 1
    input_array[:, 12:14, 12:14] = 1
    input_array = xr.DataArray(input_array, dims=("t", "y", "x"))
    output_array = images.morph_opening(input_array, 0, np.ones((3, 3)))
    assert isinstance(output_array, xr.DataArray)
    assert output_array.sizes == input_array.sizes
    assert int(input_array.sum()) == 112
    assert int(output_array.sum()) == 100

