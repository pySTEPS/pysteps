# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pysteps
from pysteps import nowcasts
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade.bandpass_filters import filter_uniform
from pysteps.cascade.decomposition import decomposition_fft, recompose_fft
from pysteps.tests.helpers import smart_assert


def test_decompose_recompose():
    """Tests cascade decomposition."""

    pytest.importorskip("netCDF4")

    root_path = pysteps.rcparams.data_sources["bom"]["root_path"]
    rel_path = os.path.join("prcp-cscn", "2", "2018", "06", "16")
    filename = os.path.join(root_path, rel_path, "2_20180616_120000.prcp-cscn.nc")
    precip_dataset = pysteps.io.import_bom_rf3(filename)

    # Convert to rain rate from mm
    precip_dataset = pysteps.utils.to_rainrate(precip_dataset)

    # Log-transform the data
    precip_dataset = pysteps.utils.dB_transform(
        precip_dataset, threshold=0.1, zerovalue=-15.0
    )
    precip_var = precip_dataset.attrs["precip_var"]
    precip = precip_dataset[precip_var].values

    # Set Nans as the fill value
    precip[~np.isfinite(precip)] = precip_dataset[precip_var].attrs["zerovalue"]

    # Set number of cascade levels
    num_cascade_levels = 9

    # Construct the Gaussian bandpass filters
    _filter = filter_gaussian(precip.shape, num_cascade_levels)

    # Decompose precip
    decomp = decomposition_fft(precip, _filter)

    # Recomposed precip from decomp
    recomposed = recompose_fft(decomp)
    # Assert
    assert_array_almost_equal(recomposed.squeeze(), precip)


test_metadata_filter = [
    ("central_freqs", None, None),
    ("central_wavenumbers", None, None),
]


@pytest.mark.parametrize("variable, expected, tolerance", test_metadata_filter)
def test_filter_uniform(variable, expected, tolerance):
    _filter = filter_uniform((8, 8), 1)
    smart_assert(_filter[variable], expected, tolerance)


def test_filter_uniform_weights_1d():
    _filter = filter_uniform((8, 8), 1)
    assert_array_almost_equal(_filter["weights_1d"], np.ones((1, 5)))


def test_filter_uniform_weights_2d():
    _filter = filter_uniform((8, 8), 1)
    assert_array_almost_equal(_filter["weights_2d"], np.ones((1, 8, 5)))
