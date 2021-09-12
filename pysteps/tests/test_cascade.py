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
from pysteps.tests.helpers import smart_assert, get_precipitation_fields


PRECIP = get_precipitation_fields(
    source="bom",
    convert_to="mm/h",
    transform_to="db",
    coarsen=2,
    filled=True,
)


def test_decompose_recompose():
    """Tests cascade decomposition."""

    # Set number of cascade levels
    num_cascade_levels = 9

    # Construct the Gaussian bandpass filters
    _filter = filter_gaussian(PRECIP.shape, num_cascade_levels)

    # Decompose precip
    decomp = decomposition_fft(PRECIP, _filter)

    assert all(key in decomp for key in ("cascade_levels", "means", "stds"))
    assert decomp["cascade_levels"].shape[0] == num_cascade_levels
    assert decomp["cascade_levels"].shape[1:] == PRECIP.shape
    assert isinstance(decomp["means"], list)
    assert isinstance(decomp["stds"], list)
    assert len(decomp["means"]) == num_cascade_levels
    assert len(decomp["stds"]) == num_cascade_levels

    # Recomposed precip from decomp
    recomposed = recompose_fft(decomp)

    assert_array_almost_equal(recomposed, PRECIP)


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
