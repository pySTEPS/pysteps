# -*- coding: utf-8 -*-

import pytest
import os
import pysteps
import numpy as np
from pysteps.tests.helpers import smart_assert
from numpy.testing import assert_array_almost_equal
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade.bandpass_filters import filter_uniform
from pysteps.cascade.decomposition import decomposition_fft


def test_decompose_recompose():
    """Tests cascade decomposition."""
    root_path = pysteps.rcparams.data_sources["bom"]["root_path"]
    rel_path = os.path.join("prcp-cscn", "2", "2018", "06", "16")
    filename = os.path.join(root_path, rel_path,
                            "2_20180616_120000.prcp-cscn.nc")
    R, _, metadata = pysteps.io.import_bom_rf3(filename)
    # Convert to rain rate from mm
    R, metadata = pysteps.utils.to_rainrate(R, metadata)
    # Log-transform the data
    R, metadata = pysteps.utils.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)
    # Set Nans as the fill value
    R[~np.isfinite(R)] = metadata["zerovalue"]
    # Set number of cascade levels
    num_cascade_levels = 9
    # Construct the Gaussian bandpass filters
    filter = filter_gaussian(R.shape, num_cascade_levels)
    # Decompose R
    R_d =[]
    decomp = decomposition_fft(R, filter)
    R_d.append(decomp)
    # Recomposed R from R_d
    R_c, mu_c, sigma_c=pysteps.nowcasts.utils.stack_cascades(R_d, num_cascade_levels)
    recomposed=pysteps.nowcasts.utils.recompose_cascade(R_c, mu_c, sigma_c)
    # Assert
    assert_array_almost_equal(recomposed.squeeze(), R)



test_metadata_filter = [
    ('central_freqs', None, None),
    ('central_wavenumbers', None, None),
    ]

@pytest.mark.parametrize("variable, expected, tolerance", test_metadata_filter)
def test_filter_uniform(variable, expected, tolerance):
    filter = filter_uniform((8,8), 1)    
    smart_assert(filter[variable], expected, tolerance)

def test_filter_uniform_weights_1d():
    filter = filter_uniform((8,8), 1)    
    assert_array_almost_equal(filter['weights_1d'], np.ones((1,5)))

def test_filter_uniform_weights_2d():
    filter = filter_uniform((8,8), 1)    
    assert_array_almost_equal(filter['weights_2d'], np.ones((1,8,5)))

