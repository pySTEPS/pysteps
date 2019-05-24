# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pysteps.utils import arrays, conversion
from numpy.testing import assert_array_equal, assert_array_almost_equal

# compute_centred_coord_array
test_data = [
    (2, 2, [np.array([[-1, 0]]).T, np.array([[-1, 0]])]),
    (3, 3, [np.array([[-1, 0, 1]]).T, np.array([[-1, 0, 1]])]),
    (3, 2, [np.array([[-1, 0, 1]]).T, np.array([[-1, 0]])]),
    (2, 3, [np.array([[-1, 0]]).T, np.array([[-1, 0, 1]])]),
    ]

@pytest.mark.parametrize("M, N, expected", test_data)
def test_compute_centred_coord_array(M, N, expected):
    """Test the compute_centred_coord_array."""
    assert_array_equal(arrays.compute_centred_coord_array(M, N)[0], expected[0])
    assert_array_equal(arrays.compute_centred_coord_array(M, N)[1], expected[1])
    
# to_rainrate
test_data = [

    (np.array([1]), {"accutime":5, "transform":None, "unit":"mm/h", "threshold":0, "zerovalue":0},
    np.array([1])),
    (np.array([1]), {"accutime":5, "transform":None, "unit":"mm", "threshold":0, "zerovalue":0},
    np.array([12])),
    
    (np.array([1]), {"accutime":5, "transform":"dB", "unit":"mm/h", "threshold":0, "zerovalue":0},
    np.array([1.25892541])),
    (np.array([1]), {"accutime":5, "transform":"dB", "unit":"mm", "threshold":0, "zerovalue":0},
    np.array([15.10710494])),
    (np.array([1]), {"accutime":5, "transform":"dB", "unit":"dBZ", "threshold":0, "zerovalue":0},
    np.array([0.02513093])),
    
    (np.array([1]), {"accutime":5, "transform":"log", "unit":"mm/h", "threshold":0, "zerovalue":0},
    np.array([1])),
    (np.array([1.]), {"accutime":5, "transform":"log", "unit":"mm", "threshold":0, "zerovalue":0},
    np.array([12.])),
    
    (np.array([1]), {"accutime":5, "transform":"sqrt", "unit":"mm/h", "threshold":0, "zerovalue":0},
    np.array([1])),
    (np.array([1.]), {"accutime":5, "transform":"sqrt", "unit":"mm", "threshold":0, "zerovalue":0},
    np.array([12.])),
    
    ]

@pytest.mark.parametrize("R, metadata, expected", test_data)
def test_to_rainrate(R, metadata, expected):
    """Test the compute_centred_coord_array."""
    assert_array_almost_equal(conversion.to_rainrate(R, metadata)[0], expected)