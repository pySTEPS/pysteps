# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pysteps.utils import arrays

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
