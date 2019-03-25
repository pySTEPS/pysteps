# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pysteps.postprocessing.ensemblestats import mean
from numpy.testing import assert_array_almost_equal

# CREATE DATASETS TO TEST

a = np.arange(9, dtype=float).reshape(3, 3)
b = np.tile(a, (4, 1, 1))
b1 = b.copy()
b1[3] = np.nan
a1 = a.copy()
a1[:] = np.nan
a2 = a.copy()
a2[0, :] = np.nan


test_data = [
    (a, False, None, a),
    (b, False, None, a),
    (b1, True, None, a),
    (b1, False, None, a1),
    (b, False, 0.0, a),
    (b, False, 3.0, a2),
    (b, True, 3.0, a2),
    (b1, True, 3.0, a2),
    ]


@pytest.mark.parametrize("X, ignore_nan, X_thr, expected", test_data)
def test_ensemblestats_mean(X, ignore_nan, X_thr, expected):
    """Test ensemblestats mean."""
    assert_array_almost_equal(mean(X, ignore_nan, X_thr), expected)


# test exceptions
test_exceptions = [(0), (None), (a[0, :]),
                   (np.tile(a, (4, 1, 1, 1))),
                   ]


@pytest.mark.parametrize("X", test_exceptions)
def test_exceptions_mean(X):
    with pytest.raises(Exception):
        mean(X)

