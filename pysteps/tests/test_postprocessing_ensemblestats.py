# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.postprocessing.ensemblestats import excprob, mean, banddepth

# CREATE DATASETS TO TEST

a = np.arange(9, dtype=float).reshape(3, 3)
b = np.tile(a, (4, 1, 1))
b1 = b.copy()
b1[3] = np.nan
a1 = a.copy()
a1[:] = np.nan
a2 = a.copy()
a2[0, :] = np.nan

#  test data
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
    """
    Test ensemblestats mean."""
    assert_array_almost_equal(mean(X, ignore_nan, X_thr), expected)


#  test exceptions
test_exceptions = [(0), (None), (a[0, :]), (np.tile(a, (4, 1, 1, 1)))]


@pytest.mark.parametrize("X", test_exceptions)
def test_exceptions_mean(X):
    with pytest.raises(Exception):
        mean(X)


#  test data
b2 = b.copy()
b2[2, 2, 2] = np.nan

test_data = [
    (b, 2.0, False, np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])),
    (b2, 2.0, False, np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, np.nan]])),
    (b2, 2.0, True, np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])),
]


@pytest.mark.parametrize("X, X_thr, ignore_nan, expected", test_data)
def test_ensemblestats_excprob(X, X_thr, ignore_nan, expected):
    """Test ensemblestats excprob."""
    assert_array_almost_equal(excprob(X, X_thr, ignore_nan), expected)


#  test exceptions
test_exceptions = [(0), (None), (a[0, :]), (a)]


@pytest.mark.parametrize("X", test_exceptions)
def test_exceptions_excprob(X):
    with pytest.raises(Exception):
        excprob(X, 2.0)


#  test data
b3 = np.tile(a, (5, 1, 1)) + 1
b3 *= np.arange(1, 6)[:, None, None]
b3[2, 2, 2] = np.nan

test_data = [
    (b3, 1, True, np.array([0.0, 0.75, 1.0, 0.75, 0.0])),
    (b3, None, False, np.array([0.4, 0.7, 0.8, 0.7, 0.4])),
]


@pytest.mark.parametrize("X, thr, norm, expected", test_data)
def test_ensemblestats_banddepth(X, thr, norm, expected):
    """Test ensemblestats banddepth."""
    assert_array_almost_equal(banddepth(X, thr, norm), expected)
