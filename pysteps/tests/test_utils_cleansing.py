# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pysteps.utils import cleansing


def test_decluster_empty():
    """
    Decluster an empty input"""

    X = np.empty((0, 2))
    V = np.empty((0, 2))
    X_dec, V_dec = cleansing.decluster(X, V, 20, 1)

    assert X_dec.ndim == 2
    assert V_dec.ndim == 2
    assert X_dec.shape[0] == 0
    assert V_dec.shape[0] == 0


def test_decluster_single():
    """decluster a single vector"""

    X = np.array([[0.0, 0.0]])
    V = np.array([[1.0, 1.0]])
    X_dec, V_dec = cleansing.decluster(X, V, 20, 1)

    assert X_dec.ndim == 2
    assert V_dec.ndim == 2
    assert np.all(X_dec == X)
    assert np.all(X_dec == X)

    X_dec, V_dec = cleansing.decluster(X, V, 20, 2)
    assert X_dec.ndim == 2
    assert V_dec.ndim == 2
    assert X_dec.shape[0] == 0
    assert V_dec.shape[0] == 0


def test_decluster():
    """decluster an input with duplicated vectors"""

    X = np.tile(np.random.randint(100, size=(10, 2)), (3, 1))
    V = np.tile(np.random.randint(100, size=(10, 2)), (3, 1))

    X_dec, V_dec = cleansing.decluster(X, V, 20, 1)

    assert X_dec.ndim == 2
    assert V_dec.ndim == 2
    assert X_dec.shape[0] <= V_dec.shape[0]
    assert X_dec.shape[0] <= 10
    assert V_dec.shape[0] <= 10

    X_dec, V_dec = cleansing.decluster(X, V, 100, 1)
    assert X_dec.ndim == 2
    assert V_dec.ndim == 2
    assert X_dec.shape[0] == 1
    assert V_dec.shape[0] == 1
    assert np.all(X_dec == np.median(X, axis=0))
    assert np.all(V_dec == np.median(V, axis=0))


def test_detect_outlier_constant():
    """Test that a constant input produces no outliers and that warnings are raised"""

    V = np.zeros(20)  # this will trigger a runtime warning
    with pytest.warns(RuntimeWarning):
        outliers = cleansing.detect_outliers(V, 1)
    assert outliers.size == V.shape[0]
    assert outliers.sum() == 0

    V = np.zeros((20, 3))  # this will trigger a singular matrix warning
    with pytest.warns(UserWarning):
        outliers = cleansing.detect_outliers(V, 1)
    assert outliers.size == V.shape[0]
    assert outliers.sum() == 0

    V = np.zeros((20, 3))  # this will trigger a singular matrix warning
    X = np.random.randint(100, size=(20, 3))
    with pytest.warns(UserWarning):
        outliers = cleansing.detect_outliers(V, 1, coord=X, k=10)
    assert outliers.size == V.shape[0]
    assert outliers.sum() == 0


def test_detect_outlier_univariate_global():
    """Test that"""

    # test input with no outliers at all
    V = np.random.randn(200)
    V = V[np.abs(V) < 1.5]
    outliers = cleansing.detect_outliers(V, 4)
    assert outliers.sum() == 0

    # test a postive outlier
    V[-1] = 10
    outliers = cleansing.detect_outliers(V, 4)
    assert outliers.sum() == 1

    # test a negative outlier
    V[-1] = -10
    outliers = cleansing.detect_outliers(V, 4)
    assert outliers.sum() == 1


def test_detect_outlier_multivariate_global():
    """Test that"""

    # test input with no outliers at all
    V = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=200)
    V = V[np.all(np.abs(V) < 1.5, axis=1), :]
    V = V[np.abs(V[:, 1] - V[:, 0]) < 0.5, :]
    outliers = cleansing.detect_outliers(V, 4)
    assert outliers.sum() == 0

    # test postive outliers
    V[-2, :] = (10, 0)
    V[-1, :] = (3, -3)
    outliers = cleansing.detect_outliers(V, 4)
    assert outliers.sum() == 2

    # test negative outliers
    V[-2] = (-10, 0)
    V[-1] = (-3, 3)
    outliers = cleansing.detect_outliers(V, 4)
    assert outliers.sum() == 2


def test_detect_outlier_univariate_local():
    """Test that"""

    # test input with no outliers at all
    V = np.random.randn(200)
    X = np.random.randint(100, size=200)
    X = X[np.abs(V) < 1.5]
    V = V[np.abs(V) < 1.5]
    outliers = cleansing.detect_outliers(V, 4, coord=X, k=50)
    assert outliers.sum() == 0

    # test a postive outlier
    V[-1] = 10
    outliers = cleansing.detect_outliers(V, 4, coord=X, k=50)
    assert outliers.sum() == 1

    # test a negative outlier
    V[-1] = -10
    outliers = cleansing.detect_outliers(V, 4, coord=X, k=50)
    assert outliers.sum() == 1


def test_detect_outlier_multivariate_local():
    """Test that"""

    # test input with no outliers at all
    V = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=200)
    X = np.random.randint(100, size=(200, 3))
    idx = np.abs(V[:, 1] - V[:, 0]) < 1
    idx = idx & np.all(np.abs(V) < 1.5, axis=1)
    X = X[idx, :]
    V = V[idx, :]
    outliers = cleansing.detect_outliers(V, 4, coord=X, k=50)
    assert outliers.sum() == 0

    # test postive outliers
    V[-2, :] = (10, 0)
    V[-1, :] = (3, -3)
    outliers = cleansing.detect_outliers(V, 4, coord=X, k=50)
    assert outliers.sum() == 2

    # test negative outliers
    V[-2] = (-10, 0)
    V[-1] = (-3, 3)
    outliers = cleansing.detect_outliers(V, 4, coord=X, k=50)
    assert outliers.sum() == 2
