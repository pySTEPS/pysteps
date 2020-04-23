# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.verification import ensscores

precip = get_precipitation_fields(num_next_files=10, return_raw=True)
np.random.seed(42)

# rankhist
test_data = [
    (precip[:10], precip[-1], None, True, 11),
    (precip[:10], precip[-1], None, False, 11),
]


@pytest.mark.parametrize("X_f, X_o, X_min, normalize, expected", test_data)
def test_rankhist_size(X_f, X_o, X_min, normalize, expected):
    """Test the rankhist."""
    assert_array_almost_equal(
        ensscores.rankhist(X_f, X_o, X_min, normalize).size, expected
    )


# ensemble_skill
test_data = [
    (
        precip[:10],
        precip[-1],
        "RMSE",
        {"axis": None, "conditioning": "single"},
        0.26054151,
    ),
    (precip[:10], precip[-1], "CSI", {"thr": 1.0, "axis": None}, 0.22017924),
    (precip[:10], precip[-1], "FSS", {"thr": 1.0, "scale": 10}, 0.63239752),
]


@pytest.mark.parametrize("X_f, X_o, metric, kwargs, expected", test_data)
def test_ensemble_skill(X_f, X_o, metric, kwargs, expected):
    """Test the ensemble_skill."""
    assert_array_almost_equal(
        ensscores.ensemble_skill(X_f, X_o, metric, **kwargs), expected
    )


# ensemble_spread
test_data = [
    (precip, "RMSE", {"axis": None, "conditioning": "single"}, 0.22635757),
    (precip, "CSI", {"thr": 1.0, "axis": None}, 0.25218158),
    (precip, "FSS", {"thr": 1.0, "scale": 10}, 0.70235667),
]


@pytest.mark.parametrize("X_f, metric, kwargs, expected", test_data)
def test_ensemble_spread(X_f, metric, kwargs, expected):
    """Test the ensemble_spread."""
    assert_array_almost_equal(
        ensscores.ensemble_spread(X_f, metric, **kwargs), expected
    )
