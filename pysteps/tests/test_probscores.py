# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.postprocessing.ensemblestats import excprob
from pysteps.tests.helpers import get_precipitation_fields
from pysteps.verification import probscores

precip = get_precipitation_fields(num_next_files=10, return_raw=True)

# CRPS
test_data = [(precip[:10], precip[-1], 0.01470871)]


@pytest.mark.parametrize("X_f, X_o, expected", test_data)
def test_CRPS(X_f, X_o, expected):
    """Test the CRPS."""
    assert_array_almost_equal(probscores.CRPS(X_f, X_o), expected)


# reldiag
test_data = [(precip[:10], precip[-1], 1.0, 10, 10, 3.38751492)]


@pytest.mark.parametrize("X_f, X_o, X_min, n_bins, min_count, expected", test_data)
def test_reldiag_sum(X_f, X_o, X_min, n_bins, min_count, expected):
    """Test the reldiag."""
    P_f = excprob(X_f, X_min, ignore_nan=False)
    assert_array_almost_equal(
        np.sum(probscores.reldiag(P_f, X_o, X_min, n_bins, min_count)[1]), expected
    )


# ROC_curve
test_data = [(precip[:10], precip[-1], 1.0, 10, True, 0.79557329)]


@pytest.mark.parametrize(
    "X_f, X_o, X_min, n_prob_thrs, compute_area, expected", test_data
)
def test_ROC_curve_area(X_f, X_o, X_min, n_prob_thrs, compute_area, expected):
    """Test the ROC_curve."""
    P_f = excprob(X_f, X_min, ignore_nan=False)
    assert_array_almost_equal(
        probscores.ROC_curve(P_f, X_o, X_min, n_prob_thrs, compute_area)[2], expected
    )
