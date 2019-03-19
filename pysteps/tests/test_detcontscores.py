# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pysteps.verification import det_cont_fcst
from numpy.testing import assert_array_almost_equal

# CREATE A DATASET TO MATCH
# EXAMPLES IN
# http://www.cawcr.gov.au/projects/verification/

obs_data = np.asarray([5, 10, 9, 15, 22, 13, 17, 17, 19, 23.0])
fct_data = np.asarray([-1, 8, 12, 13, 18, 10, 16, 19, 23, 24.0])

test_data = [
    # test None as score
    ([0.], [0.], None, []),
    # test unknown score
    ([1., 3.], [2., 5.], ('lolo'), []),
    # test unknown score and None
    ([1., 3.], [2., 5.], ('lolo', None), []),
    # Mean Absolute Error (Additive) as string
    (fct_data, obs_data, 'MAE_ADD', [2.8]),
    # Mean Absolute Error (Additive)
    (fct_data, obs_data, ('MAE_ADD'), [2.8]),
    # Mean Absolute Error (Multiplicative)
    (fct_data[1:], obs_data[1:], ('MAE_MULT'), [0.734082]),
    # Root Mean Square Error
    (fct_data, obs_data, ('RMSE_ADD'), [3.162278]),
    # Root Mean Square Error (Multiplicative)
    (fct_data[1:], obs_data[1:], ('RMSE_MULT'), [0.813752]),
    # Pearson correlation
    (fct_data, obs_data, ('CORR_P'), [0.914363]),
    # Spearman correlation
    (fct_data, obs_data, ('CORR_S'), [0.917937]),
    # Mean Error (Additive)
    (fct_data, obs_data, ('ME_ADD'), [-0.80]),
    # Mean Error (Multiplicative)
    (fct_data[1:], obs_data[1:], ('ME_MULT'), [-0.124068]),
    # Beta
    (fct_data, obs_data, ('Beta'), [0.70528]),
    # reduction of variance
    (fct_data, obs_data, ('rv_add'), [0.668874]),
    # reduction of variance (multiplicative)
    (fct_data[1:], obs_data[1:], ('rv_mult'), [0.630398]),
    # scatter
    (fct_data[1:], obs_data[1:], ('scatter'), [0.836588]),
    ]


@pytest.mark.parametrize("pred, obs, scores, expected", test_data)
def test_det_cont_fcst(pred, obs, scores, expected):
    """Test the det_cont_fcst."""
    assert_array_almost_equal(det_cont_fcst(pred, obs, scores), expected)
