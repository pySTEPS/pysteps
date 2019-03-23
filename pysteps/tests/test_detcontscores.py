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
    # Mean Absolute Error as string
    (fct_data, obs_data, 'MAE', [2.8]),
    # Mean Absolute Error
    (fct_data, obs_data, ('MAE'), [2.8]),
    # Root Mean Square Error
    (fct_data, obs_data, ('RMSE'), [3.162278]),
    # Pearson correlation
    (fct_data, obs_data, ('CORR_P'), [0.914363]),
    # Spearman correlation
    (fct_data, obs_data, ('CORR_S'), [0.917937]),
    # Mean Error
    (fct_data, obs_data, ('ME'), [-0.80]),
    # Beta
    (fct_data, obs_data, ('Beta'), [0.70528]),
    # reduction of variance
    (fct_data, obs_data, ('rv'), [0.668874]),
    # debiased RMSE
    (fct_data, obs_data, ('DRMSE'), [0.309934]),
    # scatter
    (fct_data, obs_data, ('scatter'), [0.769423])
    ]

@pytest.mark.parametrize("pred, obs, scores, expected", test_data)
def test_det_cont_fcst(pred, obs, scores, expected):
    """Test the det_cont_fcst."""
    assert_array_almost_equal(det_cont_fcst(pred, obs, scores), expected)
