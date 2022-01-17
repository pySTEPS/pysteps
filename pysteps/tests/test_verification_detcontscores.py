# -*- coding: utf-8 -*-

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.verification import det_cont_fct

# CREATE A DATASET TO MATCH
# EXAMPLES IN
# http://www.cawcr.gov.au/projects/verification/

obs_data = np.asarray(
    [7, 10, 9, 15, 22, 13, 17, 17, 19, 23.0, 0, 10, 0, 15, 0, 13, 0, 17, 0, 0.0]
)
fct_data = np.asarray(
    [1, 8, 12, 13, 18, 10, 16, 19, 23, 24.0, 0, 0, 12, 0, 0, 0, 16, 0, 0, 0.0]
)

test_data = [
    # test None as score
    ([0.0], [0.0], None, None, None, []),
    # test unknown score
    ([1.0, 3.0], [2.0, 5.0], ("lolo"), None, None, []),
    # test unknown score and None
    ([1.0, 3.0], [2.0, 5.0], ("lolo", None), None, None, []),
    # Mean Error as string
    (fct_data, obs_data, "ME", None, None, [-1.75]),
    # Mean Error
    (fct_data, obs_data, ("ME"), None, None, [-1.75]),
    # Mean Error single conditional
    (fct_data, obs_data, ("ME"), None, "single", [-2.1875]),
    # Mean Error double conditional
    (fct_data, obs_data, ("ME"), None, "double", [-0.8]),
    # Mean Absolute Error
    (fct_data, obs_data, ("MAE"), None, None, [5.55]),
    # Mean Square Error
    (fct_data, obs_data, ("MSE"), None, None, [64.15]),
    # Normalized Mean Square Error
    (fct_data, obs_data, ("NMSE"), None, None, [0.113711]),
    # Root Mean Square Error
    (fct_data, obs_data, ("RMSE"), None, None, [8.009370]),
    # Beta1
    (fct_data, obs_data, ("beta1"), None, None, [0.498200]),
    # Beta2
    (fct_data, obs_data, ("beta2"), None, None, [0.591673]),
    # reduction of variance
    (fct_data, obs_data, ("RV"), None, None, [-0.054622]),
    # debiased RMSE
    (fct_data, obs_data, ("DRMSE"), None, None, [7.815849]),
    # Pearson correlation
    (fct_data, obs_data, ("corr_p"), None, None, [0.542929]),
    # Spearman correlation
    (fct_data, obs_data, ("corr_s"), None, None, [0.565251]),
    # Spearman correlation single conditional
    (fct_data, obs_data, ("corr_s"), None, "single", [0.467913]),
    # Spearman correlation double conditional
    (fct_data, obs_data, ("corr_s"), None, "double", [0.917937]),
    # scatter
    (fct_data, obs_data, ("scatter"), None, None, [0.808023]),
    # Mean Error along axis 0 as tuple
    (
        np.tile(fct_data, (2, 1)).T,
        np.tile(obs_data, (2, 1)).T,
        "ME",
        (0,),
        None,
        [[-1.75, -1.75]],
    ),
    # Mean Error along axis 0
    (
        np.tile(fct_data, (2, 1)).T,
        np.tile(obs_data, (2, 1)).T,
        "ME",
        0,
        None,
        [[-1.75, -1.75]],
    ),
    # Mean Error along axis 1
    (
        np.tile(fct_data, (2, 1)).T,
        np.tile(obs_data, (2, 1)).T,
        "ME",
        1,
        None,
        [[-6, -2, 3, -2, -4, -3, -1, 2, 4, 1, 0, -10, 12, -15, 0, -13, 16, -17, 0, 0]],
    ),
    # Mean Error along axis (1,2)
    (
        np.tile(fct_data, (2, 1)).T,
        np.tile(obs_data, (2, 1)).T,
        "ME",
        (0, 1),
        None,
        [-1.75],
    ),
    # Mean Error along axis (2,1)
    (
        np.tile(fct_data, (2, 1)).T,
        np.tile(obs_data, (2, 1)).T,
        "ME",
        (1, 0),
        None,
        [-1.75],
    ),
    # scatter along axis 0 as tuple
    (
        np.tile(fct_data, (2, 1)).T,
        np.tile(obs_data, (2, 1)).T,
        "scatter",
        (0,),
        None,
        [[0.808023, 0.808023]],
    ),
    # scatter along axis 0
    (
        np.tile(fct_data, (2, 1)).T,
        np.tile(obs_data, (2, 1)).T,
        "scatter",
        0,
        None,
        [[0.808023, 0.808023]],
    ),
    # scatter along axis (1,2)
    (
        np.tile(fct_data, (2, 1)).T,
        np.tile(obs_data, (2, 1)).T,
        "scatter",
        (0, 1),
        None,
        [0.804806],
    ),
    # scatter along axis (2,1)
    (
        np.tile(fct_data, (2, 1)).T,
        np.tile(obs_data, (2, 1)).T,
        "scatter",
        (1, 0),
        None,
        [0.804806],
    ),
]


@pytest.mark.parametrize("pred, obs, scores, axis, conditioning, expected", test_data)
def test_det_cont_fct(pred, obs, scores, axis, conditioning, expected):
    """Test the det_cont_fct."""
    assert_array_almost_equal(
        list(det_cont_fct(pred, obs, scores, axis, conditioning).values()), expected
    )
