# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pysteps.verification import det_cat_fct
from numpy.testing import assert_array_almost_equal

# CREATE A LARGE DATASET TO MATCH
# EXAMPLES IN
# http://www.cawcr.gov.au/projects/verification/

fct_hits = 1.0*np.ones(82)
obs_hits = 1.0*np.ones(82)
fct_fa = 1.0*np.ones(38)
obs_fa = 1.0*np.zeros(38)
fct_misses = 1.0*np.zeros(23)
obs_misses = 1.0*np.ones(23)
fct_cr = 1.0*np.zeros(222)
obs_cr = 1.0*np.zeros(222)
obs_data = np.concatenate([obs_hits, obs_fa, obs_misses, obs_cr])
fct_data = np.concatenate([fct_hits, fct_fa, fct_misses, fct_cr])


test_data = [
    ([0.], [0.], 0., None, []),
    ([1., 3.], [2., 5.], 0., None, []),
    ([1., 3.], [2., 5.], 0., 'CSI', [1.0]),
    ([1., 3.], [2., 5.], 0., ('CSI', 'FAR'), [1.0, 0.0]),
    ([1., 3.], [2., 5.], 0., ('lolo',), []),
    ([1., 3.], [2., 5.], 0., ('CSI', None, 'FAR'), [1.0, 0.0]),
    ([1., 3.], [2., 5.], 1.0, ('CSI', None, 'FAR'), [0.5, 0.0]),
    ([1., 3.], [2., 5.], 1.0, ('lolo'), []),  # test unknown score
    (fct_data, obs_data, 0.0, ('ACC'), [0.83287671]),  # ACCURACY score
    (fct_data, obs_data, 0.0, ('BIAS'), [1.1428571]),  # BIAS score
    (fct_data, obs_data, 0.0, ('POD'), [0.7809524]),  # POD score
    (fct_data, obs_data, 0.0, ('FAR'), [0.316667]),  # FAR score
    # Probability of false detection (false alarm rate)
    (fct_data, obs_data, 0.0, ('FA'), [0.146154]),
    # CSI score
    (fct_data, obs_data, 0.0, ('CSI'), [0.573426]),
    # Heidke Skill Score
    (fct_data, obs_data, 0.0, ('HSS'), [0.608871]),
    # Hanssen-Kuipers Discriminant
    (fct_data, obs_data, 0.0, ('HK'), [0.6348]),
    # Gilbert Skill Score
    (fct_data, obs_data, 0.0, ('GSS'), [0.437682]),
    # Gilbert Skill Score
    (fct_data, obs_data, 0.0, ('ETS'), [0.437682]),
    # Symmetric extremal dependence index
    (fct_data, obs_data, 0.0, ('SEDI'), [0.789308]),
    ]


@pytest.mark.parametrize("pred, obs, thr, scores, expected", test_data)
def test_det_cat_fct(pred, obs, thr, scores, expected):
    """Test the det_cat_fct."""
    assert_array_almost_equal(
            list(det_cat_fct(pred, obs, thr, scores).values()), expected)
