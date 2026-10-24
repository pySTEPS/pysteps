"""Merging reusable verification summaries must preserve their inputs."""

from copy import deepcopy

import numpy as np
import pytest

from pysteps.verification import detcontscores


@pytest.mark.parametrize("axis", [None, 0, (1, 2), -1])
def test_merge_preserves_inputs(axis):
    rng = np.random.default_rng(42)
    pred = rng.uniform(0, 10, size=(2, 2, 3, 4))
    obs = rng.uniform(0, 10, size=pred.shape)
    summaries = []
    expected = detcontscores.det_cont_fct_init(axis=axis)
    for p, o in zip(pred, obs):
        summary = detcontscores.det_cont_fct_init(axis=axis)
        detcontscores.det_cont_fct_accum(summary, p, o)
        detcontscores.det_cont_fct_accum(expected, p, o)
        summaries.append(summary)

    before = deepcopy(summaries)
    first, second = summaries
    merged = detcontscores.det_cont_fct_merge(first, second)
    for summary, original in zip(summaries, before):
        for key in original:
            np.testing.assert_equal(summary[key], original[key])

    # The same partial summaries may be reused for another aggregate.
    repeated = detcontscores.det_cont_fct_merge(first, second)
    reverse = detcontscores.det_cont_fct_merge(second, first)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            for result in (merged, repeated, reverse):
                np.testing.assert_allclose(result[key], value)
                assert not np.shares_memory(result[key], summaries[0][key])
                assert not np.shares_memory(result[key], summaries[1][key])

    # Updating the merged object must also leave the partial summaries alone.
    detcontscores.det_cont_fct_accum(merged, pred[0], obs[0])
    for summary, original in zip(summaries, before):
        for key in original:
            np.testing.assert_equal(summary[key], original[key])
