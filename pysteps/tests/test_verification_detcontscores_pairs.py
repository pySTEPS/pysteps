"""Continuous scores use the same valid pairs for all moments."""

import numpy as np
import pytest

from pysteps.verification import detcontscores


@pytest.mark.parametrize("conditioning", [None, "single", "double"])
@pytest.mark.parametrize("missing", ["pred", "obs", "both"])
def test_scores_ignore_unpaired_values(conditioning, missing):
    pred = np.array([0.0, 1.0, 4.0, 3.0, 6.0, 100.0, 2.0])
    obs = np.array([1.0, 0.0, 3.0, 5.0, 7.0, 1.0, 100.0])
    if missing in ("pred", "both"):
        pred[-1] = np.nan
    if missing in ("obs", "both"):
        obs[-2] = np.nan
    pred_before, obs_before = pred.copy(), obs.copy()
    valid = np.isfinite(pred) & np.isfinite(obs)
    if conditioning == "single":
        valid &= (pred > 0) | (obs > 0)
    elif conditioning == "double":
        valid &= (pred > 0) & (obs > 0)
    p, o = pred[valid], obs[valid]
    covariance = np.mean((p - p.mean()) * (o - o.mean()))
    expected = {
        "corr_p": np.corrcoef(p, o)[0, 1],
        "beta1": covariance / p.var(),
        "beta2": covariance / o.var(),
        "RV": 1 - np.mean((p - o) ** 2) / o.var(),
    }
    actual = detcontscores.det_cont_fct(
        pred, obs, scores=list(expected), conditioning=conditioning
    )
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key])
    np.testing.assert_equal(pred, pred_before)
    np.testing.assert_equal(obs, obs_before)


@pytest.mark.parametrize("axis", [0, 1, (0, 1), -1])
def test_accumulate_unpaired_values(axis):
    pred = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0]])
    obs = np.array([[2.0, np.nan, 100.0], [4.0, 100.0, 5.0]])
    err = detcontscores.det_cont_fct_init(axis=axis)
    # Missing coverage must not affect a later fully observed batch.
    detcontscores.det_cont_fct_accum(err, pred, obs)
    detcontscores.det_cont_fct_accum(err, np.ones((2, 3)), np.full((2, 3), 2.0))
    valid = np.isfinite(pred) & np.isfinite(obs)
    reduction = () if axis == -1 else axis
    count = valid.sum(axis=reduction) + np.ones_like(pred).sum(axis=reduction)
    obs_sum = np.where(valid, obs, 0).sum(axis=reduction)
    obs_sum += np.full_like(obs, 2.0).sum(axis=reduction)
    pred_sum = np.where(valid, pred, 0).sum(axis=reduction)
    pred_sum += np.ones_like(pred).sum(axis=reduction)
    np.testing.assert_equal(err["n"], count)
    np.testing.assert_allclose(err["mobs"], obs_sum / count)
    np.testing.assert_allclose(err["mpred"], pred_sum / count)
