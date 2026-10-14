"""Calibrated generator source parity, distribution mapping, and selection tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pysteps.noise import parametric
from pysteps.postprocessing.probmatching import match_rainfall_distribution


def _parameters(rain_fraction=0.08):
    return dict(
        beta_1=-2.4, beta_2=-3.1, rain_fraction=rain_fraction, mean_db=6.0, stdev_db=3.5
    )


@pytest.fixture(scope="module")
def reference():
    root = Path(__file__).parent / "data/calibrated_generator_reference"
    metadata = json.loads(root.with_suffix(".json").read_text())
    with np.load(root.with_suffix(".npz")) as stored:
        arrays = {name: stored[name] for name in stored.files}
    return metadata, arrays


def test_full_field_parity_with_original_generator(reference):
    metadata, arrays = reference
    for index, case in enumerate(metadata["cases"]):
        field, diagnostics = parametric.generate_calibrated_field(
            case["shape"],
            case["parameters"],
            case["kmperpixel"],
            case["scale_break"],
            case["threshold"],
            seed=case["seed"],
            return_diagnostics=True,
            **case["settings"],
        )
        assert_allclose(field, arrays[f"calibrated_{index}"], atol=1e-10, rtol=1e-12)
        for name in ("attempts", "selected_attempt", "tolerance_achieved"):
            assert diagnostics[name] == case["diagnostics"][name]
        for name in ("estimated_beta_1", "estimated_beta_2", "distance"):
            assert diagnostics[name] == pytest.approx(
                case["diagnostics"][name], abs=3e-5
            )
        raw = parametric.generate_stochastic_field(
            case["shape"],
            case["parameters"],
            case["kmperpixel"],
            case["scale_break"],
            case["threshold"],
            seed=case["seed"],
        )
        assert_allclose(raw, arrays[f"uncalibrated_{index}"], atol=1e-10, rtol=1e-12)


def test_shared_mapping_matches_post_ar_source(reference):
    metadata, arrays = reference
    original = arrays["post_ar_input"].copy()
    result = match_rainfall_distribution(
        original, **metadata["post_ar_parameters"], min_rain_fraction=0.025
    )
    assert_allclose(result, arrays["post_ar_output"], atol=1e-12, rtol=1e-12)
    assert_array_equal(original, arrays["post_ar_input"])


@pytest.mark.parametrize("shape", [(64, 48), (48, 64), (63, 37)])
def test_filter_uses_half_power_exponents_and_physical_units(shape):
    spacing = 0.5
    beta_1, beta_2, wavelength = -2.0, -3.5, 10.0
    filt = parametric.power_law_filter(shape, beta_1, beta_2, spacing, wavelength)
    fy = np.fft.fftfreq(shape[0], spacing)
    fx = np.fft.fftfreq(shape[1], spacing)
    radius = np.hypot(fy[:, None], fx[None, :])
    reference = 1 / (max(shape) * spacing)
    for y, x in [(0, 1), (1, 0), (shape[0] // 2, shape[1] // 2)]:
        f = radius[y, x]
        expected_power = (
            (f / reference) ** beta_1
            if f < 1 / wavelength
            else ((1 / wavelength / reference) ** beta_1 * (f * wavelength) ** beta_2)
        )
        assert filt[y, x] ** 2 == pytest.approx(expected_power, rel=1e-13)
    assert filt[0, 0] == 1 and np.isfinite(filt).all()
    # Scaling pixel spacing and break together leaves dimensionless responses.
    assert_allclose(
        filt,
        parametric.power_law_filter(shape, beta_1, beta_2, spacing * 2, wavelength * 2),
    )
    assert_allclose(
        filt.T,
        parametric.power_law_filter(shape[::-1], beta_1, beta_2, spacing, wavelength),
    )


@pytest.mark.parametrize("threshold", [-2, 0, 2])
def test_mapping_upper_bin_edge_and_threshold_translation(threshold):
    data = np.arange(100, dtype=float).reshape(10, 10)
    result = match_rainfall_distribution(data, 0.3, 20, 1, threshold, nbins=100)
    assert_array_equal(result > threshold, data >= 70)
    base = match_rainfall_distribution(data, 0.3, 20, 1, 0, nbins=100)
    assert_allclose(result[data < 70], base[data < 70] + threshold)
    assert_allclose(result[data >= 70], base[data >= 70])
    assert_array_equal(data, np.arange(100).reshape(10, 10))


def test_marginal_quantile_bounds_and_approximate_statistics():
    parameters = dict(_parameters(0.3), mean_db=6, stdev_db=1)
    field = parametric.generate_stochastic_field(
        (128, 128), parameters, 1, 20, 0, seed=123
    )
    wet = field[field > 0]
    assert abs(wet.size / field.size - 0.3) < 0.02
    assert abs(wet.mean() - 6) < 0.1 and abs(wet.std() - 1) < 0.1
    from scipy.stats import norm

    assert wet.min() >= norm.ppf(0.001, loc=6, scale=1)
    assert wet.max() <= norm.ppf(0.999, loc=6, scale=1)
    assert field.std() != pytest.approx(1)  # Raw marginal field is not standardized.


def test_first_acceptable_candidate_and_weighted_distance(monkeypatch):
    estimates = iter([(-1.5, -2.2), (-2.3, -3), (-2.4, -3.1)])
    monkeypatch.setattr(parametric, "_estimate_beta", lambda *_: next(estimates))
    _, diagnostics = parametric.generate_calibrated_field(
        (32, 32), _parameters(0.1), 1, 20, 0, seed=4, return_diagnostics=True
    )
    assert diagnostics["attempts"] == diagnostics["selected_attempt"] == 2
    assert diagnostics["tolerance_achieved"]
    assert diagnostics["distance"] == pytest.approx(
        np.sqrt(0.1**2 / 0.067738 + 0.1**2 / 0.076405)
    )


def test_distance_tolerance_is_inclusive(monkeypatch):
    parameters = dict(_parameters(0.1), beta_1=-2, beta_2=-3)
    monkeypatch.setattr(parametric, "_estimate_beta", lambda *_: (-1, -3))
    _, diagnostics = parametric.generate_calibrated_field(
        (32, 32),
        parameters,
        1,
        20,
        0,
        seed=4,
        beta_variances=(1, 1),
        distance_threshold=1,
        return_diagnostics=True,
    )
    assert diagnostics["distance"] == 1 and diagnostics["tolerance_achieved"]
    assert diagnostics["attempts"] == 1


def test_best_candidate_after_attempt_limit(monkeypatch):
    estimates = iter([(-1, -2), (-2, -2.8), (-2.2, -2.9), (-1.8, -2.5), (-1.2, -2.1)])
    monkeypatch.setattr(parametric, "_estimate_beta", lambda *_: next(estimates))
    _, diagnostics = parametric.generate_calibrated_field(
        (32, 32),
        _parameters(0.05),
        1,
        20,
        0,
        seed=4,
        distance_threshold=0.01,
        return_diagnostics=True,
    )
    assert (
        not diagnostics["tolerance_achieved"]
        and diagnostics["status"] == "best_available"
    )
    assert diagnostics["attempts"] == 5 and diagnostics["selected_attempt"] == 3
    assert diagnostics["candidate_estimates"].shape == (5, 2)


def test_best_candidate_retains_first_tie(monkeypatch):
    monkeypatch.setattr(parametric, "_estimate_beta", lambda *_: (-2, -3))
    _, diagnostics = parametric.generate_calibrated_field(
        (32, 32),
        _parameters(),
        1,
        20,
        0,
        seed=4,
        distance_threshold=0,
        return_diagnostics=True,
    )
    assert diagnostics["attempts"] == 5 and diagnostics["selected_attempt"] == 1


def test_all_unavailable_spectra_select_last_candidate(monkeypatch):
    monkeypatch.setattr(parametric, "_estimate_beta", lambda *_: (np.nan, np.nan))
    field, diagnostics = parametric.generate_calibrated_field(
        (32, 32), _parameters(), 1, 20, 0, seed=4, return_diagnostics=True
    )
    assert np.isfinite(field).all() and np.isinf(diagnostics["distance"])
    assert diagnostics["attempts"] == diagnostics["selected_attempt"] == 5
    assert (
        diagnostics["status"] == "unavailable_spectrum"
        and not diagnostics["tolerance_achieved"]
    )


def test_numerical_spectrum_failure_is_reported(monkeypatch):
    def unavailable(*_):
        raise ValueError("unavailable numerical spectrum")

    monkeypatch.setattr(parametric, "rapsd_logarithmic", unavailable)
    field, diagnostics = parametric.generate_calibrated_field(
        (32, 32), _parameters(0.3), 1, 20, 0, seed=4, return_diagnostics=True
    )
    assert np.isfinite(field).all() and diagnostics["status"] == "unavailable_spectrum"


def test_higher_fraction_uses_one_candidate(monkeypatch):
    monkeypatch.setattr(parametric, "_estimate_beta", lambda *_: (-1, -2))
    _, diagnostics = parametric.generate_calibrated_field(
        (32, 32),
        _parameters(0.1001),
        1,
        20,
        0,
        seed=4,
        distance_threshold=0.01,
        return_diagnostics=True,
    )
    assert diagnostics["attempts"] == 1


def test_parameters_seed_stream_and_calibration_controls():
    parameters = _parameters(0.3)
    original = copy.deepcopy(parameters)
    global_state = np.random.get_state()
    first, diagnostics = parametric.generate_calibrated_field(
        (64, 96), parameters, 1, 20, 0, seed=0, return_diagnostics=True
    )
    assert_array_equal(
        first,
        parametric.generate_calibrated_field((64, 96), parameters, 1, 20, 0, seed=0),
    )
    after = np.random.get_state()
    assert global_state[0] == after[0] and global_state[2:] == after[2:]
    assert_array_equal(global_state[1], after[1])
    assert parameters == original
    assert diagnostics["calibrated_parameters"]["mean_db"] == parameters["mean_db"]
    assert diagnostics["calibrated_parameters"]["stdev_db"] == parameters["stdev_db"]
    assert diagnostics["calibrated_parameters"]["beta_1"] == parameters["beta_1"] - 0.5
    rng = np.random.default_rng(7)
    a = parametric.generate_stochastic_field((64, 96), parameters, 1, 20, 0, rng=rng)
    b = parametric.generate_stochastic_field((64, 96), parameters, 1, 20, 0, rng=rng)
    assert not np.array_equal(a, b)
    custom, custom_diagnostics = parametric.generate_calibrated_field(
        (64, 96),
        parameters,
        1,
        20,
        0,
        seed=0,
        spectral_bias=-0.2,
        return_diagnostics=True,
    )
    assert (
        custom_diagnostics["calibrated_parameters"]["beta_1"]
        == parameters["beta_1"] - 0.2
    )
    assert not np.array_equal(first, custom)


def test_dry_target_does_not_consume_random_draw():
    rng = np.random.default_rng(7)
    before = copy.deepcopy(rng.bit_generator.state)
    field, diagnostics = parametric.generate_calibrated_field(
        (32, 32),
        _parameters(0),
        1,
        20,
        0,
        rng=rng,
        zerovalue=-1,
        return_diagnostics=True,
    )
    assert_array_equal(field, -1)
    assert diagnostics["status"] == "dry" and diagnostics["attempts"] == 0
    assert not diagnostics["tolerance_achieved"] and np.isnan(diagnostics["distance"])
    assert rng.bit_generator.state == before


def test_post_ar_dry_policy_is_explicit_and_invalid_data_is_not_dry():
    field = np.random.default_rng(2).normal(size=(64, 64))
    suppressed = match_rainfall_distribution(
        field, 0.01, 6, 2, 0, min_rain_fraction=0.025
    )
    assert_array_equal(suppressed, -0.1)
    generated = match_rainfall_distribution(field, 0.01, 6, 2, 0)
    assert np.any(generated > 0)
    with pytest.raises(ValueError):
        match_rainfall_distribution(np.full((8, 8), np.nan), 0, 6, 2, 0)
    with pytest.raises(ValueError):
        match_rainfall_distribution(np.ones((8, 8)), 0.1, 6, 2, 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_attempts": 0},
        {"max_attempts": 1.5},
        {"distance_threshold": -1},
        {"distance_threshold": np.inf},
        {"low_war_threshold": 2},
        {"beta_variances": [0, 1]},
        {"beta_variances": [1]},
        {"beta_variances": np.array([1 + 1j, 1])},
        {"nbins": 1},
        {"eps": 0.5},
        {"zerovalue": 0},
        {"seed": 0, "rng": np.random.default_rng(0)},
    ],
)
def test_invalid_settings_raise(changes):
    with pytest.raises(ValueError):
        parametric.generate_calibrated_field(
            (32, 32), _parameters(), 1, 20, 0, **changes
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"rain_fraction": -1},
        {"rain_fraction": 1.1},
        {"mean_db": np.nan},
        {"stdev_db": 0},
        {"beta_1": np.inf},
        {"beta_2": None},
    ],
)
def test_invalid_parameters_raise(changes):
    with pytest.raises(ValueError):
        parametric.generate_calibrated_field(
            (32, 32), dict(_parameters(), **changes), 1, 20, 0
        )


def test_degenerate_geometry_and_too_sparse_mapping_are_explicit():
    with pytest.raises(ValueError):
        parametric.power_law_filter((16, 16), -2, -3, 1, None)
    with pytest.raises(ValueError):
        parametric.generate_calibrated_field((2, 8), _parameters(), 1, 20, 0)
    with pytest.raises(ValueError):
        parametric.generate_stochastic_field(
            (16, 16), _parameters(1e-12), 1, 20, 0, seed=0
        )
