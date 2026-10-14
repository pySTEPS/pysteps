"""Independent numerical checks and frozen source-implementation comparisons."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import truncnorm

from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.timeseries.parametric import (
    ar2_correlation_time,
    estimate_reference_correlation_time,
    fit_ar2_correlations,
    lagrangian_autocorrelation,
    model_cascade_correlations,
)
from pysteps.utils.spectral import fit_power_law_spectrum, rapsd_logarithmic
from pysteps.utils.statistics import (
    compute_field_statistics,
    fit_lower_truncated_normal,
)


def _directional_field(shape):
    y, x = np.indices(shape)
    return np.random.default_rng(23).normal(size=shape) + 4 * np.cos(
        2 * np.pi * (5 * y / shape[0] + 3 * x / shape[1])
    )


def _marginal_field():
    rng = np.random.default_rng(37)
    field = rng.normal(4, 3, size=(128, 96))
    field[rng.uniform(size=field.shape) > 0.3] = -0.1
    return field


def _lag_history():
    rng = np.random.default_rng(45)
    history = rng.normal(size=(3, 2, 24, 32))
    velocity = np.full((2, 24, 32), 0.25)
    velocity[0] = 0.6
    return history, velocity


@pytest.fixture(scope="module")
def reference():
    return json.loads(
        (
            Path(__file__).parent / "data/parametric_estimators_reference.json"
        ).read_text()
    )


def test_source_marginal_parity(reference):
    field = _marginal_field()
    original = field.copy()
    estimated = compute_field_statistics(field, 0, lower_truncation=1.25)
    for name, expected in reference["marginal"].items():
        assert estimated[name] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert_array_equal(field, original)


def test_source_spectrum_parity(reference):
    for case in reference["spectra"]:
        frequency, power = rapsd_logarithmic(_directional_field(tuple(case["shape"])))
        assert_allclose(10 * np.log10(frequency), case["frequency_db"], atol=1e-12)
        assert_allclose(10 * np.log10(power), case["power_db"], atol=1e-10)
        for expected in case["models"]:
            estimated = fit_power_law_spectrum(
                frequency, power, expected["scale_break"]
            )
            assert estimated["status"] == "ok"
            for name in ("beta_1", "beta_2", "c1", "c2"):
                if expected[name] is None:
                    assert estimated[name] is None
                else:
                    assert estimated[name] == pytest.approx(expected[name], abs=3e-5)


def test_source_temporal_and_paper_reference_parity(reference):
    history, velocity = _lag_history()
    assert_allclose(
        lagrangian_autocorrelation(history, velocity), reference["lags"], atol=1e-14
    )
    for case in reference["ar2_inverse"]:
        pair = fit_ar2_correlations(case["time"], case["alpha"], 5)
        assert_allclose(pair, case["pair"], atol=1e-12)
    paper = reference["paper_reference"]
    assert_allclose(
        estimate_reference_correlation_time(
            paper["times"], paper["wavelengths"], paper["exponent"]
        ),
        paper["reference_times"],
        rtol=1e-14,
    )


@pytest.mark.parametrize("shape", [(128, 128), (127, 127), (96, 128), (95, 127)])
def test_radial_mean_matches_independent_full_fft(shape):
    field = _directional_field(shape)
    original = field.copy()
    normalized = (field - field.mean()) / field.std()
    full_power = abs(np.fft.fft2(normalized)) ** 2
    y, x = np.meshgrid(
        np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), indexing="ij"
    )
    radius = np.hypot(y, x)
    n = min(shape) // 2
    limit = np.hypot(
        abs(np.fft.fftfreq(shape[0])[: shape[0] // 2]).max(),
        abs(np.fft.fftfreq(shape[1])[: shape[1] // 2]).max(),
    )
    # Preserve the historical endpoint rounding: exactly-boundary Fourier modes
    # can move bins if geomspace is substituted for the source logspace edges.
    edges = np.logspace(np.log10(1 / n), np.log10(limit), n)
    expected_frequency, expected_power = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (radius >= lo) & (radius < hi)
        if mask.any():
            expected_frequency.append((lo + hi) / 2)
            expected_power.append(full_power[mask].mean())
    frequency, power = rapsd_logarithmic(field)
    assert_allclose(frequency, expected_frequency, atol=1e-14)
    assert_allclose(power, expected_power, rtol=1e-12)
    for changed in (field.T, field[:, ::-1], np.rot90(field)):
        actual_frequency, actual_power = rapsd_logarithmic(changed)
        assert_allclose(actual_frequency, frequency, atol=1e-14)
        assert_allclose(actual_power, power, rtol=1e-12)
    assert_array_equal(field, original)


def test_exact_continuous_power_law_recovery():
    frequency = np.geomspace(0.005, 0.5, 120)
    breakpoint = 1 / 20
    power = np.where(
        frequency <= breakpoint, frequency**-2, breakpoint**1.5 * frequency**-3.5
    )
    result = fit_power_law_spectrum(frequency, power, scale_break=20)
    assert result["status"] == "ok"
    assert_allclose([result["beta_1"], result["beta_2"]], [-2, -3.5], atol=1e-7)
    assert result["rmse_db"] < 1e-7
    assert result["c1"] + result["beta_1"] * 10 * np.log10(breakpoint) == pytest.approx(
        result["c2"] + result["beta_2"] * 10 * np.log10(breakpoint)
    )


def test_unavailable_spectral_fits_are_explicit():
    frequency, power = rapsd_logarithmic(np.full((32, 32), -0.1))
    assert frequency.size == power.size == 0
    assert (
        fit_power_law_spectrum(frequency, power, 20)["status"] == "insufficient_samples"
    )
    frequency = np.geomspace(0.02, 0.2, 30)
    assert (
        fit_power_law_spectrum(frequency, np.ones(30), 1000)["status"]
        == "unidentified_model"
    )
    assert (
        fit_power_law_spectrum(frequency, np.zeros(30))["status"]
        == "insufficient_samples"
    )


def test_latent_normal_recovery_and_occurrence_cutoff():
    mean, scale, cutoff = 2.0, 3.0, 1.25
    probabilities = (np.arange(20000) + 0.5) / 20000
    sample = truncnorm.ppf(
        probabilities, (cutoff - mean) / scale, np.inf, loc=mean, scale=scale
    )
    fitted_mean, fitted_scale, diagnostics = fit_lower_truncated_normal(
        sample, cutoff, True
    )
    assert_allclose([fitted_mean, fitted_scale], [mean, scale], atol=0.003)
    assert diagnostics["status"] == "ok"
    field = np.array([[-0.1, 0, 0.5, 2], [1, 3, 4, 5]])
    a = compute_field_statistics(field, 0, lower_truncation=1.25)
    b = compute_field_statistics(field, 0, lower_truncation=2)
    assert a["rain_fraction"] == b["rain_fraction"] == 7 / 8
    assert a["ordinary_mean"] == b["ordinary_mean"]
    assert a["nonzero_fit"]["n_samples"] == 4


def test_dry_constant_and_bounded_marginal_fits():
    dry = compute_field_statistics(np.full((4, 4), -0.1), 0, lower_truncation=1.25)
    assert dry["rain_fraction"] == 0
    assert np.isnan(dry["ordinary_mean"]) and np.isnan(dry["latent_mean"])
    assert dry["nonzero_fit"]["status"] == "insufficient_samples"
    assert fit_lower_truncated_normal([2, 2], 1, True)[2]["status"] == "constant_sample"
    _, _, bounded = fit_lower_truncated_normal([0.01, 0.01, 0.01, 100], 0, True)
    assert bounded["status"] == "bounded_solution" and bounded["boundary_limited"]


def _summed_acf(rho1, rho2, timestep):
    phi1, phi2 = np.linalg.solve([[1, rho1], [rho1, 1]], [rho1, rho2])
    previous, current, total = 1.0, rho1, 1.0 + rho1
    for _ in range(100000):
        previous, current = current, phi1 * current + phi2 * previous
        total += current
        if abs(previous) + abs(current) < 1e-13:
            return timestep * total
    raise AssertionError("ACF recurrence did not converge")


@pytest.mark.parametrize(
    "phi1,phi2", [(1.2, -0.3), (0.4, 0.3), (-0.6, -0.3), (0.9, 0), (0, 0)]
)
def test_forward_acf_sum(phi1, phi2):
    rho1 = phi1 / (1 - phi2)
    rho2 = phi1 * rho1 + phi2
    assert ar2_correlation_time(rho1, rho2, 5) == pytest.approx(
        _summed_acf(rho1, rho2, 5), rel=1e-10
    )
    assert ar2_correlation_time(rho1, rho2, 300) == pytest.approx(
        60 * ar2_correlation_time(rho1, rho2, 5)
    )


@pytest.mark.parametrize("alpha", [0.8, 1.25, 2, 2.72, 4])
@pytest.mark.parametrize("target", [5, 5.00001, 10, 60, 420, 720])
def test_inverse_matches_independent_acf_sum(alpha, target):
    pair, diagnostics = fit_ar2_correlations(target, alpha, 5, True)
    assert _summed_acf(*pair, 5) == pytest.approx(target, rel=1e-7)
    assert diagnostics["achieved_time"] == pytest.approx(target, rel=1e-7)


def test_reference_selection_and_nominal_scale_model():
    wavelength = 1 / filter_gaussian((256, 256), 6, d=1)["central_freqs"]
    exponent = 0.7718612877449645
    observed = 120 * (wavelength / wavelength[0]) ** exponent
    assert estimate_reference_correlation_time(
        observed, wavelength, exponent
    ) == pytest.approx(120)
    altered = observed.copy()
    altered[-1] = np.nan
    assert estimate_reference_correlation_time(
        altered, wavelength, exponent
    ) == pytest.approx(120)
    altered[1] = np.nan
    assert np.isnan(estimate_reference_correlation_time(altered, wavelength, exponent))
    assert estimate_reference_correlation_time(
        altered, wavelength, exponent, levels=(0,)
    ) == pytest.approx(120)
    pairs, diagnostics = model_cascade_correlations(
        60, wavelength, exponent, 2.72, 5, return_diagnostics=True
    )
    assert_allclose(
        diagnostics["achieved_times"],
        np.maximum(diagnostics["requested_times"], 5),
        rtol=1e-7,
    )
    assert_array_equal(pairs[diagnostics["limited_below_timestep"]], 0)
    # Changing the reference convention must not change the predicted level times.
    rescaled = estimate_reference_correlation_time(
        observed, wavelength, exponent, reference_wavelength=128
    )
    alternate = model_cascade_correlations(
        rescaled, wavelength, exponent, 2.72, 5, reference_wavelength=128
    )
    original = model_cascade_correlations(120, wavelength, exponent, 2.72, 5)
    assert_allclose(alternate, original, atol=1e-12)
    flat = model_cascade_correlations(120, wavelength, 0, 2.72, 5)
    assert_allclose(ar2_correlation_time(flat[:, 0], flat[:, 1], 5), 120)


def test_lagrangian_boundary_conventions_and_input_preservation():
    base = np.random.default_rng(7).normal(size=(32, 40))
    history = np.stack([base, np.roll(base, 1, axis=1), np.roll(base, 2, axis=1)])[
        :, None
    ]
    velocity = np.zeros((2, 32, 40))
    velocity[0] = 1
    original = history.copy()
    assert_allclose(
        lagrangian_autocorrelation(history, velocity, boundary="mask"),
        [[1, 1]],
        atol=1e-14,
    )
    zero = lagrangian_autocorrelation(history, velocity)
    assert (zero < 1).all()
    assert_array_equal(history, original)
    assert np.isnan(lagrangian_autocorrelation(np.ones_like(history), velocity)).all()
    assert_allclose(
        lagrangian_autocorrelation(history, np.zeros_like(velocity)),
        [
            [
                np.corrcoef(history[-1, 0].ravel(), history[-2, 0].ravel())[0, 1],
                np.corrcoef(history[-1, 0].ravel(), history[-3, 0].ravel())[0, 1],
            ]
        ],
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda: compute_field_statistics([[np.nan]], 0),
        lambda: compute_field_statistics([[1j]], 0),
        lambda: compute_field_statistics([[1]], 1, lower_truncation=0),
        lambda: fit_lower_truncated_normal([1, 2], np.nan),
        lambda: rapsd_logarithmic(np.zeros((2, 2))),
        lambda: rapsd_logarithmic(np.full((8, 8), np.nan)),
        lambda: rapsd_logarithmic(np.full((8, 8), 1j)),
        lambda: fit_power_law_spectrum([0.2, 0.1], [1, 2]),
        lambda: fit_ar2_correlations(4, 2.72, 5),
        lambda: fit_ar2_correlations(1e20, 2.72, 5),
        lambda: estimate_reference_correlation_time([1, 2], [10, 5], 0.8),
        lambda: model_cascade_correlations(60, [10, 5], 0.8, 0, 5),
        lambda: lagrangian_autocorrelation(np.zeros((2, 1, 8, 8)), np.zeros((2, 8, 8))),
    ],
)
def test_invalid_inputs_raise(operation):
    with pytest.raises(ValueError):
        operation()


def test_invalid_ar_pairs_are_not_repaired():
    assert np.isnan(ar2_correlation_time([np.nan, 1, 0.5], [0, 0.5, 1.5], 5)).all()
    assert np.isnan(ar2_correlation_time(0.5, 0.2, 0))
