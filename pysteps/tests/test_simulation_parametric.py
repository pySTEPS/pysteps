"""Check prescribed rainfall simulation and its shared forecast evolution."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.noise.parametric import generate_calibrated_field
from pysteps.nowcasts import parametric
from pysteps.timeseries.parametric import (
    ar2_correlation_time,
    estimate_reference_correlation_time,
    lagrangian_autocorrelation,
)
from pysteps.utils.spectral import fit_power_law_spectrum, rapsd_logarithmic
from pysteps.utils.statistics import compute_field_statistics
from pysteps.utils.transformation import dB_transform


@pytest.fixture
def initial():
    """Build a translating history with known motion and a fractional cadence."""
    rng = np.random.default_rng(17)
    rain = rng.lognormal(1.0, 0.8, (32, 40))
    rain[rng.random(rain.shape) > 0.3] = 0
    return xr.Dataset(
        {
            "rain": (
                ("time", "y", "x"),
                np.stack([np.roll(rain, -2, 1), np.roll(rain, -1, 1), rain]),
                {"units": "mm/h", "transform": None},
            ),
            "velocity_x": (
                ("y", "x"),
                np.ones(rain.shape),
                {"units": "pixels/timestep"},
            ),
            "velocity_y": (("y", "x"), np.zeros(rain.shape)),
        },
        coords={
            "time": (
                "time",
                np.datetime64("2026-01-01T00:00:00")
                + np.arange(3) * np.timedelta64(450, "s"),
                {"stepsize": 450},
            ),
            "x": ("x", (np.arange(40) + 0.5) * 1000, {"units": "m", "stepsize": 1000}),
            "y": (
                "y",
                (np.arange(32)[::-1] + 0.5) * 1000,
                {"units": "m", "stepsize": -1000},
            ),
        },
        attrs={"precip_var": "rain", "projection": "+proj=longlat +datum=WGS84"},
    )


@pytest.fixture
def targets():
    """Return generation targets distinct from the history's sample statistics."""
    return dict(beta_1=-2.4, beta_2=-3.1, rain_fraction=0.2, mean_db=6.0, stdev_db=3.0)


def _simulate(initial, targets, **kwargs):
    """Supply example settings while allowing individual contract checks."""
    settings = dict(
        parameters=targets,
        reference_time=400.0,
        reference_wavelength=256.0,
        timesteps=3,
        n_ens_members=2,
        seed=42,
        return_diagnostics=True,
    )
    settings.update(kwargs)
    return parametric.simulate(initial, **settings)


def test_prescribed_scale_law_and_output_coordinates(initial, targets):
    output, diagnostics = _simulate(initial.transpose("x", "time", "y"), targets)
    expected = 400 * (diagnostics["wavelengths_km"] / 256) ** 0.81
    np.testing.assert_allclose(
        diagnostics["temporal_model"]["requested_times"], expected
    )
    assert diagnostics["settings"]["timestep_minutes"] == 7.5
    assert diagnostics["parameters"] == targets
    assert diagnostics["parameter_source"] == "supplied"
    assert output.rain.dims == ("ens_number", "time", "y", "x")
    assert output.rain.shape == (2, 3, 32, 40)
    assert np.isfinite(output.rain).all() and (output.rain >= 0).all()
    assert np.all((output.rain == 0) | (output.rain >= 1))
    np.testing.assert_array_equal(output.x, initial.x)
    np.testing.assert_array_equal(output.y, initial.y)
    np.testing.assert_array_equal(output.ens_number, [1, 2])
    np.testing.assert_array_equal(
        output.time,
        initial.time.values[-1] + np.arange(1, 4) * np.timedelta64(450, "s"),
    )
    assert output.forecast_reference_time == initial.time.values[-1]
    assert output.attrs["projection"] == initial.attrs["projection"]
    assert "transform" not in output.rain.attrs

    # Sum the AR recurrence independently to check the achieved discrete times.
    phi = diagnostics["ar_coefficients"]
    previous = np.ones(len(phi))
    current = diagnostics["model_correlations"][:, 0].copy()
    total = previous + current
    for _ in range(10000):
        previous, current = current, phi[:, 0] * current + phi[:, 1] * previous
        total += current
    np.testing.assert_allclose(total * 7.5, np.maximum(expected, 7.5), atol=1e-7)


def test_simulation_does_not_estimate_or_replace_parameters(
    initial, targets, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Simulation called an observation-based estimator")

    for name in [
        "compute_field_statistics",
        "fit_power_law_spectrum",
        "rapsd_logarithmic",
        "lagrangian_autocorrelation",
        "_generation_parameters",
    ]:
        monkeypatch.setattr(parametric, name, forbidden)
    monkeypatch.setattr(parametric.motion, "get_method", forbidden)
    initial.rain.values.fill(0)
    # These valid supplied slopes violate the forecast's regional selection bounds.
    targets.update(beta_1=-1.0, beta_2=-0.7, rain_fraction=0.04)
    output, diagnostics = _simulate(initial, targets, reference_time=1200)
    assert diagnostics["parameters"] == targets
    assert diagnostics["settings"]["reference_time_minutes"] == 1200
    assert diagnostics["settings"]["min_rain_fraction"] == 0
    assert (output.rain > 0).any()


def test_simulation_matches_forecast_with_identical_parameters():
    folder = Path(__file__).parent / "data"
    case = json.loads((folder / "parametric_nowcast_reference.json").read_text())[
        "cases"
    ][0]
    with np.load(folder / "parametric_nowcast_reference.npz") as data:
        rain, velocity = data["history_0"], data["velocity_0"]
    history = xr.Dataset(
        {
            "rain": (("time", "y", "x"), rain, {"units": "mm/h"}),
            "velocity_x": (("y", "x"), velocity[0]),
            "velocity_y": (("y", "x"), velocity[1]),
        },
        coords={
            "time": (
                "time",
                np.datetime64("2026-01-01") + np.arange(3) * np.timedelta64(5, "m"),
                {"stepsize": 300},
            ),
            "x": ("x", np.arange(rain.shape[2]), {"units": "km", "stepsize": 1}),
            "y": ("y", np.arange(rain.shape[1]), {"units": "km", "stepsize": 1}),
        },
        attrs={"precip_var": "rain"},
    )
    forecast, diagnostics = parametric.forecast(
        history,
        timesteps=3,
        n_ens_members=2,
        seed=case["seed"],
        return_diagnostics=True,
    )
    targets = {
        name: diagnostics["parameters"][name]
        for name in ("beta_1", "beta_2", "rain_fraction", "mean_db", "stdev_db")
    }
    simulation, supplied = _simulate(
        history,
        targets,
        seed=case["seed"],
        min_rain_fraction=0.025,
        reference_time=diagnostics["parameters"]["reference_time"],
        reference_wavelength=diagnostics["wavelengths_km"][0],
    )
    xr.testing.assert_identical(simulation, forecast)
    np.testing.assert_array_equal(
        supplied["ar_coefficients"], diagnostics["ar_coefficients"]
    )


def test_streaming_purity_and_member_streams(initial, targets):
    saved = initial.copy(deep=True)
    supplied_targets = targets.copy()
    noise_settings = dict(max_attempts=2)
    seed = np.random.SeedSequence(42)
    seed.spawn(2)
    seed_state = deepcopy(seed.state)
    global_state = np.random.get_state()
    output, diagnostics = _simulate(
        initial, targets, seed=seed, noise_kwargs=noise_settings
    )
    streamed = []

    def callback(field):
        streamed.append(field.copy())
        field.fill(-1)

    result, streamed_diagnostics = _simulate(
        initial,
        targets,
        seed=seed,
        noise_kwargs=noise_settings,
        callback=callback,
        return_output=False,
    )
    assert result is None
    np.testing.assert_array_equal(np.stack(streamed, axis=1), output.rain)
    assert streamed_diagnostics["seed"] == diagnostics["seed"]
    larger, _ = _simulate(
        initial, targets, seed=seed, n_ens_members=3, noise_kwargs=noise_settings
    )
    np.testing.assert_array_equal(larger.rain[:2], output.rain)
    xr.testing.assert_identical(initial, saved)
    assert targets == supplied_targets
    assert noise_settings == dict(max_attempts=2)
    assert seed.state == seed_state
    np.testing.assert_equal(np.random.get_state(), global_state)


def test_dry_target_validates_input_and_skips_generation(initial, targets, monkeypatch):
    targets["rain_fraction"] = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("A dry target consumed stochastic generation")

    monkeypatch.setattr(parametric, "generate_calibrated_field", forbidden)
    output, diagnostics = _simulate(initial, targets)
    assert (output.rain == 0).all()
    assert diagnostics["status"] == "dry_target"
    assert diagnostics["generation"] == [[], []]
    initial.rain.values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _simulate(initial, targets)


@pytest.mark.parametrize(
    "name,value",
    [
        ("reference_time", 0),
        ("reference_time", np.inf),
        ("reference_wavelength", -1),
        ("alpha", 0),
        ("temporal_exponent", np.nan),
        ("timesteps", 0),
        ("n_ens_members", 0),
        ("callback", 1),
        ("return_output", 1),
    ],
)
def test_invalid_simulation_settings(initial, targets, name, value):
    with pytest.raises(ValueError):
        _simulate(initial, targets, **{name: value})


@pytest.mark.parametrize(
    "change",
    [
        dict(stdev_db=0),
        dict(rain_fraction=-0.1),
        dict(rain_fraction=1.1),
        dict(mean_db=np.nan),
        dict(beta_1=1j),
        dict(unknown=1),
    ],
)
def test_invalid_simulation_targets(initial, targets, change):
    targets.update(change)
    with pytest.raises(ValueError):
        _simulate(initial, targets)


def test_simulation_requires_motion_and_complete_parameter_mapping(initial, targets):
    with pytest.raises(ValueError, match="requires both"):
        _simulate(initial.drop_vars(["velocity_x", "velocity_y"]), targets)
    targets.pop("stdev_db")
    with pytest.raises(ValueError, match="exactly"):
        _simulate(initial, targets)


def test_subinterval_times_resolve_to_white_noise(initial, targets):
    _, diagnostics = _simulate(initial, targets, reference_time=1.0)
    assert diagnostics["temporal_model"]["limited_below_timestep"].all()
    np.testing.assert_array_equal(diagnostics["model_correlations"], 0)
    np.testing.assert_array_equal(diagnostics["temporal_model"]["achieved_times"], 7.5)


def test_simulation_example_end_to_end_parameter_recovery():
    """Regress the fixed-seed gallery workflow through output parameter recovery."""
    reference = json.loads(
        (
            Path(__file__).parent / "data/parametric_simulation_reference.json"
        ).read_text()
    )
    case, expected = reference["case"], reference["expected"]
    shape = tuple(case["shape"])
    threshold_db = 10 * np.log10(case["rain_threshold"])
    interval = np.timedelta64(round(case["timestep_minutes"] * 60), "s")

    # Start from a generated field and its translated copies, as in the example.
    # The test uses public functions without importing plotting code or running
    # the example's Monte Carlo experiment.
    transformed = generate_calibrated_field(
        shape,
        case["parameters"],
        case["kmperpixel"],
        case["scale_break_km"],
        threshold_db,
        seed=case["initial_seed"],
    )
    rain = 10 ** (transformed / 10)
    rain[rain < case["rain_threshold"]] = 0
    velocity = np.array(
        [
            np.full(shape, component)
            for component in case["velocity_pixels_per_timestep"]
        ]
    )
    history = xr.Dataset(
        {
            "rain": (
                ("time", "y", "x"),
                np.stack([np.roll(rain, -2, axis=1), np.roll(rain, -1, axis=1), rain]),
                {"units": "mm/h"},
            ),
            "velocity_x": (("y", "x"), velocity[0], {"units": "pixels/timestep"}),
            "velocity_y": (("y", "x"), velocity[1], {"units": "pixels/timestep"}),
        },
        coords={
            "time": (
                "time",
                np.datetime64("2026-01-01T00:00:00") + np.arange(3) * interval,
                {"stepsize": case["timestep_minutes"] * 60},
            ),
            "x": (
                "x",
                (np.arange(shape[1]) + 0.5) * case["kmperpixel"],
                {"units": "km", "stepsize": case["kmperpixel"]},
            ),
            "y": (
                "y",
                (np.arange(shape[0]) + 0.5) * case["kmperpixel"],
                {"units": "km", "stepsize": case["kmperpixel"]},
            ),
        },
        attrs={"precip_var": "rain"},
    )
    output, diagnostics = parametric.simulate(
        history,
        timesteps=case["n_timesteps"],
        parameters=case["parameters"],
        reference_time=case["reference_time_minutes"],
        reference_wavelength=case["reference_wavelength_km"],
        temporal_exponent=case["temporal_exponent"],
        alpha=case["alpha"],
        n_cascade_levels=case["n_cascade_levels"],
        rain_threshold=case["rain_threshold"],
        scale_break=case["scale_break_km"],
        n_ens_members=case["n_ens_members"],
        seed=case["evolution_seed"],
        return_diagnostics=True,
    )
    assert output.rain.dims == ("ens_number", "time", "y", "x")
    assert output.rain.shape == (case["n_ens_members"], case["n_timesteps"], *shape)
    assert output.rain.dtype == np.float64
    assert np.isfinite(output.rain).all() and (output.rain >= 0).all()
    assert np.all((output.rain == 0) | (output.rain >= case["rain_threshold"]))
    assert output.rain.attrs["units"] == "mm/h"
    assert output.rain.attrs["threshold"] == case["rain_threshold"]
    assert output.rain.attrs["zerovalue"] == 0
    assert "transform" not in output.rain.attrs
    xr.testing.assert_identical(output.x.variable, history.x.variable)
    xr.testing.assert_identical(output.y.variable, history.y.variable)
    np.testing.assert_array_equal(output.ens_number, [1])
    np.testing.assert_array_equal(
        output.time,
        history.time.values[-1] + np.arange(1, case["n_timesteps"] + 1) * interval,
    )
    assert output.time.attrs["stepsize"] == case["timestep_minutes"] * 60
    assert output.forecast_reference_time == history.time.values[-1]
    assert diagnostics["parameters"] == case["parameters"]

    bandpass = filter_gaussian(shape, case["n_cascade_levels"])
    wavelengths = case["kmperpixel"] / bandpass["central_freqs"]
    np.testing.assert_allclose(wavelengths, expected["wavelengths_km"], rtol=1e-12)
    np.testing.assert_allclose(
        diagnostics["temporal_model"]["requested_times"],
        case["reference_time_minutes"]
        * (wavelengths / case["reference_wavelength_km"]) ** case["temporal_exponent"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        diagnostics["model_correlations"],
        expected["model_correlations"],
        atol=1e-10,
        rtol=1e-10,
    )

    # Recover statistics from every returned field, rather than accepting the
    # supplied parameters in diagnostics as evidence of the generated behaviour.
    fields = (
        dB_transform(
            output,
            threshold=case["rain_threshold"],
            zerovalue=threshold_db - 0.1,
        )
        .rain.isel(ens_number=0)
        .values
    )
    recovered = []
    cascades = []
    for field in fields:
        statistics = compute_field_statistics(
            field, threshold_db, case["lower_truncation_dbr"]
        )
        frequency, power = rapsd_logarithmic(field)
        spectrum = fit_power_law_spectrum(
            frequency,
            power,
            scale_break=case["scale_break_km"] / case["kmperpixel"],
        )
        assert statistics["nonzero_fit"]["status"] == "ok"
        assert spectrum["status"] == "ok"
        recovered.append(
            [
                statistics["rain_fraction"],
                statistics["latent_mean"],
                statistics["latent_stdev"],
                spectrum["beta_1"],
                spectrum["beta_2"],
            ]
        )
        cascades.append(
            decomposition_fft(
                field,
                bandpass,
                compute_stats=True,
                normalize=True,
            )["cascade_levels"]
        )
    recovered = np.asarray(recovered)
    assert np.isfinite(recovered).all()
    # Fixed seeds remove Monte Carlo sampling variability from this comparison.
    # Allow two threshold-crossing pixels and small floating-point differences
    # in fitted moments/slopes; these are numerical regression tolerances.
    tolerances = [2 / np.prod(shape), 1e-4, 1e-4, 1e-4, 1e-4]
    for column, (name, tolerance) in enumerate(
        zip(expected["field_parameter_names"], tolerances)
    ):
        np.testing.assert_allclose(
            recovered[:, column],
            np.asarray(expected["field_parameters"])[:, column],
            rtol=1e-5,
            atol=tolerance,
            err_msg=name,
        )

    # Include only simulated fields: the artificial initialization is excluded
    # from the ten overlapping three-field histories, but its transient persists.
    cascades = np.stack(cascades)
    observed = np.mean(
        [
            lagrangian_autocorrelation(cascades[end - 2 : end + 1], velocity)
            for end in range(2, len(cascades))
        ],
        axis=0,
    )
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(
        observed, expected["mean_observed_correlations"], atol=1e-6, rtol=0
    )
    times = ar2_correlation_time(
        observed[:, 0], observed[:, 1], case["timestep_minutes"]
    )
    reference_time = estimate_reference_correlation_time(
        times,
        wavelengths,
        case["temporal_exponent"],
        levels=(0, 1, 2),
        reference_wavelength=case["reference_wavelength_km"],
    )
    # The reference describes this initialization and one-hour run, not stationary
    # recovery of the prescribed 400 minutes. Do not make a bias factor a model
    # requirement. High correlations amplify roundoff in the implied times, so
    # check the lag pairs tightly and allow 0.1% on their nonlinear conversion.
    np.testing.assert_allclose(
        times, expected["implied_correlation_times_minutes"], rtol=1e-3, atol=1e-6
    )
    assert reference_time == pytest.approx(
        expected["implied_reference_time_minutes"], rel=1e-3
    )
