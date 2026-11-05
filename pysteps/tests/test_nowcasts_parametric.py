"""Forecast contracts and source numerical parity, without operational services."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter

from pysteps import nowcasts
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.extrapolation.semilagrangian import extrapolate
from pysteps.nowcasts import parametric
from pysteps.nowcasts._cascade_advection import (
    advect_field,
    prepare_advection_coordinates,
)
from pysteps.timeseries.autoregression import estimate_ar_params_yw


def _dataset(rain, velocity=None, timestep=5.0, kmperpixel=1.0):
    """Attach physical coordinates without changing the saved reference arrays."""
    dataset = xr.Dataset(
        {
            "precip_intensity": (
                ("time", "y", "x"),
                rain.copy(),
                {
                    "units": "mm/h",
                    "transform": None,
                    "threshold": 0.1,
                    "zerovalue": 0.0,
                    "accutime": timestep,
                },
            ),
        },
        coords={
            "time": (
                "time",
                np.datetime64("2026-01-01T00:00:00", "ns")
                + np.arange(rain.shape[0])
                * np.timedelta64(round(timestep * 60e9), "ns"),
                {"stepsize": timestep * 60},
            ),
            "x": (
                "x",
                (np.arange(rain.shape[2]) + 0.5) * kmperpixel * 1000,
                {"units": "m", "stepsize": kmperpixel * 1000},
            ),
            "y": (
                "y",
                (np.arange(rain.shape[1]) + 0.5) * kmperpixel * 1000,
                {"units": "m", "stepsize": kmperpixel * 1000},
            ),
        },
        attrs={
            "precip_var": "precip_intensity",
            "projection": "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +datum=WGS84 +units=m",
            "institution": "reference fixture",
        },
    )
    if velocity is not None:
        dataset["velocity_x"] = (("y", "x"), velocity[0].copy())
        dataset["velocity_y"] = (("y", "x"), velocity[1].copy())
    return dataset


def _forecast_array(rain, velocity=None, timesteps=1, **kwargs):
    """Run original numerical assertions through the public dataset interface."""
    if not isinstance(rain, np.ndarray):
        return parametric.forecast(rain, timesteps, **kwargs)
    dataset = _dataset(
        rain,
        velocity,
        timestep=kwargs.get("timestep", 5.0),
        kmperpixel=kwargs.get("kmperpixel", 1.0),
    )
    result = parametric.forecast(dataset, timesteps, **kwargs)
    if isinstance(result, tuple):
        output, diagnostics = result
        return (
            output["precip_intensity"].values if output is not None else None,
            diagnostics,
        )
    return result["precip_intensity"].values if result is not None else None


@pytest.fixture(scope="module")
def reference():
    folder = Path(__file__).parent / "data"
    metadata = json.loads((folder / "parametric_nowcast_reference.json").read_text())
    with np.load(folder / "parametric_nowcast_reference.npz") as arrays:
        yield metadata, {k: arrays[k].copy() for k in arrays.files}


def _call(reference, index=0, **kwargs):
    metadata, arrays = reference
    case = metadata["cases"][index]
    defaults = dict(
        n_ens_members=2,
        kmperpixel=1,
        timestep=case["config"]["timestep"] / 60,
        seed=case["seed"],
    )
    defaults.update(kwargs)
    return _forecast_array(
        arrays[f"history_{index}"], arrays[f"velocity_{index}"], 3, **defaults
    )


@pytest.mark.parametrize("index", [0, 1])
def test_forecast_source_preparation_and_coupled_output(reference, index):
    metadata, arrays = reference
    case = metadata["cases"][index]
    result, diagnostics = _call(reference, index, return_diagnostics=True)
    assert result.shape == (2, 3, 128, 96)
    assert np.isfinite(result).all() and np.all(result >= 0)
    assert np.all((result == 0) | (result >= 1))
    for name in ("beta_1", "beta_2", "mean_db", "stdev_db", "rain_fraction"):
        assert diagnostics["parameters"][name] == pytest.approx(
            case["parameters"][name], abs=1e-6
        )
    assert diagnostics["parameters"]["reference_time"] == pytest.approx(
        case["parameters"]["corl_zero"], abs=1e-7
    )
    np.testing.assert_allclose(
        diagnostics["model_correlations"][:, 0], case["parameters"]["lag_1"], atol=1e-12
    )
    np.testing.assert_allclose(
        diagnostics["model_correlations"][:, 1], case["parameters"]["lag_2"], atol=1e-12
    )
    assert diagnostics["safeguards"]["moment_source"] == case["moment_source"]
    assert diagnostics["safeguards"]["low_rain_fraction_fallback"] == bool(index)
    # Equivalent spectral solvers differ in the last digits. Histogram mapping
    # and its feedback can amplify that perturbation over multiple leads.
    expected = arrays[f"forecast_{index}"]
    assert np.max(abs(result - expected)) < 0.05
    assert np.sqrt(np.mean((result - expected) ** 2)) < 0.001
    assert np.count_nonzero((result > 0) != (expected > 0)) <= 2
    for member in range(2):
        for lead in range(3):
            actual = diagnostics["generation"][member][lead]
            original = case["calibration"][member][lead]
            for name in ("attempts", "selected_attempt", "tolerance_achieved"):
                assert actual[name] == original[name]


@pytest.mark.parametrize("index", [0, 1])
def test_source_evolution_with_identical_parameters(reference, index):
    metadata, arrays = reference
    case = metadata["cases"][index]
    rain, velocity = arrays[f"history_{index}"], arrays[f"velocity_{index}"]
    transformed = rain.copy()
    wet = transformed >= 1
    transformed[wet] = 10 * np.log10(transformed[wet])
    transformed[~wet] = -0.1
    bp = filter_gaussian(rain.shape[1:], 6)
    observed = [parametric._decompose(f, bp) for f in transformed]
    coordinates = prepare_advection_coordinates(velocity)
    taper = gaussian_filter(advect_field(np.ones(rain.shape[1:]), coordinates[1]), 5)
    original = case["parameters"]
    target = {
        k: original[k]
        for k in ("beta_1", "beta_2", "rain_fraction", "mean_db", "stdev_db")
    }
    phi = np.array(
        [
            estimate_ar_params_yw(pair)
            for pair in zip(original["lag_1"], original["lag_2"])
        ]
    )
    actual = np.empty_like(arrays[f"forecast_{index}"])
    for member, child in enumerate(np.random.SeedSequence(case["seed"]).spawn(2)):
        rng = np.random.default_rng(child)
        histories = [observed[-1], observed[-2]]
        for lead in range(3):
            field, _ = parametric._advance(
                histories,
                target,
                phi,
                bp,
                coordinates,
                rng,
                0,
                -0.1,
                1,
                20,
                parametric._noise_settings(None),
                0.025,
            )
            physical = 10 ** (field / 10)
            physical[physical < 1] = 0
            actual[member, lead] = physical
            histories = [parametric._decompose(field * taper, bp), histories[0]]
    np.testing.assert_allclose(
        actual, arrays[f"forecast_{index}"], atol=1e-8, rtol=1e-10
    )


@pytest.mark.parametrize("kind", ["zero", "translation", "varying"])
def test_cached_advection_exactly_matches_native(kind):
    rng = np.random.default_rng(45)
    shape = (47, 64)
    field = rng.normal(size=shape)
    velocity = np.zeros((2, *shape))
    if kind == "translation":
        velocity[0], velocity[1] = 1.3, -0.4
    elif kind == "varying":
        yy, xx = np.indices(shape)
        velocity[0] = 0.1 + xx / 30
        velocity[1] = -0.2 + yy / 40
    coordinates = prepare_advection_coordinates(velocity)
    for lag in (1, 2):
        native = extrapolate(field, velocity, lag, outval=0)[lag - 1]
        np.testing.assert_array_equal(advect_field(field, coordinates[lag]), native)
        assert not coordinates[lag].flags.writeable


def test_streams_member_isolation_and_input_purity(reference):
    metadata, arrays = reference
    original = {k: v.copy() for k, v in arrays.items()}
    seed = np.random.SeedSequence(72)
    np.random.seed(23)
    global_state = deepcopy(np.random.get_state())
    result = _call(reference, seed=seed)
    again = _call(reference, seed=seed)
    np.testing.assert_array_equal(result, again)
    assert seed.n_children_spawned == 0
    current_state = np.random.get_state()
    assert global_state[0] == current_state[0]
    np.testing.assert_array_equal(global_state[1], current_state[1])
    assert global_state[2:] == current_state[2:]
    prefix = _call(reference, n_ens_members=1, seed=seed)
    np.testing.assert_array_equal(prefix[0], result[0])
    larger = _call(reference, n_ens_members=3, seed=seed)
    np.testing.assert_array_equal(larger[:2], result)
    assert not np.array_equal(result[0], result[1])
    assert not np.array_equal(result, _call(reference, seed=73))
    for k in arrays:
        np.testing.assert_array_equal(arrays[k], original[k])


def test_callback_streaming_matches_output_and_cannot_mutate_state(reference):
    expected = _call(reference)
    leads = []

    def callback(fields):
        leads.append(fields.copy())
        fields.fill(0)

    result = _call(reference, callback=callback)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.stack(leads, axis=1), expected)
    leads.clear()
    result, diagnostics = _call(
        reference, callback=callback, return_output=False, return_diagnostics=True
    )
    assert result is None
    np.testing.assert_array_equal(np.stack(leads, axis=1), expected)
    assert len(diagnostics["generation"][0]) == 3


def test_callback_failure_propagates(reference):
    def callback(fields):
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        _call(reference, callback=callback)


def test_internal_motion_uses_latest_pair_and_reference_configuration(
    reference, monkeypatch
):
    _, arrays = reference
    history = arrays["history_0"]
    calls = []

    def motion(images, **kwargs):
        calls.append((images.copy(deep=True), kwargs))
        images["velocity_x"] = (("y", "x"), arrays["velocity_0"][0].copy())
        images["velocity_y"] = (("y", "x"), arrays["velocity_0"][1].copy())
        return images

    monkeypatch.setattr(parametric.motion, "get_method", lambda name: motion)
    overrides = dict(decl_scale=12)
    output, diagnostics = _forecast_array(
        history,
        timesteps=2,
        n_ens_members=1,
        kmperpixel=1,
        timestep=5,
        seed=4,
        motion_kwargs=overrides,
        return_diagnostics=True,
    )
    assert output.shape == (1, 2, *history.shape[1:])
    assert len(calls) == 1 and calls[0][0].sizes["time"] == 2
    expected = history[-2:].copy()
    wet = expected >= 1
    expected[wet] = 10 * np.log10(expected[wet])
    expected[~wet] = -0.1
    np.testing.assert_array_equal(calls[0][0]["precip_intensity"].values, expected)
    assert calls[0][0]["precip_intensity"].attrs["transform"] == "dB"
    assert calls[0][1]["decl_scale"] == 12
    assert overrides == {"decl_scale": 12}
    assert diagnostics["motion_source"] == "lucaskanade"


def test_bad_estimated_motion_rejected(reference, monkeypatch):
    _, arrays = reference
    monkeypatch.setattr(
        parametric.motion,
        "get_method",
        lambda name: lambda *args, **kw: np.full((2, 128, 96), np.nan),
    )
    with pytest.raises(ValueError, match="invalid velocity"):
        _forecast_array(arrays["history_0"], kmperpixel=1, timestep=5)


@pytest.mark.parametrize("prior_wet", [False, True])
def test_dry_latest_history_skips_estimation_and_motion(prior_wet, monkeypatch):
    rain = np.zeros((3, 64, 64))
    if prior_wet:
        rain[0] = 10

    def unexpected(*args, **kwargs):
        raise AssertionError("Dry history should not require motion or noise")

    monkeypatch.setattr(parametric.motion, "get_method", unexpected)
    monkeypatch.setattr(parametric, "generate_calibrated_field", unexpected)
    leads = []
    output, diagnostics = _forecast_array(
        rain,
        timesteps=2,
        kmperpixel=1,
        timestep=7.5,
        n_ens_members=3,
        callback=lambda f: leads.append(f.copy()),
        return_diagnostics=True,
    )
    assert output.shape == (3, 2, 64, 64) and not output.any()
    assert len(leads) == 2 and not np.stack(leads).any()
    assert diagnostics["status"] == "dry_latest_observation"
    assert diagnostics["parameters"] is None


def test_partly_dry_history_requires_explicit_temporal_prior(reference):
    _, arrays = reference
    history = arrays["history_0"].copy()
    history[:2] = 0
    kwargs = dict(kmperpixel=1, timestep=5, n_ens_members=1, seed=4)
    with pytest.raises(ValueError, match="Unavailable spectral/persistence"):
        _forecast_array(history, arrays["velocity_0"], 2, **kwargs)
    result, diagnostics = _forecast_array(
        history,
        arrays["velocity_0"],
        2,
        reference_time=200,
        return_diagnostics=True,
        **kwargs,
    )
    assert np.isfinite(result).all()
    assert diagnostics["parameters"]["reference_time"] == 200
    assert np.isnan(diagnostics["estimated_reference_time_minutes"])
    assert all(
        len(levels) == 6 for levels in diagnostics["constant_observed_levels"][:2]
    )


def test_low_fraction_post_ar_suppression_is_explicit(reference):
    _, arrays = reference
    history = arrays["history_1"].copy()
    wet_indices = np.flatnonzero(history[-1] >= 1)
    history[-1].flat[wet_indices[200:]] = 0
    result, diagnostics = _forecast_array(
        history,
        arrays["velocity_1"],
        2,
        kmperpixel=1,
        timestep=7.5,
        n_ens_members=1,
        seed=42,
        return_diagnostics=True,
    )
    assert not result.any()
    assert 0 < diagnostics["statistics"]["rain_fraction"] < 0.025
    assert (
        diagnostics["status"] == "ok"
        and diagnostics["safeguards"]["low_rain_fraction_fallback"]
    )


@pytest.mark.parametrize("kind", ["constant", "single_pixel"])
def test_unestimable_wet_input_is_not_dry(kind):
    rain = np.zeros((3, 64, 64))
    if kind == "constant":
        rain[:] = 5
    else:
        rain[-1, 32, 32] = 5
    with pytest.raises(ValueError, match="at least two distinct"):
        _forecast_array(
            rain, np.zeros((2, 64, 64)), kmperpixel=1, timestep=5, n_ens_members=1
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timesteps": [1, 2]},
        {"timesteps": 0},
        {"n_ens_members": True},
        {"n_ens_members": 0},
        {"timestep": 0},
        {"kmperpixel": 0},
        {"rain_threshold": 0},
        {"scale_break": np.nan},
        {"zerovalue": 0},
        {"taper_sigma": -1},
        {"min_rain_fraction": 1.1},
        {"seed": -1},
        {"callback": 4},
        {"return_output": "yes"},
        {"n_cascade_levels": 2},
        {"alpha": 0},
        {"lower_truncation": -1},
        {"reference_levels": (6,)},
        {"reference_levels": (0, 0)},
        {"reference_time": 0},
        {"noise_kwargs": {"seed": 1}},
        {"noise_kwargs": {"eps": 0}},
        {"noise_kwargs": {"beta_variances": (0, 1)}},
        {"motion_kwargs": 3},
    ],
)
def test_invalid_options_even_when_dry(kwargs):
    options = dict(kmperpixel=1, timestep=5, n_ens_members=1)
    options.update(kwargs)
    with pytest.raises(ValueError):
        _forecast_array(np.zeros((3, 64, 64)), **options)


@pytest.mark.parametrize(
    "kind", ["shape", "integer", "nan", "negative", "mask", "complex", "list"]
)
def test_invalid_observations(kind):
    rain = np.zeros((3, 64, 64))
    if kind == "shape":
        rain = rain[:2]
    elif kind == "integer":
        rain = rain.astype(int)
    elif kind == "nan":
        rain[0, 0, 0] = np.nan
    elif kind == "negative":
        rain[0, 0, 0] = -1
    elif kind == "mask":
        rain = np.ma.array(rain, mask=False)
        rain.mask[0, 0, 0] = True
    elif kind == "complex":
        rain = rain.astype(complex)
    elif kind == "list":
        rain = rain.tolist()
    with pytest.raises(ValueError):
        _forecast_array(rain, kmperpixel=1, timestep=5)


def test_method_registration():
    assert nowcasts.get_method("parametric") is parametric.forecast


def test_explicit_estimation_settings_are_retained():
    options = dict(
        lower_truncation=1.5,
        temporal_exponent=0.9,
        alpha=2.5,
        reference_levels=(0, 1, 2),
        reference_time=250,
        moment_distance_limit=0.6,
        beta_1_bounds=(-3.8, -1.3),
        beta_2_min=-4.2,
        minimum_beta_gap=0.2,
        correlation_time_bounds=(50, 800),
        low_rain_fraction=0.04,
        fallback_beta_1=-1.5,
        fallback_beta_2=-3.2,
        fallback_reference_time=320,
    )
    original = deepcopy(options)
    _, diagnostics = _forecast_array(
        np.zeros((3, 8, 8)),
        n_ens_members=1,
        return_diagnostics=True,
        **options,
    )
    assert diagnostics["settings"]["estimation"] == options
    assert options == original


def test_reference_time_conflicts_with_temporal_fallback(reference):
    with pytest.raises(ValueError, match="cannot be combined with reference_time"):
        _call(
            reference,
            reference_time=200,
            temporal_fallback={"lzero": 256, "tzero": 420},
        )


def test_safeguards_moment_selection_and_joint_bounds():
    statistics = dict(
        ordinary_mean=6,
        ordinary_stdev=3,
        latent_mean=20,
        latent_stdev=3,
        rain_fraction=0.3,
    )
    spectrum = dict(beta_1=-1, beta_2=-1)
    _, diagnostics = _forecast_array(
        np.zeros((3, 8, 8)), n_ens_members=1, return_diagnostics=True
    )
    settings = diagnostics["settings"]["estimation"]
    parameters, audit = parametric._generation_parameters(
        statistics, spectrum, 20, settings
    )
    assert audit["moment_source"] == "ordinary" and audit["moment_reason"] == "distance"
    assert parameters["reference_time"] == 60
    assert parameters["beta_1"] < -1.25
    assert parameters["beta_2"] <= parameters["beta_1"] - 0.1
    assert spectrum == dict(beta_1=-1, beta_2=-1)
    statistics["latent_mean"] = np.nan
    _, audit = parametric._generation_parameters(statistics, spectrum, 200, settings)
    assert audit["moment_reason"] == "invalid_latent"


@pytest.mark.parametrize("invalid_pair", [[np.nan, np.nan], [1.0, 1.0], [0.2, -0.99]])
def test_configured_temporal_fallback(reference, monkeypatch, invalid_pair):
    def correlations(cascades, velocity):
        return np.tile(invalid_pair, (cascades.shape[1], 1))

    monkeypatch.setattr(parametric, "lagrangian_autocorrelation", correlations)
    result, diag = _call(
        reference,
        temporal_fallback={"lzero": 256, "tzero": 420},
        timestep=7.5,
        return_diagnostics=True,
    )
    expected_time = 420 * (diag["wavelengths_km"][0] / 256) ** 0.81
    assert diag["temporal_selection"]["source"] == "configured_fallback"
    assert diag["parameters"]["reference_time"] == pytest.approx(expected_time)
    np.testing.assert_allclose(
        diag["temporal_model"]["requested_times"],
        420 * (diag["wavelengths_km"] / 256) ** 0.81,
    )
    assert not diag["safeguards"]["low_rain_fraction_fallback"]
    assert np.isfinite(result).all()


def test_configured_fallback_preserves_valid_observation(reference):
    result, diag = _call(
        reference,
        temporal_fallback={"lzero": 256, "tzero": 420},
        return_diagnostics=True,
    )
    assert diag["temporal_selection"]["source"] == "observed"
    original = _call(reference)
    np.testing.assert_array_equal(result, original)


def test_temporal_fallback_does_not_replace_spatial_estimates(reference, monkeypatch):
    monkeypatch.setattr(
        parametric,
        "fit_power_law_spectrum",
        lambda *args: {"beta_1": np.nan, "beta_2": np.nan},
    )
    with pytest.raises(ValueError, match="Unavailable spectral"):
        _call(reference, temporal_fallback={"lzero": 256, "tzero": 420})


@pytest.mark.parametrize(
    "fallback", [{}, {"lzero": 0, "tzero": 420}, {"lzero": 256, "tzero": float("nan")}]
)
def test_invalid_temporal_fallback(reference, fallback):
    with pytest.raises(ValueError):
        _call(reference, temporal_fallback=fallback)


@pytest.mark.parametrize("index", [0, 1])
def test_dataset_coordinates_metadata_and_input_purity(reference, monkeypatch, index):
    metadata, arrays = reference
    seconds = metadata["cases"][index]["config"]["timestep"]
    dataset = _dataset(
        arrays[f"history_{index}"], arrays[f"velocity_{index}"], timestep=seconds / 60
    )
    original = dataset.copy(deep=True)

    def unexpected(*args, **kwargs):
        raise AssertionError("Supplied velocity must bypass motion estimation")

    monkeypatch.setattr(parametric.motion, "get_method", unexpected)
    output, diagnostics = parametric.forecast(
        dataset,
        3,
        n_ens_members=2,
        seed=metadata["cases"][index]["seed"],
        return_diagnostics=True,
    )
    xr.testing.assert_identical(dataset, original)
    field = output[output.attrs["precip_var"]]
    assert field.dims == ("ens_number", "time", "y", "x")
    assert field.dtype == np.float64
    expected_attrs = {**dataset.precip_intensity.attrs, "threshold": 1.0}
    expected_attrs.pop("transform")
    assert field.attrs == expected_attrs
    assert output.attrs == dataset.attrs
    assert diagnostics["motion_source"] == "supplied"
    assert diagnostics["settings"]["timestep_minutes"] == seconds / 60
    assert diagnostics["settings"]["kmperpixel"] == 1.0
    np.testing.assert_array_equal(output.ens_number, [1, 2])
    np.testing.assert_array_equal(
        output.time.values,
        dataset.time.values[-1] + np.arange(1, 4) * np.timedelta64(int(seconds), "s"),
    )
    assert output.forecast_reference_time.values == dataset.time.values[-1]
    xr.testing.assert_identical(output.x.variable, dataset.x.variable)
    xr.testing.assert_identical(output.y.variable, dataset.y.variable)
    assert output.time.attrs["stepsize"] == seconds
    assert "velocity_x" not in output and "velocity_y" not in output
    # Returned coordinates must not expose mutable aliases into the input.
    output.x.attrs["units"] = "changed on output only"
    xr.testing.assert_identical(dataset, original)


def test_dataset_dimension_order_and_coordinate_units(reference):
    _, arrays = reference
    dataset = _dataset(arrays["history_0"], arrays["velocity_0"])
    expected = parametric.forecast(dataset, 1, n_ens_members=1, seed=4)
    for axis in ("x", "y"):
        dataset[axis] = dataset[axis] / 1000
        dataset[axis].attrs.update(units="km", stepsize=1.0)
    dataset = dataset.transpose("x", "time", "y")
    actual = parametric.forecast(dataset, 1, n_ens_members=1, seed=4)
    np.testing.assert_array_equal(actual.precip_intensity, expected.precip_intensity)
    xr.testing.assert_identical(actual.x.variable, dataset.x.variable)
    xr.testing.assert_identical(actual.y.variable, dataset.y.variable)


def test_descending_y_and_dry_forecast_metadata():
    dataset = _dataset(np.zeros((3, 8, 9)), timestep=7.5)
    dataset = dataset.isel(y=slice(None, None, -1))
    dataset.y.attrs["stepsize"] = -1000.0
    output = parametric.forecast(dataset, 2, n_ens_members=1)
    xr.testing.assert_identical(output.y.variable, dataset.y.variable)
    assert not output.precip_intensity.values.any()
    assert "transform" not in output.precip_intensity.attrs
    np.testing.assert_array_equal(
        np.diff(output.time.values) / np.timedelta64(1, "s"), [450]
    )


@pytest.mark.parametrize("overrides", [{"kmperpixel": 2}, {"timestep": 5}])
def test_explicit_geometry_must_agree_with_dataset(overrides):
    dataset = _dataset(np.zeros((3, 8, 9)), timestep=7.5)
    with pytest.raises(ValueError, match="disagrees with dataset coordinates"):
        parametric.forecast(dataset, **overrides)


@pytest.mark.parametrize(
    "kind, message",
    [
        ("array", "xarray.Dataset"),
        ("precip_var", "precip_var"),
        ("units", "untransformed rain rates"),
        ("transform", "untransformed rain rates"),
        ("integer", "floating point"),
        ("missing", "finite nonnegative"),
        ("negative", "finite nonnegative"),
        ("coordinate_units", "coordinate units"),
        ("irregular_x", "regularly spaced"),
        ("square", "square pixels"),
        ("spatial_stepsize", "stepsize disagrees"),
        ("missing_coordinate", "coordinate is required"),
        ("time_stepsize", "spacing in seconds"),
        ("irregular_time", "increasing and regularly spaced"),
        ("backwards_time", "increasing and regularly spaced"),
        ("nat", "valid datetime64"),
        ("numeric_time", "valid datetime64"),
        ("partial_velocity", "both velocity_x and velocity_y"),
        ("missing_velocity", "finite real"),
        ("velocity_units", "pixels per observation timestep"),
        ("velocity_time_dimension", r"dimensions \(y, x\)"),
    ],
)
def test_dataset_rejects_invalid_physical_inputs_before_dry_shortcut(kind, message):
    dataset = _dataset(np.zeros((3, 8, 9)), np.zeros((2, 8, 9)))
    if kind == "array":
        dataset = dataset.precip_intensity.values
    elif kind == "precip_var":
        dataset.attrs["precip_var"] = "absent"
    elif kind in ("units", "transform"):
        dataset.precip_intensity.attrs[kind] = "dB" if kind == "transform" else "mm"
    elif kind == "integer":
        dataset["precip_intensity"] = dataset.precip_intensity.astype(int)
    elif kind in ("missing", "negative"):
        dataset.precip_intensity.values[0, 0, 0] = np.nan if kind == "missing" else -1
    elif kind == "coordinate_units":
        dataset.x.attrs["units"] = "degrees_east"
    elif kind == "irregular_x":
        values = dataset.x.values.copy()
        values[2] += 100
        dataset = dataset.assign_coords(x=("x", values, dataset.x.attrs))
    elif kind == "square":
        dataset["y"] = dataset.y * 2
        dataset.y.attrs.update(units="m", stepsize=2000)
    elif kind == "spatial_stepsize":
        dataset.y.attrs["stepsize"] = -1000
    elif kind == "missing_coordinate":
        dataset = dataset.drop_vars("x")
    elif kind == "time_stepsize":
        dataset.time.attrs["stepsize"] = 5  # Minutes must not be read as seconds.
    elif kind == "irregular_time":
        values = dataset.time.values.copy()
        values[1] += np.timedelta64(1, "s")
        dataset = dataset.assign_coords(time=("time", values, dataset.time.attrs))
    elif kind == "backwards_time":
        dataset = dataset.isel(time=slice(None, None, -1))
    elif kind == "nat":
        values = dataset.time.values.copy()
        values[0] = np.datetime64("NaT", "ns")
        dataset = dataset.assign_coords(time=("time", values, dataset.time.attrs))
    elif kind == "numeric_time":
        dataset = dataset.assign_coords(time=[0, 300, 600])
    elif kind == "partial_velocity":
        dataset = dataset.drop_vars("velocity_y")
    elif kind == "missing_velocity":
        dataset.velocity_x.values[0, 0] = np.nan
    elif kind == "velocity_units":
        dataset.velocity_x.attrs["units"] = "m/s"
    elif kind == "velocity_time_dimension":
        dataset["velocity_x"] = (("time", "y", "x"), np.zeros((3, 8, 9)))
    with pytest.raises(ValueError, match=message):
        parametric.forecast(dataset, n_ens_members=1)


@pytest.mark.parametrize("case", [0, 1, "dry"])
def test_forecast_native_netcdf_roundtrip(reference, tmp_path, case):
    """Exercise actual importer metadata and lossless CF decoding at both cadences."""
    from datetime import datetime

    from pysteps.xarray_helpers import convert_input_to_xarray_dataset

    metadata, arrays = reference
    index = 0 if case == "dry" else case
    seconds = 450 if case == "dry" else metadata["cases"][index]["config"]["timestep"]
    rain = arrays[f"history_{index}"].copy()
    if case == "dry":
        rain.fill(0)
    height, width = rain.shape[-2:]
    dataset = convert_input_to_xarray_dataset(
        rain,
        None,
        {
            "unit": "mm/h",
            "threshold": 0.1,
            "zerovalue": 0.0,
            "accutime": seconds / 60,
            "projection": _dataset(rain).attrs["projection"] + " +x_0=0 +y_0=0",
            "institution": "reference fixture",
            "x1": 0,
            "x2": width * 1000,
            "y1": 0,
            "y2": height * 1000,
            "xpixelsize": 1000,
            "ypixelsize": 1000,
            "cartesian_unit": "m",
            "yorigin": "upper",
        },
        startdate=datetime(2026, 1, 1),
        timestep=seconds,
    )
    dataset["velocity_x"] = (("y", "x"), arrays[f"velocity_{index}"][0])
    dataset["velocity_y"] = (("y", "x"), arrays[f"velocity_{index}"][1])
    original = dataset.copy(deep=True)
    output = parametric.forecast(dataset, 3, n_ens_members=2, seed=4)
    precip_var = output.attrs["precip_var"]
    output_path = tmp_path / "forecast.nc"
    output.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding={precip_var: {"zlib": True, "complevel": 1}},
    )
    with xr.open_dataset(output_path, engine="netcdf4") as reopened:
        loaded = reopened.load()
    xr.testing.assert_identical(output, loaded)
    xr.testing.assert_identical(dataset, original)
    np.testing.assert_array_equal(output[precip_var], loaded[precip_var])
    assert loaded[precip_var].dtype == np.float64
    assert loaded.time.attrs["stepsize"] == seconds
    assert "transform" not in loaded[precip_var].attrs
    assert "lat" in loaded.coords and "lon" in loaded.coords
    assert loaded[precip_var].attrs["grid_mapping"] in loaded.coords
    assert "velocity_x" not in loaded and "velocity_y" not in loaded
