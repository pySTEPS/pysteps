"""Compare complete forecasts on frozen three-observation paper histories.

The archives store [current, previous, future1, ...]; select [1, 0, 2] as
chronological observations and estimate motion anew for the latest pair.
Nothing is read from a database or written to the research repository.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.ndimage import gaussian_filter

from pysteps import motion
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.nowcasts import parametric
from pysteps.nowcasts._cascade_advection import (
    advect_field,
    prepare_advection_coordinates,
)
from pysteps.timeseries.autoregression import estimate_ar_params_yw
from pysteps.timeseries.parametric import lagrangian_autocorrelation
from pysteps.utils.spectral import rapsd_logarithmic, fit_power_law_spectrum

from generate_nowcast_reference import source_engine, config, prepare, evolve


def fixed_parameter_forecast(rain, velocity, cfg, parameters, seed, n_members, n_leads):
    """Isolate evolution using the source's selected parameters and lag pairs."""
    bp = filter_gaussian(rain.shape[1:], 6)
    transformed = 10 * np.log10(np.maximum(rain, 1))
    transformed[rain < 1] = -0.1
    observed = [parametric._decompose(f, bp) for f in transformed]
    coordinates = prepare_advection_coordinates(velocity)
    taper = gaussian_filter(advect_field(np.ones(rain.shape[1:]), coordinates[1]), 5)
    target = {
        k: getattr(parameters, k)
        for k in ("beta_1", "beta_2", "rain_fraction", "mean_db", "stdev_db")
    }
    phi = np.array(
        [
            estimate_ar_params_yw(pair)
            for pair in zip(parameters.lag_1, parameters.lag_2)
        ]
    )
    output = np.empty((n_members, n_leads, *rain.shape[1:]))
    for member, child in enumerate(np.random.SeedSequence(seed).spawn(n_members)):
        rng = np.random.default_rng(child)
        histories = [observed[-1], observed[-2]]
        for lead in range(n_leads):
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
            output[member, lead] = physical
            histories = [parametric._decompose(field * taper, bp), histories[0]]
    return output


def metrics(forecast, observations, velocity):
    bp = filter_gaussian(forecast.shape[-2:], 6)
    result = []
    for member in forecast:
        history = []
        for field in observations[-2:]:
            db = 10 * np.log10(np.maximum(field, 1))
            db[field < 1] = -0.1
            history.append(parametric._decompose(db, bp)["cascade_levels"])
        for field in member:
            wet = field >= 1
            db = 10 * np.log10(np.maximum(field, 1))
            db[~wet] = -0.1
            frequency, power = rapsd_logarithmic(db)
            slopes = fit_power_law_spectrum(frequency, power, 20)
            history.append(parametric._decompose(db, bp)["cascade_levels"])
            correlations = lagrangian_autocorrelation(np.stack(history[-3:]), velocity)
            result.append(
                [
                    float(wet.mean()),
                    float(field.mean()),
                    float(db[wet].mean()) if wet.any() else np.nan,
                    float(db[wet].std()) if wet.any() else np.nan,
                    slopes["beta_1"],
                    slopes["beta_2"],
                    *correlations.ravel(),
                ]
            )
    return np.asarray(result)


def finite_max(values):
    values = np.asarray(values)
    return (
        float(np.max(values[np.isfinite(values)]))
        if np.isfinite(values).any()
        else None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_repo", type=Path)
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leads", type=int, default=12)
    args = parser.parse_args()
    source, provenance = source_engine(args.parameter_repo)
    import scipy

    report = dict(
        provenance=provenance,
        paper_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.paper_repo, text=True
        ).strip(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        leads=args.leads,
        members=2,
        seeds=[0, 20261015],
        observation_policy="Archive indices [1,0,2], oldest first; newest is original cycle +5 minutes",
        motion_policy="Reference Lucas-Kanade settings, re-estimated from latest pair",
        config=dict(config((256, 256)), n_forecasts=args.leads),
        comparisons=[],
    )
    for name in ("control_1", "rain_fraction_low_2", "beta_1_low_2"):
        folder = args.paper_repo / "figs/extreme_parameter_experiment" / name
        path = folder / "inputs.npz"
        with np.load(path) as saved:
            rain = saved["rain"][[1, 0, 2]].astype(float)
        metadata = json.loads((folder / "input_metadata.json").read_text())
        transformed = source["DBTransformer"](threshold=1).transform(rain)
        velocity = motion.get_method("LK")(
            transformed[-2:], **parametric._MOTION_DEFAULTS
        )
        cfg = config(rain.shape[1:])
        cfg["n_forecasts"] = args.leads
        parameters, _, _, _ = prepare(source, rain, velocity, cfg)
        for seed in report["seeds"]:
            original, calibration = evolve(
                source,
                rain,
                velocity,
                cfg,
                parameters,
                seed,
                n_members=2,
                n_leads=args.leads,
            )
            fixed = fixed_parameter_forecast(
                rain, velocity, cfg, parameters, seed, 2, args.leads
            )
            # Exercise the public call including its internal motion estimation.
            coupled, diagnostics = parametric.forecast(
                rain,
                timesteps=args.leads,
                n_ens_members=2,
                kmperpixel=1,
                timestep=5,
                seed=seed,
                return_diagnostics=True,
            )
            np.testing.assert_array_equal(velocity, diagnostics["velocity"])
            for key in ("beta_1", "beta_2", "rain_fraction", "mean_db", "stdev_db"):
                np.testing.assert_allclose(
                    diagnostics["parameters"][key],
                    getattr(parameters, key),
                    atol=1e-6,
                    rtol=0,
                )
            np.testing.assert_allclose(
                diagnostics["parameters"]["reference_time"],
                parameters.corl_zero,
                atol=1e-7,
                rtol=0,
            )
            np.testing.assert_allclose(
                diagnostics["model_correlations"],
                np.array([parameters.lag_1, parameters.lag_2]).T,
                atol=1e-12,
                rtol=0,
            )
            np.testing.assert_allclose(fixed, original, atol=1e-8, rtol=1e-10)
            difference = abs(coupled - original)
            rms = float(np.sqrt(np.mean((coupled - original) ** 2)))
            original_metrics = metrics(original, rain, velocity)
            coupled_metrics = metrics(coupled, rain, velocity)
            deltas = abs(coupled_metrics - original_metrics)
            assert np.array_equal(
                np.isfinite(original_metrics), np.isfinite(coupled_metrics)
            )
            assert np.isfinite(coupled).all() and np.all(coupled >= 0)
            selection_changes = sum(
                diagnostics["generation"][m][lead]["selected_attempt"]
                != calibration[m][lead]["selected_attempt"]
                for m in range(2)
                for lead in range(args.leads)
            )
            report["comparisons"].append(
                dict(
                    case=name,
                    seed=seed,
                    input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    observation_times=[
                        metadata["rain"][i]["valid_time"] for i in (1, 0, 2)
                    ],
                    parameters=diagnostics["parameters"],
                    source_parameters=parameters.active_parameters(),
                    moment_source=diagnostics["safeguards"]["moment_source"],
                    low_coverage_fallback=diagnostics["safeguards"][
                        "low_rain_fraction_fallback"
                    ],
                    identical_parameters_maximum_error_mm_per_hour=float(
                        abs(fixed - original).max()
                    ),
                    coupled_maximum_error_mm_per_hour=float(difference.max()),
                    coupled_rmse_mm_per_hour=rms,
                    rain_classification_changes=int(
                        np.count_nonzero((coupled >= 1) != (original >= 1))
                    ),
                    total_output_pixels=int(coupled.size),
                    candidate_selection_changes=selection_changes,
                    maximum_wet_fraction_difference=finite_max(deltas[:, 0]),
                    maximum_mean_rain_difference=finite_max(deltas[:, 1]),
                    maximum_wet_dbr_mean_difference=finite_max(deltas[:, 2]),
                    maximum_wet_dbr_stdev_difference=finite_max(deltas[:, 3]),
                    maximum_spectral_slope_difference=finite_max(deltas[:, 4:6]),
                    maximum_lagrangian_correlation_difference=finite_max(deltas[:, 6:]),
                )
            )
            print(
                f"{name}, seed {seed}: fixed max {abs(fixed-original).max():.3g}, coupled max {difference.max():.3g}, RMSE {rms:.3g}"
            )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
