"""Check original and ported generator outputs on saved paper rainfall targets.

Requires the source project's dependencies, including xarray. Inputs and outputs
are local files; no database is accessed. Raw estimates are passed to both
generators without regional generation QA, which belongs to the engine stage.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

from pysteps.noise.parametric import generate_calibrated_field
from pysteps.utils.spectral import fit_power_law_spectrum, rapsd_logarithmic
from pysteps.utils.statistics import compute_field_statistics
from pysteps.utils.transformation import dB_transform
from generate_calibrated_reference import source_generator
from generate_parametric_reference import _load_functions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_repo", type=Path)
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, provenance = source_generator(args.parameter_repo)
    source.update(minimize_scalar=minimize_scalar, log_ndtr=log_ndtr)
    _load_functions(
        args.parameter_repo / "src/pysteps_param/rain_stats_compute.py",
        {"fit_lower_truncated_normal", "compute_field_stats"},
        source,
    )
    report = dict(
        provenance=provenance,
        paper_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.paper_repo, text=True
        ).strip(),
        seeds=[0, 20260908],
        scale_break_km=15.0,
        pixel_size_km=1.0,
        threshold_mm_per_hour=1.0,
        lower_truncation_dbr=1.25,
        target_policy="Raw estimator outputs; regional generation QA not applied",
        coupled_maximum_absolute_error_tolerance_dbr=1e-4,
        cases=[],
    )
    paths = sorted((args.paper_repo / "figs/individual_field_fits").glob("*.npz"))
    if not paths:
        raise ValueError("No saved paper fields found")
    for path in paths:
        with np.load(path) as stored:
            rain = stored["rain"].astype(float)
        transformed, metadata = dB_transform(rain, threshold=1.0, zerovalue=-0.1)
        _, spectrum = source["power_spectrum_1D"](transformed, 15)
        marginal = source["compute_field_stats"](
            transformed, metadata, fit_truncated=True, lower_truncation=1.25
        )
        parameters = dict(
            beta_1=spectrum["beta_1"],
            beta_2=spectrum["beta_2"],
            rain_fraction=marginal["nonzero_fraction"],
            mean_db=marginal["latent_mean"],
            stdev_db=marginal["latent_stdev"],
        )
        frequency, power = rapsd_logarithmic(transformed)
        port_spectrum = fit_power_law_spectrum(frequency, power, 15)
        port_marginal = compute_field_statistics(
            transformed, metadata["threshold"], 1.25
        )
        port_parameters = dict(
            beta_1=port_spectrum["beta_1"],
            beta_2=port_spectrum["beta_2"],
            rain_fraction=port_marginal["rain_fraction"],
            mean_db=port_marginal["latent_mean"],
            stdev_db=port_marginal["latent_stdev"],
        )
        case = dict(
            name=path.stem,
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            parameters=parameters,
            comparisons=[],
        )
        for seed in report["seeds"]:
            original = source["gen_calibrated_stoch_field"](
                source["StepsParameters"](**parameters),
                rain.shape[1],
                rain.shape[0],
                1,
                15,
                metadata["threshold"],
                rng=np.random.default_rng(seed),
                zerovalue=metadata["zerovalue"],
            )
            fixed, fixed_diagnostics = generate_calibrated_field(
                rain.shape,
                parameters,
                1,
                15,
                metadata["threshold"],
                seed=seed,
                zerovalue=metadata["zerovalue"],
                return_diagnostics=True,
            )
            coupled, coupled_diagnostics = generate_calibrated_field(
                rain.shape,
                port_parameters,
                1,
                15,
                metadata["threshold"],
                seed=seed,
                zerovalue=metadata["zerovalue"],
                return_diagnostics=True,
            )
            np.testing.assert_allclose(fixed, original.field, atol=1e-10, rtol=1e-12)
            # Equivalent spectral regression solvers produce slightly different
            # slopes. Histogram interpolation can amplify those differences.
            # Check a 1e-4 dBR absolute limit (about 2.3e-5 relative rain rate)
            # separately from the tight identical-target generator comparison.
            np.testing.assert_allclose(coupled, original.field, atol=1e-4, rtol=0)
            for diagnostics in (fixed_diagnostics, coupled_diagnostics):
                for name in ("attempts", "selected_attempt", "tolerance_achieved"):
                    assert diagnostics[name] == getattr(original, name)
                for name in ("estimated_beta_1", "estimated_beta_2", "distance"):
                    np.testing.assert_allclose(
                        diagnostics[name], getattr(original, name), atol=3e-5, rtol=1e-7
                    )
            case["comparisons"].append(
                dict(
                    seed=seed,
                    attempts=original.attempts,
                    selected_attempt=original.selected_attempt,
                    tolerance_achieved=original.tolerance_achieved,
                    distance=original.distance,
                    identical_targets_maximum_field_difference=float(
                        np.max(abs(fixed - original.field))
                    ),
                    coupled_estimators_maximum_field_difference=float(
                        np.max(abs(coupled - original.field))
                    ),
                    coupled_distance_difference=float(
                        abs(coupled_diagnostics["distance"] - original.distance)
                    ),
                    coupled_threshold_classification_changes=int(
                        np.count_nonzero(
                            (coupled > metadata["threshold"])
                            != (original.field > metadata["threshold"])
                        )
                    ),
                )
            )
        report["cases"].append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Compared {len(paths)} paper cases at {len(report['seeds'])} seeds; report: {args.output}"
    )


if __name__ == "__main__":
    main()
