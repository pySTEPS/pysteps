"""Freeze original generator outputs on synthetic targets without database access.

Run with the source project's dependencies (including xarray):
python generate_calibrated_reference.py /path/to/pysteps_param

Normal tests read only the JSON and NPZ fixtures, not the source repository.
"""

import argparse
import ast
from dataclasses import dataclass, make_dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Optional

import numpy as np
from scipy import interpolate, stats
from scipy.optimize import curve_fit

from generate_parametric_reference import _load_functions


def source_generator(parameter_repo):
    """Load the actual source functions, result class, and constants.

    Supply a five-scalar dataclass in place of the operational StepsParameters.
    The numerical functions access only these five attributes; replace() retains
    its original semantics. No QA selection or database identity is introduced.
    """
    import xarray as xr

    source = parameter_repo / "src/pysteps_param"
    parameters = make_dataclass(
        "ScalarParameters",
        [
            (name, float)
            for name in ("beta_1", "beta_2", "rain_fraction", "mean_db", "stdev_db")
        ],
    )
    namespace = dict(
        __name__=__name__,
        np=np,
        xr=xr,
        stats=stats,
        interpolate=interpolate,
        curve_fit=curve_fit,
        dataclass=dataclass,
        replace=replace,
        Optional=Optional,
        StepsParameters=parameters,
    )
    stats_hash = _load_functions(
        source / "rain_stats_compute.py", {"power_spectrum_1D"}, namespace
    )
    path = source / "stochastic_generator.py"
    original = path.read_text()
    nodes = [
        node
        for node in ast.parse(original).body
        if isinstance(node, ast.FunctionDef)
        or isinstance(node, ast.ClassDef)
        and node.name == "CalibratedStochasticField"
        or isinstance(node, ast.Assign)
        and all(
            isinstance(target, ast.Name) and target.id.isupper()
            for target in node.targets
        )
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    provenance = dict(
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=parameter_repo, text=True
        ).strip(),
        source_sha256={
            "rain_stats_compute.py": stats_hash,
            "stochastic_generator.py": hashlib.sha256(original.encode()).hexdigest(),
        },
    )
    return namespace, provenance


def main():
    import scipy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_repo", type=Path)
    args = parser.parse_args()
    source, provenance = source_generator(args.parameter_repo)
    provenance.update(numpy_version=np.__version__, scipy_version=scipy.__version__)
    metadata = dict(provenance=provenance, cases=[])
    arrays = {}
    cases = [
        ((128, 96), 0.08, 4, {}),
        ((128, 96), 0.3, 17, {}),
        ((95, 127), 0.05, 11, {"distance_threshold": 0.01}),
    ]
    for index, (shape, fraction, seed, settings) in enumerate(cases):
        parameters = dict(
            beta_1=-2.4, beta_2=-3.1, rain_fraction=fraction, mean_db=6.0, stdev_db=3.5
        )
        target = source["StepsParameters"](**parameters)
        result = source["gen_calibrated_stoch_field"](
            target,
            shape[1],
            shape[0],
            1.0,
            20.0,
            0.0,
            rng=np.random.default_rng(seed),
            **settings,
        )
        arrays[f"calibrated_{index}"] = result.field
        arrays[f"uncalibrated_{index}"] = source["gen_stoch_field"](
            target, shape[1], shape[0], 1.0, 20.0, 0.0, rng=np.random.default_rng(seed)
        )
        metadata["cases"].append(
            dict(
                shape=shape,
                parameters=parameters,
                seed=seed,
                kmperpixel=1.0,
                scale_break=20.0,
                threshold=0.0,
                settings=settings,
                diagnostics={
                    name: getattr(result, name)
                    for name in (
                        "estimated_beta_1",
                        "estimated_beta_2",
                        "distance",
                        "attempts",
                        "selected_attempt",
                        "tolerance_achieved",
                    )
                },
            )
        )
    target = source["StepsParameters"](
        beta_1=-2.0, beta_2=-3.0, rain_fraction=0.35, mean_db=6.5, stdev_db=1.2
    )
    arrays["post_ar_input"] = np.random.default_rng(19).normal(size=(48, 64))
    arrays["post_ar_output"] = source["normalize_db_field"](
        arrays["post_ar_input"], target, 0.0, -0.1
    )
    metadata["post_ar_parameters"] = dict(
        rain_fraction=0.35, mean=6.5, stdev=1.2, threshold=0.0, zerovalue=-0.1
    )
    prefix = Path(__file__).with_name("calibrated_generator_reference")
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    prefix.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(prefix)


if __name__ == "__main__":
    main()
