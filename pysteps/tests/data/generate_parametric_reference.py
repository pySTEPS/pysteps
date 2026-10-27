"""Regenerate numerical references from the two original source repositories.

Run in the source project's environment (including xarray):
python generate_parametric_reference.py /path/to/pysteps_param /path/to/psteps_param_paper

Tests read only the generated JSON and do not need either external repository.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import brentq, curve_fit, minimize_scalar
from scipy.special import log_ndtr

from pysteps import extrapolation
from pysteps.cascade.bandpass_filters import filter_gaussian


def _load_functions(path, names, namespace):
    # Load the original numerical definitions without executing package-level
    # imports of database or operational modules.
    source = path.read_text()
    definitions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != set(names):
        raise ValueError(f"Missing source functions in {path}")
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return hashlib.sha256(source.encode()).hexdigest()


def main():
    import xarray as xr
    import scipy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_repo", type=Path)
    parser.add_argument("paper_repo", type=Path)
    args = parser.parse_args()
    source = args.parameter_repo / "src/pysteps_param"
    namespace = dict(
        np=np,
        xr=xr,
        curve_fit=curve_fit,
        minimize_scalar=minimize_scalar,
        log_ndtr=log_ndtr,
        brentq=brentq,
        extrapolation=extrapolation,
    )
    hashes = {}
    hashes["pysteps_param/src/pysteps_param/rain_stats_compute.py"] = _load_functions(
        source / "rain_stats_compute.py",
        {
            "power_spectrum_1D",
            "fit_lower_truncated_normal",
            "compute_field_stats",
            "correlation_length",
            "lagr_auto_cor",
        },
        namespace,
    )
    hashes["pysteps_param/src/pysteps_param/shared_utils.py"] = _load_functions(
        source / "shared_utils.py", {"fit_auto_cors", "_adjust_model_lag2"}, namespace
    )
    hashes["psteps_param_paper/workflows/reference_timescale_experiment.py"] = (
        _load_functions(
            args.paper_repo / "workflows/reference_timescale_experiment.py",
            {"reference_log"},
            namespace,
        )
    )
    result = dict(
        provenance=dict(
            source_revisions={
                name: subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip()
                for name, root in [
                    ("pysteps_param", args.parameter_repo),
                    ("psteps_param_paper", args.paper_repo),
                ]
            },
            source_sha256=hashes,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            description="Numerical references evaluated from original function definitions on deterministic synthetic inputs; no database access.",
        ),
        spectra=[],
        ar2_inverse=[],
    )
    for shape in [(128, 128), (95, 127)]:
        y, x = np.indices(shape)
        field = np.random.default_rng(23).normal(size=shape) + 4 * np.cos(
            2 * np.pi * (5 * y / shape[0] + 3 * x / shape[1])
        )
        models = []
        for scale_break in (None, 20):
            spectrum, model = namespace["power_spectrum_1D"](field, scale_break)
            models.append(model)
        result["spectra"].append(
            dict(
                shape=shape,
                frequency_db=spectrum.psd_bins.values.tolist(),
                power_db=spectrum.psd.values.tolist(),
                models=models,
            )
        )
    rng = np.random.default_rng(37)
    field = rng.normal(4, 3, size=(128, 96))
    field[rng.uniform(size=field.shape) > 0.3] = -0.1
    stats = namespace["compute_field_stats"](
        field,
        {"threshold": 0, "zerovalue": -0.1, "transform": "dB"},
        fit_truncated=True,
        lower_truncation=1.25,
    )
    result["marginal"] = {
        name: stats[name]
        for name in [
            "ordinary_mean",
            "ordinary_stdev",
            "latent_mean",
            "latent_stdev",
            "mean",
            "stdev",
        ]
    }
    result["marginal"]["rain_fraction"] = stats["nonzero_fraction"]
    history = np.random.default_rng(45).normal(size=(3, 2, 24, 32))
    velocity = np.full((2, 24, 32), 0.25)
    velocity[0] = 0.6
    result["lags"] = namespace["lagr_auto_cor"](history, velocity).tolist()
    for alpha in (0.8, 2.72, 4):
        for time in (5, 10, 60, 420):
            r1, r2, _ = namespace["fit_auto_cors"](time, alpha, 5)
            result["ar2_inverse"].append(dict(time=time, alpha=alpha, pair=[r1, r2]))
    wavelengths = 1 / filter_gaussian((256, 256), 6, d=1)["central_freqs"]
    times = np.array([[60, 42, 25, 10, 8, 6], [120, 70, 31, 20, 10, 5]])
    exponent = 0.7718612877449645
    result["paper_reference"] = dict(
        wavelengths=wavelengths.tolist(),
        times=times.tolist(),
        exponent=exponent,
        reference_times=np.exp(
            namespace["reference_log"](times, wavelengths, exponent)
        ).tolist(),
    )
    path = Path(__file__).with_name("parametric_estimators_reference.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
