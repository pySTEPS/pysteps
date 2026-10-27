"""Freeze the original preparation, QA, and multi-lead forecast numerics.

Run in the source project's environment:
python generate_nowcast_reference.py /path/to/pysteps_param

No database is accessed. AST extraction loads original numerical definitions;
only a relative import inside QA is removed to resolve the extracted helpers.
Reference independent streams are supplied explicitly to the original generator.
"""

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, ClassVar, Mapping, TypedDict, cast
import warnings
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

from pysteps import extrapolation
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.timeseries.autoregression import (
    estimate_ar_params_yw,
    adjust_lag2_corrcoef2,
)

from generate_calibrated_reference import source_generator
from generate_parametric_reference import _load_functions


def source_engine(parameter_repo):
    namespace, provenance = source_generator(parameter_repo)
    source = parameter_repo / "src/pysteps_param"
    namespace.update(
        __file__=str(source / "steps_params.py"),
        deepcopy=deepcopy,
        copy=__import__("copy"),
        np=np,
        Path=Path,
        json=json,
        dataclass=dataclass,
        field=field,
        ClassVar=ClassVar,
        Mapping=Mapping,
        warnings=warnings,
        Any=Any,
        TypedDict=TypedDict,
        cast=cast,
        minimize_scalar=minimize_scalar,
        log_ndtr=log_ndtr,
        extrapolation=extrapolation,
        map_coordinates=map_coordinates,
        estimate_ar_params_yw=estimate_ar_params_yw,
        adjust_lag2_corrcoef2=adjust_lag2_corrcoef2,
        filter_gaussian=filter_gaussian,
        decomposition_fft=decomposition_fft,
    )
    from pysteps.cascade.decomposition import recompose_fft

    namespace["recompose_fft"] = recompose_fft
    for filename, names in [
        (
            "rain_stats_compute.py",
            {
                "compute_field_stats",
                "fit_lower_truncated_normal",
                "correlation_length",
                "lagr_auto_cor",
            },
        ),
        ("cascade_utils.py", {"calculate_wavelengths"}),
        (
            "shared_utils.py",
            {
                "update_field",
                "prepare_advection_coordinates",
                "_adjust_model_lag2",
                "fit_auto_cors",
                "calc_model_auto_cors",
                "model_correlation_diagnostics",
            },
        ),
    ]:
        provenance["source_sha256"][filename] = _load_functions(
            source / filename, names, namespace
        )
    for filename, wanted in [
        ("model_config.py", None),
        ("transformer.py", {"BaseTransformer", "DBTransformer"}),
        ("steps_params.py", {"StepsParameters"}),
    ]:
        path = source / filename
        tree = ast.parse(path.read_text())
        if wanted is not None:
            tree.body = [
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name in wanted
            ]
            # QA's relative import resolves to the same original functions above.
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    node.body = [
                        item
                        for item in node.body
                        if not isinstance(item, ast.ImportFrom)
                    ]
        exec(compile(tree, str(path), "exec"), namespace)
        provenance["source_sha256"][filename] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    namespace["get_transformer"] = lambda name, threshold: namespace["DBTransformer"](
        threshold=threshold
    )
    namespace["CORRELATION_TIME_DEFINITION"] = "ar_acf_sum_including_zero_v2"
    provenance["source_sha256"]["cli/nowcast.py"] = hashlib.sha256(
        (source / "cli/nowcast.py").read_bytes()
    ).hexdigest()
    provenance["source_sha256"]["config/low_war_fallback.json"] = hashlib.sha256(
        (source / "config/low_war_fallback.json").read_bytes()
    ).hexdigest()
    import scipy

    provenance.update(
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        native_pysteps_sha256={
            name: hashlib.sha256(
                Path(inspect.getsourcefile(function)).read_bytes()
            ).hexdigest()
            for name, function in (
                ("cascade/decomposition.py", decomposition_fft),
                ("cascade/bandpass_filters.py", filter_gaussian),
                ("timeseries/autoregression.py", estimate_ar_params_yw),
                (
                    "extrapolation/semilagrangian.py",
                    extrapolation.get_method("semilagrangian"),
                ),
            )
        },
    )
    return namespace, provenance


def config(shape, timestep=5.0, kmperpixel=1.0, scale_break=20.0):
    return dict(
        n_rows=shape[0],
        n_cols=shape[1],
        n_cascade_levels=6,
        n_forecasts=3,
        n_ensembles=2,
        p_size=kmperpixel * 1000,
        ht=0.81,
        alpha=2.72,
        timestep=timestep * 60,
        ar_order=2,
        rain_threshold=1.0,
        scale_break_km=scale_break,
        transform="dbr",
    )


def prepare(source, rain, velocity, cfg):
    transformer = source["DBTransformer"](threshold=cfg["rain_threshold"])
    transformed = transformer.transform(rain)
    bp_filter = filter_gaussian(rain.shape[1:], cfg["n_cascade_levels"])
    cascades = [
        decomposition_fft(f, bp_filter, compute_stats=True, normalize=True)
        for f in transformed
    ]
    marginal = source["compute_field_stats"](
        transformed[-1], transformer.metadata, fit_truncated=True, lower_truncation=1.25
    )
    _, spectrum = source["power_spectrum_1D"](
        transformed[-1], cfg["scale_break_km"] / (cfg["p_size"] / 1000)
    )
    correlations = source["lagr_auto_cor"](
        np.stack([c["cascade_levels"] for c in cascades]), velocity
    )
    reference_time = source["correlation_length"](
        *correlations[0], cfg["timestep"] / 60
    )
    parameters = source["StepsParameters"](
        metadata={},
        ordinary_mean_db=marginal["ordinary_mean"],
        ordinary_stdev_db=marginal["ordinary_stdev"],
        latent_mean_db=marginal["latent_mean"],
        latent_stdev_db=marginal["latent_stdev"],
        rain_fraction=marginal["nonzero_fraction"],
        beta_1=spectrum["beta_1"],
        beta_2=spectrum["beta_2"],
        corl_zero=reference_time,
    ).apply_qa(cfg)
    return parameters, cascades, bp_filter, transformer


def evolve(source, rain, velocity, cfg, parameters, seed, n_members=2, n_leads=3):
    bp_filter = filter_gaussian(rain.shape[1:], cfg["n_cascade_levels"])
    transformer = source["DBTransformer"](threshold=cfg["rain_threshold"])
    init = [
        decomposition_fft(
            transformer.transform(f), bp_filter, compute_stats=True, normalize=True
        )
        for f in rain[-1:-3:-1]
    ]
    coordinates = source["prepare_advection_coordinates"](velocity)
    taper = gaussian_filter(
        extrapolation.get_method("semilagrangian")(
            np.ones(rain.shape[1:]), velocity, 1, outval=0
        )[0],
        5.0,
    )
    seeds = np.random.SeedSequence(seed).spawn(n_members)
    original_generator = source["gen_calibrated_stoch_field"]
    output = np.empty((n_members, n_leads, *rain.shape[1:]))
    calibration = [[] for _ in range(n_members)]
    try:
        for member in range(n_members):
            histories = deepcopy(init)
            rng = np.random.default_rng(seeds[member])

            def seeded_generator(*args, **kwargs):
                result = original_generator(*args, **kwargs, rng=rng)
                calibration[member].append(
                    {k: v for k, v in vars(result).items() if k != "field"}
                )
                return result

            source["gen_calibrated_stoch_field"] = seeded_generator
            for lead in range(n_leads):
                field = source["update_field"](
                    histories,
                    velocity,
                    parameters,
                    bp_filter,
                    cfg,
                    advection_coordinates=coordinates,
                )
                output[member, lead] = transformer.inverse_transform(field)
                histories = [
                    decomposition_fft(
                        field * taper, bp_filter, compute_stats=True, normalize=True
                    ),
                    histories[0],
                ]
    finally:
        source["gen_calibrated_stoch_field"] = original_generator
    return output, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_repo", type=Path)
    args = parser.parse_args()
    source, provenance = source_engine(args.parameter_repo)
    arrays, cases = {}, []
    for index, fraction in enumerate([0.2, 0.04]):
        shape = (128, 96)
        cfg = config(shape, timestep=7.5 if index else 5.0)
        target = source["StepsParameters"](
            metadata={},
            beta_1=-2.4,
            beta_2=-3.1,
            rain_fraction=fraction,
            mean_db=6,
            stdev_db=3,
        )
        dbr = source["gen_stoch_field"](
            target, shape[1], shape[0], 1, 20, 0, rng=np.random.default_rng(9 + index)
        )
        rain = 10 ** (dbr / 10)
        rain[rain < 1] = 0
        history = np.stack([np.roll(rain, -2, axis=1), np.roll(rain, -1, axis=1), rain])
        velocity = np.zeros((2, *shape))
        velocity[0] = 1
        parameters, _, _, _ = prepare(source, history, velocity, cfg)
        seed = 20261015 + index
        output, calibration = evolve(source, history, velocity, cfg, parameters, seed)
        arrays.update(
            {
                f"history_{index}": history,
                f"velocity_{index}": velocity,
                f"forecast_{index}": output,
            }
        )
        cases.append(
            dict(
                config=cfg,
                seed=seed,
                parameters=parameters.active_parameters(),
                moment_source=parameters.qa["moment_source"],
                calibration=calibration,
            )
        )
    folder = Path(__file__).parent
    np.savez_compressed(folder / "parametric_nowcast_reference.npz", **arrays)
    (folder / "parametric_nowcast_reference.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2, allow_nan=False)
        + "\n"
    )
    print(f"Wrote {len(cases)} source forecast cases")


if __name__ == "__main__":
    main()
