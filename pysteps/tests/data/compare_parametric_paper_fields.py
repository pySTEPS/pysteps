"""Compare source and ported estimators on saved paper rainfall fields.

Requires the source project's environment, including xarray, and the saved
figs/individual_field_fits/*.npz inputs. Does not connect to a database.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.special import log_ndtr

from pysteps.utils.spectral import fit_power_law_spectrum, rapsd_logarithmic
from pysteps.utils.statistics import compute_field_statistics
from pysteps.utils.transformation import dB_transform
from generate_parametric_reference import _load_functions


def main():
    import xarray as xr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_repo", type=Path)
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    namespace = dict(
        np=np,
        xr=xr,
        curve_fit=curve_fit,
        minimize_scalar=minimize_scalar,
        log_ndtr=log_ndtr,
    )
    source = args.parameter_repo / "src/pysteps_param/rain_stats_compute.py"
    source_hash = _load_functions(
        source,
        {"power_spectrum_1D", "compute_field_stats", "fit_lower_truncated_normal"},
        namespace,
    )
    report = dict(
        source_sha256=source_hash,
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.parameter_repo, text=True
        ).strip(),
        threshold_mm_per_hour=1.0,
        lower_truncation_dbr=1.25,
        scale_break_km=15.0,
        pixel_size_km=1.0,
        cases=[],
    )
    paths = sorted((args.paper_repo / "figs/individual_field_fits").glob("*.npz"))
    if not paths:
        raise ValueError("No saved paper fields found")
    for path in paths:
        with np.load(path) as saved:
            rain = saved["rain"]
        transformed, metadata = dB_transform(
            rain.astype(float), threshold=1.0, zerovalue=-0.1
        )
        source_spectrum, source_model = namespace["power_spectrum_1D"](transformed, 15)
        source_statistics = namespace["compute_field_stats"](
            transformed, metadata, fit_truncated=True, lower_truncation=1.25
        )
        frequency, power = rapsd_logarithmic(transformed)
        model = fit_power_law_spectrum(frequency, power, 15)
        statistics = compute_field_statistics(transformed, metadata["threshold"], 1.25)
        np.testing.assert_allclose(
            10 * np.log10(frequency), source_spectrum.psd_bins.values, atol=1e-12
        )
        np.testing.assert_allclose(
            10 * np.log10(power), source_spectrum.psd.values, atol=1e-10
        )
        differences = {
            name: abs(model[name] - source_model[name])
            for name in ("beta_1", "beta_2", "c1", "c2")
        }
        assert max(differences.values()) < 3e-5
        for name in ("ordinary_mean", "ordinary_stdev", "latent_mean", "latent_stdev"):
            np.testing.assert_allclose(
                statistics[name], source_statistics[name], rtol=1e-12
            )
        np.testing.assert_allclose(
            statistics["rain_fraction"],
            source_statistics["nonzero_fraction"],
            rtol=1e-12,
        )
        report["cases"].append(
            dict(
                name=path.stem,
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                shape=list(rain.shape),
                maximum_spectrum_error_db=float(
                    np.max(abs(10 * np.log10(power) - source_spectrum.psd.values))
                ),
                spectral_parameter_absolute_differences=differences,
                rain_fraction=statistics["rain_fraction"],
                marginal_fit_status=statistics["nonzero_fit"]["status"],
                spectrum_fit_status=model["status"],
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Compared {len(paths)} saved fields; report: {args.output}")


if __name__ == "__main__":
    main()
