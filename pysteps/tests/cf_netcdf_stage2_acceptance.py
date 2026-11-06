"""Reproduce stage 2 shared-metadata examples without the future file APIs.

Run ``python -m pysteps.tests.cf_netcdf_stage2_acceptance --output /tmp/cf-stage2``.
Add ``--with-archive`` to exercise the configured local MCH sample archive.
Files are serialized by xarray directly, then checked by an independent CF
checker using the command recorded in the stage 2 documentation.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from pysteps.dataset_metadata import normalize_dataset, physical_variables
from pysteps.tests.cf_netcdf_reference import reference_cases
from pysteps.utils.conversion import to_raindepth, to_rainrate
from pysteps.utils.dimension import aggregate_fields_time
from pysteps.xarray_helpers import convert_output_to_xarray_dataset


def examples():
    """Build representative projected physical datasets for acceptance checks."""
    normalized = {
        name: normalize_dataset(
            dataset, "precip_accum" if name == "multiple_variables" else None
        )
        for name, dataset in reference_cases().items()
    }
    depth = normalized["interval_depth"].copy(deep=True)
    depth.precip_accum.attrs.update(threshold=0.1, zerovalue=0.0)
    normalized.update(
        converted_mean=to_rainrate(depth),
        integrated_point=to_raindepth(
            normalized["instantaneous_rate"],
            integration="constant",
            duration_seconds=600,
        ),
        aggregated_depth=aggregate_fields_time(normalized["interval_depth"], 15),
        forecast_mean_sparse=convert_output_to_xarray_dataset(
            normalized["mean_rate"], [1, 2, 4], np.ones((3, 2, 3))
        ),
        dbz_to_rate=to_rainrate(normalized["reflectivity"]),
    )
    midpoint = reference_cases()["mean_rate"]
    midpoint = midpoint.assign_coords(
        time=(
            "time",
            midpoint.time.values - np.timedelta64(300, "s"),
            midpoint.time.attrs,
        )
    )
    normalized["midpoint_mean"] = normalize_dataset(midpoint)
    missing = normalized["interval_depth"].copy(deep=True)
    missing.precip_accum.data[0, 0, 1] = np.nan
    normalized["partial_aggregation"] = aggregate_fields_time(
        missing, 15, ignore_nan=True
    )
    return normalized


def archived_example():
    """Load the archived rainfall example used for optional acceptance checks."""
    from pysteps import io, motion, rcparams
    from pysteps.nowcasts.extrapolation import forecast
    from pysteps.tests.helpers import _reference_dates, get_precipitation_fields
    from pysteps.utils.dimension import aggregate_fields_space
    from pysteps.utils.transformation import dB_transform

    source = get_precipitation_fields(num_prev_files=2, return_raw=True, source="mch")
    physical = aggregate_fields_space(to_rainrate(source), 2000)
    modeled = dB_transform(physical, threshold=0.1, zerovalue=-15.0)
    predicted = forecast(motion.get_method("LK")(modeled), [1, 2, 4])
    output = to_rainrate(predicted)
    variable = output.attrs["precip_var"]
    original = output[variable].values.copy()
    normalized = normalize_dataset(output)
    np.testing.assert_allclose(original, normalized[variable], equal_nan=True)
    assert np.array_equal(np.isnan(original), np.isnan(normalized[variable]))
    normalized.attrs.update(
        title="Stage 2 archived-radar metadata integration",
        history="Three archived MCH fields; rate at 2 km; explicit dBR threshold 0.1 mm/h and floor -15; LK motion; extrapolate leads 1,2,4; return to physical rate; normalize metadata",
    )
    config = rcparams["data_sources"]["mch"]
    filenames, _ = io.archive.find_by_date(
        _reference_dates["mch"],
        config["root_path"],
        config["path_fmt"],
        config["fn_pattern"],
        config["fn_ext"],
        config["timestep"],
        num_prev_files=2,
    )
    record = {
        "source": "Configured local MCH sample archive",
        "input_files": [
            {
                "name": Path(name).name,
                "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest(),
            }
            for name in filenames
        ],
        "input_times": [str(t) for t in physical.time.values],
        "output_times": [str(t) for t in normalized.time.values],
        "shape": list(normalized[variable].shape),
        "missing_pixels": int(np.isnan(original).sum()),
        "mask_preserved_by_normalization": True,
        "numerical_values_preserved_by_normalization": True,
        "lead_steps": [1, 2, 4],
        "rate_threshold_mmph": 0.1,
        "modelling_floor_dbr": -15.0,
    }
    return normalized, record


def run(directory, with_archive=False):
    """Run the acceptance or benchmark workflow and return its report."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    datasets = examples()
    record = {
        "scope": "Stage 2 shared metadata and helper integration; no new file importer/exporter, packing implementation or forecast-skill claim",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "xarray", "netCDF4", "pyproj")
        },
    }
    if with_archive:
        datasets["archived_radar_forecast"], record["archive"] = archived_example()
    inventory = {}
    for name, dataset in datasets.items():
        path = directory / (name + ".nc")
        dataset.to_netcdf(path)
        with xr.open_dataset(path, decode_coords="all") as reopened:
            decoded = normalize_dataset(reopened, dataset.attrs["precip_var"])
            for variable in physical_variables(dataset):
                np.testing.assert_allclose(
                    dataset[variable], decoded[variable], rtol=0, atol=0, equal_nan=True
                )
                np.testing.assert_array_equal(
                    np.isnan(dataset[variable]), np.isnan(decoded[variable])
                )
            for coordinate in (
                "x",
                "y",
                "time",
                "ens_number",
                "forecast_reference_time",
            ):
                if coordinate in dataset:
                    np.testing.assert_array_equal(
                        dataset[coordinate], decoded[coordinate]
                    )
        with netCDF4.Dataset(path) as raw:
            for variable in raw.variables.values():
                if getattr(variable, "standard_name", None) in (
                    "time",
                    "forecast_reference_time",
                ):
                    assert variable.dtype == np.dtype("float64")
                if getattr(variable, "standard_name", None) == "realization":
                    assert variable.dtype == np.dtype("int32")
        inventory[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "decoded_values_masks_coordinates_preserved": True,
            "classic_numeric_time_member_types": True,
        }
    record["inventory"] = inventory
    (directory / "stage2_examples.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--with-archive", action="store_true")
    args = parser.parse_args()
    report = run(args.output, args.with_archive)
    print(f"Recorded {len(report['inventory'])} stage 2 examples in {args.output}")
