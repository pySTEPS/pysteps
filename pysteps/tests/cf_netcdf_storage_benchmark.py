"""Repeatable warm-cache CF storage measurements, without CI timing limits."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time

import netCDF4
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from pysteps import io
from pysteps.dataset_metadata import normalize_dataset
from pysteps.tests.cf_netcdf_reference import EPOCH, reference_cases


def synthetic(shape=(4, 8, 384, 512), seed=5128):
    members, times, height, width = shape
    rng = np.random.default_rng(seed)
    terrain = gaussian_filter(rng.standard_normal((height, width)), 8.0, mode="wrap")
    terrain /= terrain.std()
    values = np.empty(shape, dtype="float32")
    for member in range(members):
        for step in range(times):
            shifted = np.roll(
                terrain, (2 * step + member, 3 * member - 4 * step), axis=(0, 1)
            )
            perturbation = 0.08 * gaussian_filter(
                rng.standard_normal((height, width)), 1.0
            )
            rate = np.expm1(np.clip(2.0 * (shifted + perturbation) - 0.25, 0.0, 5.5))
            rate *= np.exp(0.08 * rng.standard_normal((height, width)))
            missing = rng.random((height, width)) < 0.003
            missing[: max(1, height // 64), :] = True
            rate[missing] = np.nan
            values[member, step] = rate
    template = reference_cases()["projected_without_latlon"]
    dataset = xr.Dataset(
        {
            "precip_intensity": (
                ("ens_number", "time", "y", "x"),
                values,
                {
                    "standard_name": "lwe_precipitation_rate",
                    "long_name": "synthetic instantaneous rain rate",
                    "units": "mm/h",
                    "cell_methods": "time: point",
                    "grid_mapping": "crs",
                },
            )
        },
        coords={
            "x": ("x", 1600000.0 + np.arange(width) * 1000, template.x.attrs),
            "y": ("y", 5900000.0 - np.arange(height) * 1000, template.y.attrs),
            "time": (
                "time",
                EPOCH + (np.arange(times) + 1) * np.timedelta64(450, "s"),
                {"standard_name": "time"},
            ),
            "ens_number": (
                "ens_number",
                np.arange(members) * 7,
                {"standard_name": "realization", "units": "1"},
            ),
            "crs": template.crs,
        },
        attrs={
            "precip_var": "precip_intensity",
            "title": "Synthetic rainfall storage benchmark",
            "history": f"Seed {seed}; spatially correlated advected rainfall; not a forecast-skill experiment",
        },
    )
    return normalize_dataset(dataset)


def profiles(dataset):
    variable = dataset.attrs["precip_var"]
    field = dataset[variable]

    def chunks(tile=256, time_count=1, member_count=1):
        return tuple(
            min(
                field.sizes[dim],
                (
                    tile
                    if dim in ("x", "y")
                    else time_count if dim == "time" else member_count
                ),
            )
            for dim in field.dims
        )

    cases = []
    for dtype, prefix in (("float32", "f32"), ("int16", "i16")):
        for level in (0, 1, 4, 6, 9):
            options = dict(
                dtype=dtype,
                zlib=level > 0,
                complevel=level,
                shuffle=True,
                chunksizes=chunks(),
            )
            if dtype == "int16":
                options.update(scale_factor=0.1, add_offset=0.0, _FillValue=-32768)
            cases.append((f"{prefix}_zlib{level}_tile256", options))
    for name, tile, nt, nm, shuffle in (
        ("i16_zlib4_tile128", 128, 1, 1, True),
        ("i16_zlib4_frame", max(field.sizes["y"], field.sizes["x"]), 1, 1, True),
        ("i16_zlib4_time4", 256, 4, 1, True),
        ("i16_zlib4_members4", 256, 1, 4, True),
        ("i16_zlib4_no_shuffle", 256, 1, 1, False),
    ):
        if "members4" in name and "ens_number" not in field.dims:
            continue
        cases.append(
            (
                name,
                dict(
                    dtype="int16",
                    scale_factor=0.1,
                    add_offset=0.0,
                    _FillValue=-32768,
                    zlib=True,
                    complevel=4,
                    shuffle=shuffle,
                    chunksizes=chunks(tile, nt, nm),
                ),
            )
        )
    return cases


def _read(path, variable, pattern):
    with netCDF4.Dataset(path) as source:
        field = source[variable]
        dims = field.dimensions
        if pattern == "full":
            index = tuple(slice(None) for _ in dims)
        elif pattern == "map":
            index = tuple(slice(None) if dim in ("y", "x") else 0 for dim in dims)
        else:
            index = tuple(
                (
                    slice(None)
                    if dim == "time"
                    else field.shape[n] // 2 if dim in ("y", "x") else 0
                )
                for n, dim in enumerate(dims)
            )
        return np.asarray(field[index].filled(np.nan), dtype="float32")


def _summary(samples):
    return {
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "samples": samples,
    }


def measure(dataset, directory, repeats=3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    variable = dataset.attrs["precip_var"]
    values = np.asarray(dataset[variable], dtype="float32")
    dataset = dataset.copy(deep=False)
    dataset[variable] = (dataset[variable].dims, values, dataset[variable].attrs)
    geographic = [
        name
        for name in dataset.coords
        if dataset[name].attrs.get("standard_name") in ("latitude", "longitude")
    ]
    dataset = dataset.drop_vars(geographic)
    result = {
        "variable": variable,
        "shape": list(values.shape),
        "dtype": "float32",
        "input_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "input_bytes": values.nbytes,
        "missing_pixels": int(np.isnan(values).sum()),
        "observed_dry_pixels": int((values == 0).sum()),
        "positive_pixels": int((values > 0).sum()),
        "positive_traces_below_half_step": int(((values > 0) & (values < 0.05)).sum()),
        "maximum_mmph": float(np.nanmax(values)),
        "profiles": {},
    }
    for name, encoding in profiles(dataset):
        path = directory / (name + ".nc")
        timings = {key: [] for key in ("write", "full", "map", "point_series")}
        for _ in range(repeats):
            start = time.perf_counter()
            io.export_netcdf_cf(
                dataset, path, encoding={variable: encoding}, overwrite=True
            )
            timings["write"].append(time.perf_counter() - start)
            for pattern in ("full", "map", "point_series"):
                start = time.perf_counter()
                actual = _read(path, variable, pattern)
                timings[pattern].append(time.perf_counter() - start)
        actual = _read(path, variable, "full")
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(values))
        tolerance = (
            0.05 + 4 * np.finfo("float32").eps * (float(np.nanmax(values)) + 1)
            if encoding["dtype"] == "int16"
            else 0
        )
        tolerance = float(tolerance)
        maximum_error = float(
            np.nanmax(np.abs(actual.astype(float) - values.astype(float)))
        )
        assert maximum_error <= tolerance, (name, maximum_error, tolerance)
        with netCDF4.Dataset(path) as raw:
            field = raw[variable]
            filters, chunks = field.filters(), field.chunking()
            field.set_auto_maskandscale(False)
            if encoding["dtype"] == "int16":
                codes = field[:]
                np.testing.assert_array_equal(codes == -32768, np.isnan(values))
                assert field.scale_factor == 0.1 and field._FillValue == -32768
        result["profiles"][name] = {
            "encoding": {
                k: list(v) if isinstance(v, tuple) else v for k, v in encoding.items()
            },
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "maximum_abs_error_mmph": maximum_error,
            "checked_tolerance_mmph": tolerance,
            "missing_mask_exact": True,
            "filters": filters,
            "chunksizes": chunks,
            "seconds": {key: _summary(samples) for key, samples in timings.items()},
        }
        print(
            f"{directory.name}/{name}: {path.stat().st_size} bytes; write {statistics.median(timings['write']):.3f}s; map {statistics.median(timings['map']):.3f}s",
            flush=True,
        )
    baseline = result["profiles"]["f32_zlib4_tile256"]
    for record in result["profiles"].values():
        record["size_relative_to_float32_default"] = record["bytes"] / baseline["bytes"]
    return result


def run(output, shape=(4, 8, 384, 512), repeats=3, with_archive=False):
    if repeats < 1 or len(shape) != 4 or min(shape) < 1:
        raise ValueError(
            "Positive repeats and four positive shape entries are required"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Warm operating-system cache storage/backend measurements on identical physical inputs; no cold-cache, fsync throughput, forecast-skill or cross-machine timing claim",
        "repeats": repeats,
        "source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parents[2] / name).read_bytes()
            ).hexdigest()
            for name in (
                "pysteps/io/netcdf_cf.py",
                "pysteps/tests/cf_netcdf_storage_benchmark.py",
            )
        },
        "seed": 5128,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "xarray", "netCDF4", "pyproj")
        },
        "netcdf_library": netCDF4.getlibversion(),
        "hdf5_library": netCDF4.__hdf5libversion__,
        "read_method": "netCDF4 auto mask/scale into float32; reopen per read, materialize all requested data",
        "scenarios": {},
    }
    report["scenarios"]["synthetic_ensemble"] = measure(
        synthetic(shape), output / "synthetic_ensemble", repeats
    )
    if with_archive:
        from pysteps.tests.cf_netcdf_stage2_acceptance import archived_example

        dataset, provenance = archived_example()
        report["archive_provenance"] = provenance
        report["scenarios"]["archived_radar_forecast"] = measure(
            dataset, output / "archived_radar_forecast", repeats
        )
    (output / "storage_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--shape",
        nargs=4,
        type=int,
        default=(4, 8, 384, 512),
        metavar=("MEMBERS", "TIMES", "Y", "X"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--with-archive", action="store_true")
    args = parser.parse_args()
    run(args.output, tuple(args.shape), args.repeats, args.with_archive)
