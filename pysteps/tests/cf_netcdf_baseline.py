"""Reproduce pre-implementation CF I/O gaps; this is a diagnostic, not a test.

Run with ``python -m pysteps.tests.cf_netcdf_baseline --output /tmp/baseline.json``.
Observations are deliberately separate from the reference fixtures and do not
make existing incorrect behaviour a regression requirement.
"""

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import warnings

import numpy as np
import xarray as xr

from pysteps import xarray_helpers
from pysteps.io import exporters, nowcast_importers
from pysteps.tests.cf_netcdf_reference import reference_cases
from pysteps.utils import conversion, dimension


def collect_baseline():
    """Collect baseline observations of physical metadata and storage behavior."""
    observations = {}

    def observe(name, intended, operation):
        """Record a baseline result or exception alongside its intended behavior."""
        try:
            actual = operation()
            if isinstance(actual, xr.Dataset):
                actual = {"variables": list(actual.variables)}
        except Exception as error:
            actual = {"exception": type(error).__name__, "message": str(error)}
        observations[name] = {"intended": intended, "observed": actual}

    def simple(unit, values, **attributes):
        """Build a minimal legacy field for baseline observations."""
        name = {
            "mm/h": "precip_intensity",
            "mm": "precip_accum",
            "dBZ": "reflectivity",
        }[unit]
        return xr.Dataset(
            {
                name: (
                    ("time", "y", "x"),
                    np.asarray(values, dtype=float).reshape(-1, 1, 1),
                    {
                        "units": unit,
                        "threshold": 0.1,
                        "zerovalue": 0.0,
                        "accutime": 5.0,
                        **attributes,
                    },
                )
            },
            coords={
                "time": np.array(
                    ["2023-01-01T00:05", "2023-01-01T00:10"], dtype="datetime64[s]"
                )[: len(values)]
            },
            attrs={"precip_var": name},
        )

    def aggregation():
        """Measure the depth, duration, and rate produced by temporal aggregation."""
        ds = dimension.aggregate_fields_time(simple("mm", [1, 1]), 10)
        return {
            "depth_mm": float(ds.precip_accum[0, 0, 0]),
            "duration_minutes": ds.precip_accum.attrs["accutime"],
            "converted_rate_mmph": float(
                conversion.to_rainrate(ds).precip_intensity[0, 0, 0]
            ),
        }

    observe(
        "aggregation_duration",
        "2 mm over ten minutes is a mean rate of 12 mm/h",
        aggregation,
    )
    for marker in (False, True):

        def dbz(marker=marker):
            """
            Record reflectivity conversion with or without a statistical transform
            marker.
            """
            ds = simple("dBZ", [20], **({"transform": "dB"} if marker else {}))
            return float(conversion.to_rainrate(ds).precip_intensity[0, 0, 0])

        observe(
            f"dbz_transform_{marker}",
            "20 dBZ with a=200,b=1.6 gives 0.6484197773 mm/h",
            dbz,
        )
    observe(
        "transform_none",
        "None means no optional transformation",
        lambda: conversion.to_rainrate(simple("mm/h", [1], transform=None)),
    )

    def alias():
        """Record the handling of an alternative physical variable name."""
        ds = simple("mm/h", [1])
        ds.precip_intensity.attrs["units"] = "mm h-1"
        conversion.to_rainrate(ds)
        return "converted"

    observe("unit_alias", "mm h-1 is accepted and normalized to mm/h", alias)
    cases = reference_cases()

    def bounded_depth():
        """Exercise conversion with explicit depth interval bounds."""
        ds = cases["interval_depth"].copy(deep=True)
        ds.attrs["precip_var"] = "precip_accum"
        ds.precip_accum.attrs.update(threshold=0.1, zerovalue=0.0)
        conversion.to_rainrate(ds)
        return "converted using bounds"

    observe("bounds_without_accutime", "derive duration from CF bounds", bounded_depth)

    def forecast_bounds():
        """Record forecast interval bounds for the baseline fixture."""
        ds = cases["interval_depth"].copy(deep=True)
        ds.attrs["precip_var"] = "precip_accum"
        ds.time.attrs["stepsize"] = (
            450  # isolate bounds handling from cadence inference
        )
        out = xarray_helpers.convert_output_to_xarray_dataset(ds, 2, np.ones((2, 2, 3)))
        return {
            "bounds_link": out.time.attrs.get("bounds"),
            "bounds_present": "time_bnds" in out,
        }

    observe(
        "forecast_bounds",
        "construct forecast bounds and leave no dangling references",
        forecast_bounds,
    )
    metadata = dict(
        projection="EPSG:2193",
        x1=0.0,
        x2=3000.0,
        y1=0.0,
        y2=2000.0,
        xpixelsize=1000.0,
        ypixelsize=1000.0,
        yorigin="upper",
        unit="mm/h",
        cartesian_unit="m",
        threshold=0.1,
        zerovalue=0.0,
        institution="synthetic",
    )
    observe(
        "epsg_crs",
        "accept a supported EPSG CRS",
        lambda: xarray_helpers.convert_input_to_xarray_dataset(
            np.ones((2, 3)), None, metadata
        ),
    )
    stere = (
        "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +R=6371000 +x_0=0 +y_0=0 +units=m"
    )
    with tempfile.TemporaryDirectory(prefix="pysteps-cf-baseline-") as directory:
        directory = Path(directory)

        def serialize_none():
            """Record serialization behavior for metadata containing None."""
            ds = xarray_helpers.convert_input_to_xarray_dataset(
                np.ones((2, 3)),
                None,
                {**metadata, "projection": stere, "transform": None},
            )
            ds.to_netcdf(directory / "none.nc")
            return "written"

        observe(
            "serialize_none",
            "absent optional transformations are serializable",
            serialize_none,
        )

        def packed_missing():
            """Inspect missing values after packing and independent decoding."""
            exporter = exporters.initialize_forecast_exporter_netcdf(
                str(directory),
                "packed",
                datetime(2023, 1, 1),
                5,
                1,
                (2, 3),
                {**metadata, "projection": stere},
                datatype=np.int16,
                scale_factor=0.1,
                offset=0.0,
                fill_value=-32768,
            )
            data = np.ones((1, 2, 3))
            data[0, 0, 0] = np.nan
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    exporters.export_forecast_dataset(data, exporter)
            finally:
                exporters.close_forecast_files(exporter)
            with xr.open_dataset(directory / "packed.nc") as ds:
                return {
                    "missing_preserved": bool(np.isnan(ds.precip_intensity[0, 0, 0])),
                    "decoded_value": (
                        float(ds.precip_intensity[0, 0, 0])
                        if np.isfinite(ds.precip_intensity[0, 0, 0])
                        else None
                    ),
                }

        observe(
            "packed_missing", "missing input remains missing, not dry", packed_missing
        )
        cases["instantaneous_rate"].to_netcdf(directory / "input.nc")

        def importer_options():
            """Record importer handling of dtype and missing-value options."""
            with nowcast_importers.import_netcdf_pysteps(
                directory / "input.nc", dtype="single", fillna=-999
            ) as ds:
                return {
                    "dtype": str(ds.precip_intensity.dtype),
                    "nan_still_present": bool(np.isnan(ds.precip_intensity).any()),
                }

        observe(
            "importer_options",
            "documented dtype/fillna options have an effect",
            importer_options,
        )
    root = Path(__file__).resolve().parents[2]
    return {
        "date": "2026-11-05",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "xarray", "netCDF4", "pyproj", "pytest")
        },
        "observations": observations,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(collect_baseline(), indent=2, allow_nan=False) + "\n"
    )
    print(f"Baseline recorded in {args.output}")
