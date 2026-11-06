"""Reproduce production CF file I/O acceptance, with an optional MCH nowcast.

Run ``python -m pysteps.tests.cf_netcdf_stage34_acceptance --output /tmp/cf-io``.
Add ``--with-archive`` for the configured local MCH archive. Generated exports
can also be inspected with an external CF-1.8 checker.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr
from pyproj import CRS

from pysteps import io
from pysteps.dataset_metadata import linked, normalize_dataset, physical_variables
from pysteps.tests.cf_netcdf_reference import reference_cases
from pysteps.tests.cf_netcdf_stage2_acceptance import archived_example, examples


def _compare(expected, actual):
    """Compare physical values, masks, and metadata after a round trip."""
    for name in physical_variables(expected):
        np.testing.assert_allclose(
            expected[name], actual[name], rtol=0, atol=0, equal_nan=True
        )
        np.testing.assert_array_equal(np.isnan(expected[name]), np.isnan(actual[name]))
    for name in expected.coords:
        if name in actual and (
            np.issubdtype(expected[name].dtype, np.datetime64)
            or name in ("x", "y", "ens_number")
        ):
            np.testing.assert_array_equal(expected[name], actual[name])
    variable = expected.attrs["precip_var"]
    mapping = expected[variable].attrs["grid_mapping"]
    assert CRS.from_cf(expected[mapping].attrs).equals(
        CRS.from_cf(actual[linked(actual[variable], "grid_mapping")].attrs)
    )


def run(directory, with_archive=False):
    """Run the acceptance or benchmark workflow and return its report."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    independent = directory / "independent-inputs"
    independent.mkdir(exist_ok=True)
    input_inventory = {}
    # External fixture -> core importer, without calling the production exporter.
    for name, source in reference_cases().items():
        if name == "multiple_variables":
            source.attrs["precip_var"] = "precip_accum"
        path = independent / (name + ".nc")
        source.to_netcdf(path)
        actual = io.import_netcdf_cf(path)
        expected = normalize_dataset(source)
        _compare(expected, actual)
        input_inventory[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "values_masks_coordinates_crs_preserved": True,
        }

    datasets = examples()
    report = {
        "scope": "Stages 3/4 physical CF file importer/exporter and legacy region-write adapters; no new packing-profile or forecast-skill claim",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "xarray", "netCDF4", "pyproj", "dask")
        },
        "independent_inputs": input_inventory,
    }
    if with_archive:
        datasets["archived_radar_forecast"], report["archive"] = archived_example()
    for name, crs, x, y in (
        (
            "utm_60s",
            "EPSG:32760",
            [500000.0, 501000.0, 502000.0],
            [5901000.0, 5900000.0],
        ),
        (
            "australia_albers",
            "EPSG:3577",
            [1000000.0, 1001000.0, 1002000.0],
            [-3000000.0, -3001000.0],
        ),
    ):
        ds = reference_cases()["projected_without_latlon"]
        ds.crs.attrs = CRS(crs).to_cf()
        ds = ds.assign_coords(x=("x", x, ds.x.attrs), y=("y", y, ds.y.attrs))
        datasets[name] = normalize_dataset(ds)
    datasets["singleton"] = normalize_dataset(
        reference_cases()["projected_without_latlon"].isel(time=slice(0, 1))
    )
    inventory = {}
    for name, dataset in datasets.items():
        path = directory / (name + ".nc")
        include_latlon = name in ("utm_60s", "australia_albers")
        io.export_netcdf_cf(
            dataset, path, overwrite=True, include_latlon=include_latlon
        )
        with xr.open_dataset(path, decode_coords="all") as external:
            _compare(dataset, external)
        actual = io.import_netcdf_cf(path)
        _compare(dataset, actual)
        with netCDF4.Dataset(path) as raw:
            for variable in physical_variables(dataset):
                stored = raw[variable]
                np.testing.assert_allclose(
                    stored[:].filled(np.nan),
                    dataset[variable],
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
                assert "scale_factor" not in stored.ncattrs()
            assert raw.pysteps_write_status == "complete"
            assert raw["time"].dtype == np.dtype("float64")
        inventory[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "values_masks_coordinates_crs_preserved": True,
            "independent_xarray_and_raw_netcdf4_reads": True,
            "include_latlon": include_latlon,
        }
    # Identical physical products through the existing callback entry points.
    modes = {}
    for mode in (None, "timestep", "member"):
        name = "legacy_" + (mode or "bulk")
        source = datasets["ensemble_rate"]
        values = source.precip_intensity.values
        x, y = source.x.values, source.y.values
        dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
        metadata = dict(
            projection=source.attrs["projection"],
            x1=x.min() - dx / 2,
            x2=x.max() + dx / 2,
            y1=y.min() - dy / 2,
            y2=y.max() + dy / 2,
            unit="mm/h",
            yorigin="lower",
            cartesian_unit="m",
        )
        origin = source.time.values[0] - np.timedelta64(450, "s")
        exporter = io.initialize_forecast_exporter_netcdf(
            str(directory),
            name,
            origin,
            7.5,
            3,
            (2, 3),
            metadata,
            n_ens_members=2,
            incremental=mode,
            ensemble_ids=[0, 7],
            datatype="float64",
            overwrite=True,
        )
        if mode is None:
            io.export_forecast_dataset(values, exporter)
        elif mode == "timestep":
            for index in range(3):
                io.export_forecast_dataset(values[:, index], exporter)
        else:
            for index in range(2):
                io.export_forecast_dataset(values[index], exporter)
        io.close_forecast_files(exporter)
        path = directory / (name + ".nc")
        actual = io.import_netcdf_cf(path)
        _compare(source, actual)
        modes[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "values_masks_coordinates_crs_preserved": True,
        }
    report.update(exports=inventory, legacy_modes=modes)
    (directory / "stage34_examples.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--with-archive", action="store_true")
    args = parser.parse_args()
    report = run(args.output, args.with_archive)
    print(
        f"Validated {len(report['independent_inputs'])} independent inputs, {len(report['exports'])} bulk exports and {len(report['legacy_modes'])} callback modes"
    )
