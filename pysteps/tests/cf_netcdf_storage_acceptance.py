"""Reproduce packed CF exports and independent missing-code/value checks."""

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
from pysteps.dataset_metadata import (
    linked,
    normalize_dataset,
    physical_variables,
    unit_spec,
)
from pysteps.tests.cf_netcdf_reference import reference_cases, EPOCH


def _verify(path, expected, encodings):
    """Check raw storage and independently decoded values against expectations."""
    fields = {}
    with (
        xr.open_dataset(path, decode_coords="all") as external,
        netCDF4.Dataset(path) as raw,
    ):
        actual = io.import_netcdf_cf(path)
        for name in physical_variables(expected):
            source = expected[name].values
            decoded = actual[name].values
            step = encodings[name]["scale_factor"]
            offset = encodings[name].get("add_offset", 0.0)
            fill = encodings[name].get("_FillValue", -32768)
            valid = np.isfinite(source)
            missing = ~valid
            epsilon = (
                16
                * np.finfo("float64").eps
                * (
                    max(
                        float(np.max(np.abs(source[valid]))) if valid.any() else 0,
                        abs(offset),
                    )
                    + 1
                )
            )
            maximum_error = (
                float(np.max(np.abs(decoded[valid] - source[valid])))
                if valid.any()
                else 0.0
            )
            assert maximum_error <= step / 2 + epsilon
            for values in (decoded, external[name].values, raw[name][:].filled(np.nan)):
                np.testing.assert_array_equal(np.isnan(values), missing)
                if valid.any():
                    assert (
                        np.max(np.abs(values[valid] - source[valid]))
                        <= step / 2 + epsilon
                    )
            packed = raw[name]
            assert packed.dtype == np.dtype("int16")
            assert packed._FillValue == fill and packed.scale_factor == step
            assert packed.add_offset == offset
            packed.set_auto_maskandscale(False)
            codes = packed[:]
            np.testing.assert_array_equal(codes == fill, missing)
            np.testing.assert_allclose(
                decoded[valid],
                codes[valid].astype(float) * step + offset,
                rtol=0,
                atol=epsilon,
            )
            fields[name] = {
                "scale_factor": step,
                "add_offset": offset,
                "raw_fill_code": int(fill),
                "raw_fill_count": int((codes == fill).sum()),
                "missing_mask_exact_in_three_read_paths": True,
                "maximum_abs_error": maximum_error,
                "checked_tolerance": step / 2 + epsilon,
            }
        for coordinate in expected.coords:
            if coordinate in actual and (
                np.issubdtype(expected[coordinate].dtype, np.datetime64)
                or coordinate in ("x", "y", "ens_number")
            ):
                np.testing.assert_array_equal(expected[coordinate], actual[coordinate])
        field = expected.attrs["precip_var"]
        assert CRS.from_cf(
            expected[linked(expected[field], "grid_mapping")].attrs
        ).equals(CRS.from_cf(actual[linked(actual[field], "grid_mapping")].attrs))
        for auxiliary in (linked(expected[field], "ancillary_variables") or "").split():
            np.testing.assert_array_equal(actual[auxiliary], expected[auxiliary])
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fields": fields,
    }


def external_input(directory):
    """Write an independently packed input and verify its decoded import."""
    source = reference_cases()["instantaneous_rate"].drop_vars("precip_intensity")
    source.attrs["precip_var"] = "external_rate"
    path = directory / "external_packed.nc"
    source.to_netcdf(path)
    with netCDF4.Dataset(path, "a") as raw:
        var = raw.createVariable(
            "external_rate", "i2", ("time", "y", "x"), fill_value=-32768
        )
        var.setncatts(
            dict(
                standard_name="lwe_precipitation_rate",
                units="mm h-1",
                cell_methods="time: point",
                grid_mapping="crs",
                scale_factor=0.1,
                add_offset=2.0,
                missing_value=np.int16(-32767),
                valid_range=np.array([-20, 32766], dtype="int16"),
            )
        )
        var.set_auto_maskandscale(False)
        var[:] = np.broadcast_to(
            np.array([[-20, -19, -5], [-32768, -32767, 0]], dtype="int16"), (3, 2, 3)
        )
    # Both raw missing markers must be removed before scale/offset and validity.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xr.SerializationWarning)
        actual = io.import_netcdf_cf(path)
    expected = np.broadcast_to([[0.0, 0.1, 1.5], [np.nan, np.nan, 2.0]], (3, 2, 3))
    np.testing.assert_allclose(
        actual.external_rate, expected, rtol=0, atol=1e-12, equal_nan=True
    )
    return actual, {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_fill": -32768,
        "raw_missing_value": -32767,
        "scale_factor": 0.1,
        "add_offset": 2.0,
        "independent_handwritten_codes": True,
        "both_raw_missing_markers_masked_before_scaling": True,
    }


def run(output, with_archive=False):
    """Run the acceptance or benchmark workflow and return its report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "independent-inputs"
    inputs.mkdir(exist_ok=True)
    cases = reference_cases()
    cases["multiple_variables"].attrs["precip_var"] = "precip_accum"
    dry = cases["instantaneous_rate"].copy(deep=True)
    dry.precip_intensity.data[:] = 0.0
    cases["all_dry"] = dry
    cases["offset_rate"] = cases["mean_rate"].copy(deep=True)
    cases["offset_dbz"] = cases["reflectivity"].copy(deep=True)
    cases["custom_zero_fill"] = cases["instantaneous_rate"].copy(deep=True)
    positive = cases["instantaneous_rate"].copy(deep=True)
    positive.precip_intensity.data[:] = [[2.0, 3.0, 4.0], [5.0, 6.0, np.nan]]
    cases["custom_positive_fill"] = positive
    bounded = cases["instantaneous_rate"].copy(deep=True)
    bounded.precip_intensity.attrs["valid_range"] = np.array([0.0, 200.0])
    cases["validity_domain"] = bounded
    external, external_record = external_input(inputs)
    cases["repacked_external"] = external
    input_path = inputs / "view_source.nc"
    cases["instantaneous_rate"].to_netcdf(input_path)
    # The view's physical expectation is the original NaN-containing dataset.
    view = io.import_netcdf_pysteps(input_path, fillna=0.0, onerror="raise")
    cases["finite_fill_view"] = view
    report = {
        "scope": "Stage 5 explicit int16 packing, raw missing codes, decoded tolerances and compatibility writes; no forecast-skill claim",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "xarray", "netCDF4", "pyproj")
        },
        "external_input": external_record,
    }
    if with_archive:
        from pysteps.tests.cf_netcdf_stage2_acceptance import archived_example

        cases["archived_radar_forecast"], report["archive"] = archived_example()
    inventory = {}
    for name, dataset in cases.items():
        expected = normalize_dataset(
            cases["instantaneous_rate"] if name == "finite_fill_view" else dataset
        )
        options = {}
        for variable in physical_variables(expected):
            quantity = unit_spec(expected[variable].attrs["units"])[0]
            options[variable] = dict(
                dtype="int16", scale_factor=0.01 if quantity == "depth" else 0.1
            )
            if name in ("offset_rate", "offset_dbz"):
                options[variable]["add_offset"] = 2.0 if name == "offset_rate" else 10.0
        if name == "custom_zero_fill":
            options[expected.attrs["precip_var"]].update(_FillValue=0, add_offset=200.0)
        elif name == "custom_positive_fill":
            options[expected.attrs["precip_var"]]["_FillValue"] = 10
        path = io.export_netcdf_cf(
            dataset, output / (name + ".nc"), encoding=options, overwrite=True
        )
        inventory[name] = _verify(path, expected, options)
    modes = {}
    source = normalize_dataset(reference_cases()["ensemble_rate"])
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
        cartesian_unit="m",
        yorigin="lower",
    )
    for mode in (None, "timestep", "member"):
        name = "legacy_" + (mode or "bulk")
        exporter = io.initialize_forecast_exporter_netcdf(
            str(output),
            name,
            EPOCH,
            7.5,
            3,
            (2, 3),
            metadata,
            n_ens_members=2,
            datatype="int16",
            fill_value=-32768,
            scale_factor=0.1,
            offset=2.0,
            incremental=mode,
            ensemble_ids=[0, 7],
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
        modes[name] = _verify(
            output / (name + ".nc"),
            source,
            {"precip_intensity": {"scale_factor": 0.1, "add_offset": 2.0}},
        )
    report.update(exports=inventory, legacy_modes=modes)
    (output / "storage_acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Verified {len(inventory)} packed bulk exports, {len(modes)} callback modes and one handwritten external packed input"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--with-archive", action="store_true")
    args = parser.parse_args()
    run(args.output, args.with_archive)
