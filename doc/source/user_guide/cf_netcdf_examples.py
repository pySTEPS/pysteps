"""Run physical-quantity and ensemble CF NetCDF round trips.

Usage: python cf_netcdf_examples.py --output /tmp/cf-user-examples
Requires pySTEPS with xarray, pyproj and netCDF4; no radar archive is needed.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS

from pysteps import io


def example_datasets():
    """Construct small physical datasets independently of the file writer."""
    origin = np.datetime64("2023-01-01T00:00:00", "us")
    values = np.broadcast_to([[0.0, 0.02, 1.5], [2.0, 10.0, np.nan]], (3, 2, 3)).copy()
    grid = xr.Dataset(
        coords={
            "x": (
                "x",
                [1600000.0, 1601000.0, 1602000.0],
                {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"},
            ),
            "y": (
                "y",
                [5901000.0, 5900000.0],
                {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"},
            ),
        }
    )
    grid["crs"] = xr.DataArray(0, attrs=CRS.from_epsg(2193).to_cf())
    products = {}
    definitions = [
        (
            "instantaneous_rate",
            "precip_intensity",
            "lwe_precipitation_rate",
            "mm h-1",
            "point",
        ),
        ("mean_rate", "precip_mean_rate", "lwe_precipitation_rate", "mm h-1", "mean"),
        (
            "interval_depth",
            "precip_accum",
            "lwe_thickness_of_precipitation_amount",
            "mm",
            "sum",
        ),
        (
            "cumulative_depth",
            "precip_cumulative",
            "lwe_thickness_of_precipitation_amount",
            "mm",
            "sum",
        ),
        (
            "reflectivity",
            "reflectivity",
            "equivalent_reflectivity_factor",
            "dBZ",
            "point",
        ),
    ]
    for product, name, standard, units, method in definitions:
        ds = grid.copy(deep=True)
        seconds = np.array([450, 900, 1350])
        if product in ("interval_depth", "cumulative_depth"):
            seconds[-1] = 1800
        times = origin + seconds.astype("timedelta64[s]")
        ds = ds.assign_coords(
            time=("time", times, {"standard_name": "time", "axis": "T"})
        )
        data = values.copy()
        if product == "reflectivity":
            data = np.broadcast_to(
                [[-10.0, 0.0, 20.0], [40.0, -5.0, np.nan]], data.shape
            ).copy()
        if method != "point":
            durations = (
                np.array([450, 450, 900])
                if product == "interval_depth"
                else np.full(3, 600)
            )
            if product == "cumulative_depth":
                durations = seconds
                data = np.cumsum(values, axis=0)
                ds = ds.assign_coords(
                    forecast_reference_time=xr.DataArray(
                        origin, attrs={"standard_name": "forecast_reference_time"}
                    )
                )
            starts = times - durations.astype("timedelta64[s]")
            ds["time_bounds"] = (("time", "bounds"), np.column_stack([starts, times]))
            ds.time.attrs["bounds"] = "time_bounds"
        ds[name] = (
            ("time", "y", "x"),
            data,
            {
                "standard_name": standard,
                "units": units,
                "cell_methods": "time: " + method,
                "grid_mapping": "crs",
            },
        )
        ds.attrs["precip_var"] = name
        products[product] = ds
    ensemble = products["instantaneous_rate"].copy(deep=True)
    ensemble["precip_intensity"] = ensemble.precip_intensity.expand_dims(
        ens_number=[0, 7]
    )
    ensemble.ens_number.attrs = {"standard_name": "realization", "units": "1"}
    ensemble = ensemble.assign_coords(
        forecast_reference_time=xr.DataArray(
            origin, attrs={"standard_name": "forecast_reference_time"}
        )
    )
    products["ensemble_rate"] = ensemble
    return products


def main():
    """Parse command-line options and run the documented file checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for product, dataset in example_datasets().items():
        variable = dataset.attrs["precip_var"]
        step = 0.01 if "depth" in product else 0.1
        for packed in (False, True):
            path = args.output / (product + ("_int16" if packed else "_float") + ".nc")
            encoding = (
                {
                    variable: {
                        "dtype": "int16",
                        "scale_factor": step,
                        "add_offset": 0.0,
                        "_FillValue": -32768,
                    }
                }
                if packed
                else None
            )
            io.export_netcdf_cf(dataset, path, encoding=encoding)
            restored = io.import_netcdf_cf(path)
            expected = dataset[variable].values
            actual = restored[variable].values
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=0,
                atol=step / 2 + 1e-12 if packed else 1e-12,
                equal_nan=True,
            )
            print(path)


if __name__ == "__main__":
    main()
