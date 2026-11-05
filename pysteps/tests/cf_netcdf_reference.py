"""Small, independent stage-1 CF datasets; no production I/O implementation.

Generate reference NetCDF inputs with::

    python -m pysteps.tests.cf_netcdf_reference --output /tmp/cf-reference

The source arrays and metadata are the fixtures. Generated binaries are not
committed. This module deliberately does not call pySTEPS conversion or I/O.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

EPOCH = np.datetime64("2023-01-01T00:00:00", "us")
TIME_ENCODING = {
    "units": "seconds since 2023-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "float64",
    "_FillValue": None,
}
CASE_IDS = (
    "instantaneous_rate",
    "mean_rate",
    "interval_depth",
    "cumulative_depth",
    "reflectivity",
    "ensemble_rate",
    "multiple_variables",
    "unit_alias_rate",
    "all_missing",
    "polar_sphere",
    "albers",
    "projected_without_latlon",
)


def _times(seconds):
    return EPOCH + np.asarray(seconds, dtype="timedelta64[s]")


def _grid(crs_input="EPSG:2193", descending=True):
    crs = CRS.from_user_input(crs_input)
    x = 1600000.0 + np.arange(3) * 1000.0
    y = 5900000.0 + np.arange(2) * 1000.0
    if crs_input == "EPSG:3812":
        x = 300000.0 + np.arange(3) * 1000.0
        y = 150000.0 + np.arange(2) * 1000.0
    if descending:
        y = y[::-1]
    xx, yy = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True).transform(
        xx, yy
    )
    ds = xr.Dataset(
        coords={
            "x": (
                "x",
                x,
                {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"},
            ),
            "y": (
                "y",
                y,
                {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"},
            ),
            "lat": (
                ("y", "x"),
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                ("y", "x"),
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
        attrs={
            "Conventions": "CF-1.8",
            "standard_name_vocabulary": "CF Standard Name Table v95",
            "title": "pySTEPS CF I/O stage 1 synthetic reference",
            "institution": "pySTEPS development reference fixtures",
            "source": "Hand-specified synthetic fields, not observations or forecasts",
            "history": "2026-11-05: defined independently of pySTEPS I/O",
            "references": "https://github.com/pySTEPS/pysteps/issues/5",
            "comment": "Reference fixture; does not demonstrate production exporter compliance",
        },
    )
    ds["crs"] = xr.DataArray(
        np.int32(0), attrs={"long_name": "coordinate reference system", **crs.to_cf()}
    )
    ds = ds.set_coords("crs")
    ds.crs.encoding["_FillValue"] = None
    for name in ("x", "y", "lat", "lon"):
        ds[name].encoding.update(dtype="float64", _FillValue=None)
    return ds


def _time_axis(ds, seconds, durations=None, name="time"):
    ds = ds.assign_coords(
        {name: (name, _times(seconds), {"standard_name": "time", "axis": "T"})}
    )
    ds[name].encoding = TIME_ENCODING.copy()
    if durations is not None:
        ends = np.asarray(seconds)
        starts = ends - np.asarray(durations)
        bounds = name + "_bnds"
        ds[bounds] = ((name, "nv"), _times(np.column_stack([starts, ends])))
        ds[bounds].encoding = TIME_ENCODING.copy()
        ds = ds.set_coords(bounds)
        ds[name].attrs["bounds"] = bounds
    return ds


def _field(ds, name, values, units, standard_name, method="point", time="time"):
    ds[name] = (
        (time, "y", "x"),
        np.asarray(values, dtype="float64"),
        {
            "units": units,
            "standard_name": standard_name,
            "long_name": name.replace("_", " "),
            "cell_methods": f"{time}: {method}",
            "grid_mapping": "crs",
        },
    )
    ds[name].encoding.update(dtype="float64", _FillValue=np.nan)
    return ds


def reference_cases():
    """Return editable reference datasets keyed by the documented matrix IDs."""
    cases = {}
    rain = np.array([[0.0, 0.04, 0.14], [1.26, 100.0, np.nan]])
    values = np.broadcast_to(rain, (3, 2, 3)).copy()
    rate_name = "lwe_precipitation_rate"
    depth_name = "lwe_thickness_of_precipitation_amount"
    regular = [450, 900, 1350]
    point = _time_axis(_grid(), regular)
    cases["instantaneous_rate"] = _field(
        point, "precip_intensity", values, "mm h-1", rate_name
    )
    # Cadence is 450 seconds; support is 600 seconds. It must not be inferred.
    mean = _time_axis(_grid(), regular, [600, 600, 600])
    cases["mean_rate"] = _field(
        mean, "precip_mean_rate", values, "mm h-1", rate_name, "mean"
    )
    durations = np.array([450, 450, 900])
    depth = _time_axis(_grid(), [450, 900, 1800], durations)
    depths = np.broadcast_to(durations[:, None, None] / 300.0, (3, 2, 3)).copy()
    depths[:, 0, 0] = 0.0
    depths[:, 1, 2] = np.nan
    cases["interval_depth"] = _field(
        depth, "precip_accum", depths, "mm", depth_name, "sum"
    )
    cumulative = _time_axis(_grid(), [450, 900, 1800], [450, 900, 1800])
    cumulative["forecast_reference_time"] = xr.DataArray(
        EPOCH, attrs={"standard_name": "forecast_reference_time"}
    )
    cumulative.forecast_reference_time.encoding = TIME_ENCODING.copy()
    cases["cumulative_depth"] = _field(
        cumulative,
        "precip_cumulative",
        np.cumsum(depths, axis=0),
        "mm",
        depth_name,
        "sum",
    )
    reflectivity = _time_axis(_grid(), regular)
    dbz = np.broadcast_to([[-10.0, 0.0, 20.0], [40.0, -32.0, np.nan]], (3, 2, 3))
    reflectivity = _field(
        reflectivity, "reflectivity", dbz, "dBZ", "equivalent_reflectivity_factor"
    )
    status = np.broadcast_to([[1, 1, 1], [1, 0, np.nan]], (3, 2, 3)).astype("float32")
    reflectivity["detection_status"] = (
        ("time", "y", "x"),
        status,
        {
            "long_name": "reflectivity detection status",
            "units": "1",
            "flag_values": np.array([0, 1], dtype="int8"),
            "flag_meanings": "not_detected detected",
            "grid_mapping": "crs",
        },
    )
    reflectivity.detection_status.encoding.update(dtype="int8", _FillValue=np.int8(9))
    reflectivity.reflectivity.attrs["ancillary_variables"] = "detection_status"
    cases["reflectivity"] = reflectivity
    ensemble = _time_axis(_grid("EPSG:3812", descending=False), regular)
    ensemble = _field(ensemble, "precip_intensity", values, "mm h-1", rate_name)
    ensemble = ensemble.expand_dims(ens_number=[0, 7])
    # Only the precipitation has a realization dimension; spatial coordinates and
    # the CRS must not acquire an ensemble dimension from expand_dims.
    ensemble = _time_axis(_grid("EPSG:3812", descending=False), regular).assign(
        precip_intensity=ensemble.precip_intensity
    )
    ensemble.ens_number.attrs = {"standard_name": "realization", "units": "1"}
    ensemble.ens_number.encoding.update(dtype="int32", _FillValue=None)
    cases["ensemble_rate"] = ensemble
    multi = cases["instantaneous_rate"].copy(deep=True)
    multi = _time_axis(multi, [900, 1800], [900, 900], name="time_depth")
    multi = _field(
        multi,
        "precip_accum",
        np.full((2, 2, 3), 3.0),
        "mm",
        depth_name,
        "sum",
        time="time_depth",
    )
    cases["multiple_variables"] = multi
    alias = cases["instantaneous_rate"].copy(deep=True)
    alias.precip_intensity.data /= 3600000.0
    alias.precip_intensity.attrs["units"] = "m s-1"
    cases["unit_alias_rate"] = alias
    missing = cases["instantaneous_rate"].copy(deep=True)
    missing.precip_intensity.data[:] = np.nan
    cases["all_missing"] = missing
    for case_id, projection in [
        (
            "polar_sphere",
            "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +R=6371000 +units=m",
        ),
        (
            "albers",
            "+proj=aea +lat_0=-37.852 +lon_0=144.752 +lat_1=-18 +lat_2=-36 +ellps=GRS80 +units=m",
        ),
    ]:
        cases[case_id] = _field(
            _time_axis(_grid(projection), regular),
            "precip_intensity",
            values,
            "mm h-1",
            rate_name,
        )
    # The reference explicitly supplies the known origin. pyproj 3.8.0
    # omits this required CF attribute for polar stereographic variant B.
    cases["polar_sphere"].crs.attrs["latitude_of_projection_origin"] = 90.0
    cases["projected_without_latlon"] = cases["instantaneous_rate"].drop_vars(
        ["lat", "lon"]
    )
    assert tuple(cases) == CASE_IDS
    return cases


def write_reference_files(directory):
    """Serialize the independent fixtures, without a pySTEPS exporter."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for case_id, dataset in reference_cases().items():
        path = directory / f"{case_id}.nc"
        dataset.to_netcdf(path, engine="netcdf4", format="NETCDF4")
        inventory[case_id] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    (directory / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_reference_files(args.output), indent=2))
