"""Scientific and resource-ownership checks for the CF NetCDF file APIs."""

from datetime import datetime
import netCDF4
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Transformer

from pysteps import io
from pysteps.dataset_metadata import normalize_dataset, physical_variables
from pysteps.tests.cf_netcdf_reference import CASE_IDS, EPOCH, reference_cases


def reference(name, active=None):
    ds = reference_cases()[name]
    if active or name == "multiple_variables":
        ds.attrs["precip_var"] = active or "precip_accum"
    return ds


@pytest.mark.parametrize("name", CASE_IDS)
def test_import_independently_written_reference(name, tmp_path):
    source = reference(name)
    path = tmp_path / "external.nc"
    source.to_netcdf(path)
    actual = io.import_netcdf_cf(path)
    expected = normalize_dataset(source)
    for variable in physical_variables(expected):
        np.testing.assert_allclose(
            actual[variable], expected[variable], rtol=0, atol=0, equal_nan=True
        )
    assert actual[actual.attrs["precip_var"]].dtype == np.dtype("float64")
    assert CRS.from_cf(actual.crs.attrs).equals(CRS.from_cf(expected.crs.attrs))
    assert actual._close is None


@pytest.mark.parametrize("name", CASE_IDS)
def test_export_independent_reader_and_import_roundtrip(name, tmp_path):
    source = reference(name)
    snapshot = source.copy(deep=True)
    expected = normalize_dataset(source)
    path = io.export_netcdf_cf(source, tmp_path / "output.nc")
    xr.testing.assert_identical(source, snapshot)
    with xr.open_dataset(path, decode_coords="all") as independent:
        for variable in physical_variables(expected):
            np.testing.assert_allclose(
                independent[variable],
                expected[variable],
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        assert "lat" not in independent and "lon" not in independent
        assert CRS.from_cf(independent.crs.attrs).equals(
            CRS.from_cf(expected.crs.attrs)
        )
    with netCDF4.Dataset(path) as raw:
        assert raw.Conventions == "CF-1.8"
        assert raw.pysteps_write_status == "complete"
        assert raw["crs"].dimensions == ()
        assert raw["x"].dtype == raw["y"].dtype == np.dtype("float64")
        assert "_FillValue" not in raw["time"].ncattrs()
        assert raw["time"].dtype == np.dtype("float64")
        assert raw[expected.attrs["precip_var"]].filters()["complevel"] == 4
    actual = io.import_netcdf_cf(path)
    for variable in physical_variables(expected):
        np.testing.assert_allclose(
            actual[variable], expected[variable], rtol=0, atol=0, equal_nan=True
        )
    if "ens_number" in expected:
        np.testing.assert_array_equal(actual.ens_number, [0, 7])


@pytest.mark.parametrize("units, factor", [("mm h-1", 1), ("m s-1", 3600000)])
def test_packed_external_masks_are_applied_before_single_decode(
    units, factor, tmp_path
):
    ds = reference("instantaneous_rate").drop_vars(["lat", "lon"])
    path = tmp_path / "packed.nc"
    ds.to_netcdf(path)
    raw_codes = np.array([0, 1, 123, -32768, -32767, 8], dtype="int16").reshape(2, 3)
    # Define independently of both production writer and xarray packing.
    with netCDF4.Dataset(path, "a") as raw:
        var = raw.createVariable(
            "packed_rate", "i2", ("time", "y", "x"), fill_value=-32768
        )
        var.setncatts(
            dict(
                standard_name="lwe_precipitation_rate",
                units=units,
                grid_mapping="crs",
                cell_methods="time: point",
                missing_value=np.int16(-32767),
                scale_factor=0.1 / factor,
                add_offset=0.0,
            )
        )
        var.set_auto_maskandscale(False)
        var[:] = np.broadcast_to(raw_codes, (3, 2, 3))
    with pytest.warns(xr.SerializationWarning, match="multiple fill values"):
        actual = io.import_netcdf_cf(path, variable="packed_rate", dtype="single")
    expected = np.array([[0, 0.1, 12.3], [np.nan, np.nan, 0.8]], dtype="float32")
    np.testing.assert_allclose(
        actual.packed_rate[0], expected, rtol=1e-7, atol=1e-7, equal_nan=True
    )
    assert actual.packed_rate.dtype == np.dtype("float32")
    assert actual.packed_rate.attrs["pysteps_source_dtype"] == "int16"
    assert (
        actual.encoding["pysteps_source_encoding"]["packed_rate"]["scale_factor"]
        == 0.1 / factor
    )
    output = tmp_path / "decoded.nc"
    io.export_netcdf_cf(actual, output, variables="packed_rate")
    with netCDF4.Dataset(output) as raw:
        assert raw["packed_rate"].dtype == np.dtype("float32")
        assert "scale_factor" not in raw["packed_rate"].ncattrs()
        np.testing.assert_array_equal(
            np.ma.getmaskarray(raw["packed_rate"][:])[0], np.isnan(expected)
        )


@pytest.mark.parametrize("load", [True, False])
def test_file_ownership_and_lazy_data(load, tmp_path, monkeypatch):
    da = pytest.importorskip("dask.array")
    from dask.callbacks import Callback
    from pysteps.io import netcdf_cf

    path = tmp_path / "owned.nc"
    reference("instantaneous_rate").to_netcdf(path)
    opened, closed, computed = [], [], []
    real_open = xr.open_dataset

    def tracked_open(*args, **kwargs):
        ds = real_open(*args, **kwargs)
        close = ds._close

        def tracked_close():
            closed.append(True)
            close()

        ds.set_close(tracked_close)
        opened.append(ds)
        return ds

    monkeypatch.setattr(netcdf_cf.xr, "open_dataset", tracked_open)
    with Callback(pretask=lambda *args: computed.append(True)):
        actual = io.import_netcdf_cf(path, load=load)
        if not load:
            assert isinstance(actual.precip_intensity.data, da.Array)
            assert not computed and not closed
            actual.load()
            assert computed
        else:
            assert closed == [True]
        actual.close()
    assert closed == [True]
    assert len(opened) == 1


def test_lazy_zero_cumulative_origin_does_not_load_rainfall(tmp_path, monkeypatch):
    pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    ds = reference("cumulative_depth").isel(time=slice(0, 1)).copy(deep=True)
    origin = ds.forecast_reference_time.values
    ds = ds.assign_coords(time=("time", [origin], ds.time.attrs))
    ds.time_bnds.data[:] = origin
    ds.precip_cumulative.data[:] = 0
    path = tmp_path / "origin.nc"
    ds.time.encoding = {
        "units": "seconds since 2023-01-01",
        "calendar": "proleptic_gregorian",
        "dtype": "float64",
    }
    ds.time_bnds.encoding = ds.time.encoding.copy()
    ds.to_netcdf(path)
    from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

    getitem = NetCDF4ArrayWrapper.__getitem__
    accessed = []

    def tracked(self, key):
        accessed.append(self.variable_name)
        return getitem(self, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", tracked)
    with io.import_netcdf_cf(path, load=False) as actual:
        assert "precip_cumulative" not in accessed
        np.testing.assert_array_equal(actual.precip_cumulative, 0)
        assert "precip_cumulative" in accessed


def test_group_selection_and_errors_close_sources(tmp_path, monkeypatch):
    from pysteps.io import netcdf_cf

    path = tmp_path / "groups.nc"
    reference("instantaneous_rate").to_netcdf(path, group="radar")
    actual = io.import_netcdf_cf(path, group="radar")
    assert actual.precip_intensity.shape == (3, 2, 3)
    real_open = xr.open_dataset
    closed = []

    def tracked(*args, **kwargs):
        ds = real_open(*args, **kwargs)
        close = ds._close

        def done():
            closed.append(True)
            close()

        ds.set_close(done)
        return ds

    monkeypatch.setattr(netcdf_cf.xr, "open_dataset", tracked)
    with pytest.raises(ValueError, match="ambiguous"):
        io.import_netcdf_cf(path, load=False)
    assert closed == [True]


def test_missing_metadata_can_be_supplied_but_never_contradicted(tmp_path):
    ds = reference("instantaneous_rate").drop_vars(["crs", "lat", "lon"])
    ds.precip_intensity.attrs.pop("grid_mapping")
    ds = ds.isel(time=0, drop=True)
    path = tmp_path / "incomplete.nc"
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match="time coordinate|grid_mapping"):
        io.import_netcdf_cf(path)
    overrides = {
        "attrs": {"projection": "EPSG:2193"},
        "coords": {"time": EPOCH},
        "variables": {"precip_intensity": {"threshold": 0.1}},
    }
    actual = io.import_netcdf_cf(path, metadata=overrides)
    assert actual.precip_intensity.shape == (1, 2, 3)
    assert "Supplied missing metadata" in actual.attrs["history"]
    assert CRS.from_cf(actual.crs.attrs).equals(CRS.from_epsg(2193))
    overrides["variables"]["precip_intensity"]["units"] = "mm"
    with pytest.raises(ValueError, match="conflicts"):
        io.import_netcdf_cf(path, metadata=overrides)


@pytest.mark.parametrize("load", [True, False])
def test_core_rejects_invalid_physical_values_eagerly_or_at_export(load, tmp_path):
    ds = reference("instantaneous_rate")
    ds.precip_intensity.data[0, 0, 0] = -0.1
    path = tmp_path / "negative.nc"
    ds.to_netcdf(path)
    if load:
        with pytest.raises(ValueError, match="physical"):
            io.import_netcdf_cf(path)
    else:
        pytest.importorskip("dask.array")
        with io.import_netcdf_cf(path, load=False) as actual:
            with pytest.raises(ValueError, match="physical"):
                io.export_netcdf_cf(actual, tmp_path / "invalid.nc")


def test_variable_subsets_keep_required_links_and_correct_active_field(tmp_path):
    ds = reference("multiple_variables")
    path = io.export_netcdf_cf(ds, tmp_path / "subset.nc", variables="precip_intensity")
    actual = io.import_netcdf_cf(path)
    assert actual.attrs["precip_var"] == "precip_intensity"
    assert "precip_accum" not in actual
    assert "time_depth" not in actual and "time_depth_bnds" not in actual
    path = io.export_netcdf_cf(
        reference("reflectivity"), tmp_path / "dbz.nc", variables="reflectivity"
    )
    actual = io.import_netcdf_cf(path)
    assert "detection_status" in actual
    assert actual.reflectivity.attrs["ancillary_variables"] == "detection_status"
    assert actual.reflectivity[0, 0, 0] == -10
    assert actual.reflectivity[0, 0, 1] == 0


@pytest.mark.parametrize("include", [False, True])
def test_optional_geolocation_and_float_precision(include, tmp_path):
    ds = reference("projected_without_latlon")
    ds["precip_intensity"] = ds.precip_intensity.astype("float32")
    path = io.export_netcdf_cf(ds, tmp_path / "grid.nc", include_latlon=include)
    with xr.open_dataset(path, decode_coords="all") as actual:
        assert actual.precip_intensity.dtype == np.dtype("float32")
        assert ("lat" in actual) == include
        assert ("lon" in actual) == include
        if include:
            lon, lat = Transformer.from_crs(2193, 4167, always_xy=True).transform(
                *np.meshgrid(ds.x, ds.y)
            )
            np.testing.assert_allclose(actual.lon, lon)
            np.testing.assert_allclose(actual.lat, lat)


@pytest.mark.parametrize("crs", ["EPSG:2193", "EPSG:32760", "EPSG:3577"])
def test_nztm_utm_and_australian_national_albers(crs, tmp_path):
    ds = reference("projected_without_latlon")
    ds.crs.attrs = CRS(crs).to_cf()
    if crs == "EPSG:3577":
        ds = ds.assign_coords(
            x=("x", [1000000.0, 1001000.0, 1002000.0], ds.x.attrs),
            y=("y", [-3000000.0, -3001000.0], ds.y.attrs),
        )
    elif crs == "EPSG:32760":
        ds = ds.assign_coords(x=("x", [500000.0, 501000.0, 502000.0], ds.x.attrs))
    path = io.export_netcdf_cf(ds, tmp_path / "crs.nc", include_latlon=True)
    actual = io.import_netcdf_cf(path)
    assert CRS.from_cf(actual.crs.attrs).equals(CRS(crs))
    np.testing.assert_array_equal(actual.x, ds.x)
    np.testing.assert_array_equal(actual.y, ds.y)


def test_midpoint_subsecond_and_shared_source_times(tmp_path):
    ds = reference("mean_rate")
    shift = np.timedelta64(123456, "us")
    ds.time_bnds.data[:] += shift
    ends = ds.time.values + shift
    ds = ds.assign_coords(time=("time", ends - np.timedelta64(300, "s"), ds.time.attrs))
    path = io.export_netcdf_cf(ds, tmp_path / "times.nc")
    actual = io.import_netcdf_cf(path)
    np.testing.assert_array_equal(actual.time, ends)
    np.testing.assert_array_equal(actual.time_source, ends - np.timedelta64(300, "s"))
    assert actual.precip_mean_rate.attrs["accutime"] == 10
    assert actual.time.attrs["stepsize"] == 450


@pytest.mark.parametrize("fill", [0.0, -9999.0])
def test_legacy_finite_fill_view_restores_missing_on_export(fill, tmp_path):
    source = reference("instantaneous_rate")
    path = tmp_path / "source.nc"
    source.to_netcdf(path)
    view = io.import_netcdf_pysteps(path, dtype="single", fillna=fill, onerror="raise")
    assert view.precip_intensity.dtype == np.dtype("float32")
    assert not np.isnan(view.precip_intensity).any()
    mask = source.precip_intensity.isnull()
    np.testing.assert_array_equal(
        view.precip_intensity.where(mask).values[mask.values], fill
    )
    out = io.export_netcdf_cf(view, tmp_path / "restored.nc")
    actual = io.import_netcdf_cf(out)
    np.testing.assert_array_equal(actual.precip_intensity.isnull(), mask)
    assert actual.precip_intensity[0, 0, 0] == 0
    assert actual.precip_intensity[0, 0, 1] > 0


def test_onerror_dtype_and_incomplete_status(tmp_path):
    path = tmp_path / "source.nc"
    reference("instantaneous_rate").to_netcdf(path)
    with pytest.raises(ValueError, match="dtype"):
        io.import_netcdf_pysteps(path, dtype="int16", onerror="raise")
    with pytest.warns(RuntimeWarning, match="error processing"):
        assert io.import_netcdf_pysteps(tmp_path / "absent.nc") == (None, None)
    with pytest.raises(ValueError, match="onerror"):
        io.import_netcdf_pysteps(path, onerror="ignore")
    with netCDF4.Dataset(path, "a") as raw:
        raw.pysteps_write_status = "incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        io.import_netcdf_cf(path)


def test_invalid_export_is_atomic_and_overwrite_is_explicit(tmp_path):
    ds = reference("instantaneous_rate")
    path = tmp_path / "forecast.nc"
    path.write_bytes(b"prior output")
    with pytest.raises(FileExistsError):
        io.export_netcdf_cf(ds, path)
    invalid = ds.copy(deep=True)
    invalid.precip_intensity.attrs["transform"] = "dB"
    with pytest.raises(ValueError, match="untransformed"):
        io.export_netcdf_cf(invalid, path, overwrite=True)
    assert path.read_bytes() == b"prior output"
    assert not list(tmp_path.glob(".*incomplete*"))
    io.export_netcdf_cf(ds, path, overwrite=True)
    assert io.import_netcdf_cf(path).precip_intensity.shape == (3, 2, 3)


@pytest.mark.parametrize(
    "encoding, message",
    [
        ({"precip_intensity": {"dtype": "int16"}}, "floating"),
        ({"precip_intensity": {"scale_factor": 0.1}}, "stage 5"),
        ({"precip_intensity": {"_FillValue": 0}}, "collide"),
        ({"precip_intensity": {"chunksizes": (9, 2, 3)}}, "chunksizes"),
        ({"absent": {}}, "outside"),
    ],
)
def test_unsafe_or_unsupported_encodings_fail_before_creation(
    encoding, message, tmp_path
):
    with pytest.raises(ValueError, match=message):
        io.export_netcdf_cf(
            reference("instantaneous_rate"), tmp_path / "bad.nc", encoding=encoding
        )
    assert not list(tmp_path.iterdir())


def legacy(tmp_path, mode, *, unit="mm/h", encoding=None):
    return io.initialize_forecast_exporter_netcdf(
        str(tmp_path),
        "legacy",
        datetime(2023, 1, 1),
        7.5,
        [1, 2, 4],
        (2, 3),
        {
            "projection": "EPSG:2193",
            "x1": 1600000.0,
            "x2": 1603000.0,
            "y1": 5900000.0,
            "y2": 5902000.0,
            "unit": unit,
            "yorigin": "upper",
            "cartesian_unit": "m",
            **({"accutime": 10} if unit == "mm" else {}),
        },
        n_ens_members=2,
        incremental=mode,
        ensemble_ids=[0, 7],
        **(encoding or {}),
    )


@pytest.mark.parametrize("mode", [None, "timestep", "member"])
@pytest.mark.parametrize("unit", ["mm/h", "mm", "dBZ"])
def test_legacy_write_modes_have_identical_schema_and_masks(mode, unit, tmp_path):
    writer = legacy(tmp_path, mode, unit=unit)
    values = np.arange(36, dtype=float).reshape(2, 3, 2, 3) / 10
    if unit == "dBZ":
        values -= 2
    values[0, 0, 1, 2] = np.nan
    assert not (tmp_path / "legacy.nc").exists()
    if mode is None:
        io.export_forecast_dataset(values, writer)
    elif mode == "timestep":
        for index in range(3):
            io.export_forecast_dataset(values[:, index], writer)
    else:
        for index in range(2):
            io.export_forecast_dataset(values[index], writer)
    io.close_forecast_files(writer)
    actual = io.import_netcdf_cf(tmp_path / "legacy.nc", dtype="single")
    variable = actual.attrs["precip_var"]
    np.testing.assert_allclose(
        actual[variable], values, rtol=1e-7, atol=1e-7, equal_nan=True
    )
    np.testing.assert_array_equal(actual.ens_number, [0, 7])
    np.testing.assert_array_equal(
        actual.time, EPOCH + np.array([450, 900, 1800], dtype="timedelta64[s]")
    )
    assert actual.forecast_reference_time == EPOCH
    if unit == "mm":
        assert actual[variable].attrs["accutime"] == 10
        np.testing.assert_array_equal(
            np.diff(actual.time_bnds, axis=1), np.timedelta64(600, "s")
        )
    else:
        assert "time_bnds" not in actual


@pytest.mark.parametrize(
    "failure", ["incomplete", "shape", "negative", "too_many", "overflow"]
)
def test_incremental_failures_close_and_remove_unpublished_files(failure, tmp_path):
    writer = legacy(
        tmp_path,
        "timestep",
        encoding=(
            {"datatype": "int16", "fill_value": -32768, "scale_factor": 0.1}
            if failure == "overflow"
            else None
        ),
    )
    with pytest.raises(ValueError):
        if failure == "incomplete":
            io.close_forecast_files(writer)
        elif failure == "shape":
            io.export_forecast_dataset(np.ones((2, 3)), writer)
        elif failure == "negative":
            io.export_forecast_dataset(-np.ones((2, 2, 3)), writer)
        elif failure == "overflow":
            io.export_forecast_dataset(np.full((2, 2, 3), 4000.0), writer)
        else:
            for _ in range(4):
                io.export_forecast_dataset(np.ones((2, 2, 3)), writer)
    assert writer["_cf_writer"].closed
    assert not list(tmp_path.iterdir())


def test_registry_exposes_new_apis():
    assert io.get_method("netcdf_cf", "importer") is io.import_netcdf_cf
    assert io.get_method("netcdf_cf", "exporter") is io.export_netcdf_cf


def test_auxiliary_flags_preserve_missing_codes_with_classic_storage(tmp_path):
    source = reference("reflectivity")
    path = io.export_netcdf_cf(source, tmp_path / "flags.nc")
    with netCDF4.Dataset(path) as raw:
        flags = raw["detection_status"]
        assert flags.dtype == flags.flag_values.dtype == np.dtype("int8")
        np.testing.assert_array_equal(
            np.ma.getmaskarray(flags[:]), np.isnan(source.detection_status)
        )
        np.testing.assert_array_equal(
            flags[:].compressed(),
            source.detection_status.values[np.isfinite(source.detection_status)],
        )
    actual = io.import_netcdf_cf(path)
    np.testing.assert_array_equal(
        actual.detection_status.isnull(), source.detection_status.isnull()
    )


def test_bounds_inherit_the_same_time_serialization(tmp_path):
    path = io.export_netcdf_cf(reference("interval_depth"), tmp_path / "bounds.nc")
    with netCDF4.Dataset(path) as raw:
        assert "units" not in raw["time_bnds"].ncattrs()
        assert "calendar" not in raw["time_bnds"].ncattrs()
    actual = io.import_netcdf_cf(path)
    np.testing.assert_array_equal(
        actual.time_bnds, reference("interval_depth").time_bnds
    )


@pytest.mark.parametrize("link", ["bounds", "grid_mapping", "ancillary_variables"])
def test_import_rejects_dangling_links_before_xarray_can_drop_them(link, tmp_path):
    ds = reference("instantaneous_rate")
    target = ds.time if link == "bounds" else ds.precip_intensity
    target.attrs[link] = "missing_dependency"
    path = tmp_path / "dangling.nc"
    ds.to_netcdf(path)
    with pytest.raises(ValueError, match="Dangling"):
        io.import_netcdf_cf(path)


def test_scalar_cf_time_and_realization_are_retained_as_singleton_axes(tmp_path):
    ds = reference("ensemble_rate").isel(time=0, ens_number=0)
    ds.precip_intensity.attrs["coordinates"] = "time ens_number"
    path = tmp_path / "scalar.nc"
    ds.to_netcdf(path)
    actual = io.import_netcdf_cf(path)
    assert actual.precip_intensity.dims == ("ens_number", "time", "y", "x")
    assert actual.precip_intensity.shape == (1, 1, 2, 3)
    assert actual.ens_number[0] == 0
    assert actual.time[0] == ds.time


def test_kilometre_bounds_normalize_numerically_and_in_metadata(tmp_path):
    ds = reference("projected_without_latlon")
    for axis in ("x", "y"):
        coordinate = ds[axis]
        values = coordinate.values / 1000
        name = axis + "_bnds"
        ds = ds.assign_coords(
            {axis: (axis, values, {**coordinate.attrs, "units": "km", "bounds": name})}
        )
        ds[name] = (
            (axis, "nv"),
            np.column_stack([values - 0.5, values + 0.5]),
            {"units": "km"},
        )
    path = tmp_path / "km.nc"
    ds.to_netcdf(path)
    # Explicitly include redundant bound units in the external file; xarray
    # normally omits them in favour of inheritance when writing our fixture.
    with netCDF4.Dataset(path, "a") as raw:
        raw["x_bnds"].units = "km"
        raw["y_bnds"].units = "km"
    actual = io.import_netcdf_cf(path)
    np.testing.assert_array_equal(actual.x, [1600000, 1601000, 1602000])
    assert actual.x_bnds.attrs["units"] == "m"
    io.export_netcdf_cf(actual, tmp_path / "metres.nc")


def test_write_failure_preserves_existing_output_and_closes_handle(
    tmp_path, monkeypatch
):
    from pysteps.io.netcdf_cf import _CFWriter

    path = tmp_path / "existing.nc"
    path.write_bytes(b"previous complete forecast")
    captured = []

    def fail(self, arrays):
        captured.append(self)
        raise OSError("simulated disk failure")

    monkeypatch.setattr(_CFWriter, "write", fail)
    with pytest.raises(OSError, match="disk failure"):
        io.export_netcdf_cf(reference("instantaneous_rate"), path, overwrite=True)
    assert path.read_bytes() == b"previous complete forecast"
    assert captured[0].closed and captured[0].ncfile is None
    assert not list(tmp_path.glob(".*incomplete*"))


def test_concurrent_publication_cannot_overwrite_an_existing_file(tmp_path):
    exporter = legacy(tmp_path, None)
    io.export_forecast_dataset(np.ones((2, 3, 2, 3)), exporter)
    path = tmp_path / "legacy.nc"
    path.write_bytes(b"concurrent output")
    with pytest.raises(FileExistsError):
        io.close_forecast_files(exporter)
    assert path.read_bytes() == b"concurrent output"
    assert not list(tmp_path.glob(".*incomplete*"))


@pytest.mark.parametrize("name", ["instantaneous_rate", "mean_rate"])
def test_registered_cf_importer_stacks_archived_singleton_frames(name, tmp_path):
    source = reference(name)
    files = []
    for index in range(3):
        path = tmp_path / f"frame-{index}.nc"
        source.isel(time=slice(index, index + 1)).to_netcdf(path)
        files.append(path)
    dates = [
        time.astype("datetime64[us]").astype(datetime) for time in source.time.values
    ]
    actual = io.read_timeseries((files, dates), io.get_method("netcdf_cf", "importer"))
    variable = (
        source.attrs["precip_var"]
        if "precip_var" in source.attrs
        else "precip_intensity" if name == "instantaneous_rate" else "precip_mean_rate"
    )
    np.testing.assert_allclose(actual[variable], source[variable], equal_nan=True)
    assert actual.time.attrs["stepsize"] == 450
    if name == "mean_rate":
        np.testing.assert_array_equal(actual.time_bnds, source.time_bnds)
        assert actual[variable].attrs["accutime"] == 10
    bad_dates = dates.copy()
    bad_dates[0] = dates[0] - (dates[1] - dates[0])
    with pytest.raises(ValueError, match="evenly spaced|archive date"):
        io.read_timeseries((files, bad_dates), io.get_method("netcdf_cf", "importer"))


def test_cf_archive_missing_frame_remains_missing_with_its_support(tmp_path):
    source = reference("mean_rate")
    files = []
    for index in (0, 2):
        path = tmp_path / f"frame-{index}.nc"
        source.isel(time=slice(index, index + 1)).to_netcdf(path)
        files.append(path)
    dates = [
        time.astype("datetime64[us]").astype(datetime) for time in source.time.values
    ]
    actual = io.read_timeseries(
        ([files[0], None, files[1]], dates), io.import_netcdf_cf
    )
    assert np.isnan(actual.precip_mean_rate[1]).all()
    np.testing.assert_array_equal(actual.time_bnds, source.time_bnds)


@pytest.mark.parametrize("scale", [0.1, -0.1])
def test_packed_valid_range_is_applied_before_unit_normalization(scale, tmp_path):
    source = reference("instantaneous_rate").drop_vars(["lat", "lon"])
    path = tmp_path / "range.nc"
    source.to_netcdf(path)
    with netCDF4.Dataset(path, "a") as raw:
        value = raw.createVariable(
            "bounded", "i2", ("time", "y", "x"), fill_value=-32768
        )
        value.setncatts(
            {
                "standard_name": "lwe_precipitation_rate",
                "units": "mm s-1",
                "cell_methods": "time: point",
                "grid_mapping": "crs",
                "valid_range": np.array([0, 10], dtype="int16"),
                "scale_factor": scale,
                "add_offset": 1.0,
            }
        )
        value.set_auto_maskandscale(False)
        value[:] = np.broadcast_to(
            np.array([[0, 5, 10], [11, -1, -32768]], dtype="int16"), (3, 2, 3)
        )
    actual = io.import_netcdf_cf(path, variable="bounded")
    expected = (
        np.array([[1, 1 + 5 * scale, 1 + 10 * scale], [np.nan, np.nan, np.nan]]) * 3600
    )
    np.testing.assert_allclose(actual.bounded[0], expected, equal_nan=True)
    assert "valid_range" not in actual.bounded.attrs
    np.testing.assert_array_equal(
        actual.encoding["pysteps_source_encoding"]["bounded"]["valid_range"], [0, 10]
    )
    io.export_netcdf_cf(actual, tmp_path / "normalized.nc", variables="bounded")


def test_partial_geolocation_is_completed_when_requested(tmp_path):
    source = reference("instantaneous_rate").drop_vars("lon")
    path = io.export_netcdf_cf(source, tmp_path / "geolocation.nc", include_latlon=True)
    actual = io.import_netcdf_cf(path)
    assert "lat" in actual and "lon" in actual
    np.testing.assert_array_equal(actual.lat, source.lat)


def test_physical_valid_range_scales_with_direct_dataset_units(tmp_path):
    source = reference("unit_alias_rate")
    source.precip_intensity.attrs["valid_range"] = np.array([0.0, 1.0])
    path = io.export_netcdf_cf(source, tmp_path / "scaled-range.nc")
    with netCDF4.Dataset(path) as raw:
        np.testing.assert_array_equal(
            raw["precip_intensity"].valid_range, [0.0, 3600000.0]
        )


@pytest.mark.parametrize(
    "encoding, message",
    [
        ({"x": {"dtype": "float32"}}, "coordinates require float64"),
        ({"ens_number": {"dtype": "float64"}}, "coordinates require int32"),
        (
            {"precip_intensity": {"dtype": "float32", "_FillValue": 1e100}},
            "Unrepresentable fill",
        ),
    ],
)
def test_coordinate_and_fill_encodings_cannot_break_the_profile(
    encoding, message, tmp_path
):
    with pytest.raises(ValueError, match=message):
        io.export_netcdf_cf(
            reference("ensemble_rate"), tmp_path / "invalid.nc", encoding=encoding
        )
    assert not list(tmp_path.iterdir())


def test_inactive_fields_are_serialized_in_cf_dimension_order(tmp_path):
    source = reference("multiple_variables")
    source["precip_intensity"] = source.precip_intensity.transpose("y", "x", "time")
    path = io.export_netcdf_cf(source, tmp_path / "order.nc")
    actual = io.import_netcdf_cf(path)
    assert actual.precip_intensity.dims == ("time_aux", "y", "x")
    np.testing.assert_allclose(
        actual.precip_intensity,
        source.precip_intensity.transpose("time", "y", "x"),
        equal_nan=True,
    )


def test_flag_metadata_is_not_silently_cast_out_of_range(tmp_path):
    source = reference("reflectivity")
    source.detection_status.attrs["flag_values"] = np.array(
        [0, 1, 2**40], dtype="int64"
    )
    source.detection_status.attrs["flag_meanings"] = "not_detected detected reserved"
    with pytest.raises(ValueError, match="Flag codes"):
        io.export_netcdf_cf(source, tmp_path / "flags.nc")
    assert not list(tmp_path.iterdir())


def test_incomplete_earth_model_requires_an_explicit_noncontradictory_crs(tmp_path):
    source = reference("projected_without_latlon")
    attrs = source.crs.attrs
    for key in (
        "crs_wkt",
        "earth_radius",
        "semi_major_axis",
        "semi_minor_axis",
        "inverse_flattening",
    ):
        attrs.pop(key, None)
    path = tmp_path / "unspecified-earth.nc"
    source.to_netcdf(path)
    with pytest.raises(ValueError, match="earth model is missing"):
        io.import_netcdf_cf(path)
    actual = io.import_netcdf_cf(path, metadata={"attrs": {"projection": "EPSG:2193"}})
    assert CRS.from_cf(actual.crs.attrs).equals(CRS.from_epsg(2193))
    with netCDF4.Dataset(path, "a") as raw:
        raw["crs"].false_easting += 1
    with pytest.raises(ValueError, match="conflicts"):
        io.import_netcdf_cf(path, metadata={"attrs": {"projection": "EPSG:2193"}})


def test_finite_fill_view_rejects_dtype_overflow(tmp_path):
    path = tmp_path / "input.nc"
    reference("instantaneous_rate").to_netcdf(path)
    with pytest.raises(ValueError, match="fillna cannot be represented"):
        io.import_netcdf_pysteps(path, dtype="single", fillna=1e100, onerror="raise")
