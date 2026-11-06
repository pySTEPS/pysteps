"""Inspect raw storage codes independently of CF decoders."""

from datetime import datetime

import netCDF4
import numpy as np
import pytest
import xarray as xr

from pysteps import io
from pysteps.dataset_metadata import normalize_dataset
from pysteps.tests.cf_netcdf_reference import reference_cases

FILL = np.int16(-32768)


def source(name="instantaneous_rate", values=None, dtype="float64"):
    """Return a physical reference dataset with the requested values and dtype."""
    ds = reference_cases()[name].copy(deep=True)
    variable = {
        "interval_depth": "precip_accum",
        "cumulative_depth": "precip_cumulative",
        "reflectivity": "reflectivity",
        "mean_rate": "precip_mean_rate",
    }.get(name, "precip_intensity")
    ds.attrs["precip_var"] = variable
    if values is not None:
        data = np.broadcast_to(
            np.asarray(values, dtype=dtype).reshape(2, 3), ds[variable].shape
        ).copy()
        ds[variable] = (ds[variable].dims, data, ds[variable].attrs)
    else:
        ds[variable] = ds[variable].astype(dtype)
    return ds, variable


def packing(variable, step=0.1, **kwargs):
    """Build explicit int16 packing options for a physical variable."""
    return {variable: {"dtype": "int16", "scale_factor": step, **kwargs}}


def assert_decoders(path, variable, expected, codes, step, offset=0, fill=FILL):
    """Check raw codes and physical values with independent NetCDF decoders."""
    missing = np.isnan(expected)
    with netCDF4.Dataset(path) as raw:
        var = raw[variable]
        assert var.dtype == np.dtype("int16")
        assert np.asarray(var._FillValue).dtype == np.dtype("int16")
        assert var._FillValue == fill
        assert var.scale_factor == step
        assert var.add_offset == offset
        assert np.asarray(var.scale_factor).dtype == np.dtype("float64")
        assert np.asarray(var.add_offset).dtype == np.dtype("float64")
        assert "_Unsigned" not in var.ncattrs()
        decoded = var[:]
        np.testing.assert_array_equal(np.ma.getmaskarray(decoded), missing)
        np.testing.assert_allclose(
            decoded.filled(np.nan), expected, rtol=0, atol=1e-12, equal_nan=True
        )
        var.set_auto_maskandscale(False)
        np.testing.assert_array_equal(var[:], codes)
        np.testing.assert_array_equal(var[:] == fill, missing)
    with xr.open_dataset(path, decode_coords="all") as external:
        np.testing.assert_allclose(
            external[variable], expected, rtol=0, atol=1e-12, equal_nan=True
        )
        np.testing.assert_array_equal(external[variable].isnull(), missing)
    actual = io.import_netcdf_cf(path)
    np.testing.assert_allclose(
        actual[variable], expected, rtol=0, atol=1e-12, equal_nan=True
    )
    np.testing.assert_array_equal(actual[variable].isnull(), missing)
    return actual


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("fill", [-32768, -9999])
def test_int16_scale_and_missing_code_are_separate_operations(dtype, fill, tmp_path):
    """Verify int16 scale and missing code are separate operations."""
    values = [[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]]
    ds, variable = source(values=values, dtype=dtype)
    snapshot = ds.copy(deep=True)
    path = io.export_netcdf_cf(
        ds, tmp_path / "packed.nc", encoding=packing(variable, _FillValue=fill)
    )
    codes = np.broadcast_to(
        np.array([[0, 0, 1], [2, 13, fill]], dtype="int16"), ds[variable].shape
    )
    expected = np.broadcast_to(
        np.array([[0.0, 0.0, 0.1], [0.2, 1.3, np.nan]]), ds[variable].shape
    )
    actual = assert_decoders(path, variable, expected, codes, 0.1, fill=fill)
    xr.testing.assert_identical(ds, snapshot)
    assert actual[variable].attrs["pysteps_source_dtype"] == "int16"
    # Positive 0.04 quantizes to zero but remains observed, never missing.
    assert not actual[variable].isnull()[0, 0, 1]
    assert actual[variable][0, 0, 1] == 0
    assert actual[variable].isnull()[0, 1, 2]


def test_packing_terms_are_stored_with_explicit_float64_precision(tmp_path):
    """Use one declared unpacked dtype even when callers supply float32 scalars."""
    ds, variable = source()
    path = io.export_netcdf_cf(
        ds,
        tmp_path / "packing-attribute-types.nc",
        encoding=packing(
            variable,
            np.float32(0.1),
            add_offset=np.float32(0.0),
        ),
    )
    with netCDF4.Dataset(path) as raw:
        var = raw[variable]
        assert np.asarray(var.scale_factor).dtype == np.dtype("float64")
        assert np.asarray(var.add_offset).dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "name, step, offset, values, raw_codes",
    [
        (
            "interval_depth",
            0.01,
            0.0,
            [[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]],
            [[0, 4, 6], [25, 126, -32768]],
        ),
        (
            "cumulative_depth",
            0.01,
            0.0,
            [[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]],
            [[0, 4, 6], [25, 126, -32768]],
        ),
        (
            "reflectivity",
            0.1,
            10.0,
            [[-10.0, 0.0, 20.0], [40.0, -32.0, np.nan]],
            [[-200, -100, 100], [300, -420, -32768]],
        ),
        (
            "mean_rate",
            0.1,
            2.0,
            [[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]],
            [[-20, -20, -19], [-18, -7, -32768]],
        ),
    ],
)
def test_precision_is_explicit_for_each_quantity_and_offset(
    name, step, offset, values, raw_codes, tmp_path
):
    """Verify precision is explicit for each quantity and offset."""
    ds, variable = source(name, values)
    path = io.export_netcdf_cf(
        ds,
        tmp_path / "quantity.nc",
        encoding=packing(variable, step, add_offset=offset),
    )
    codes = np.broadcast_to(np.array(raw_codes, dtype="int16"), ds[variable].shape)
    expected = codes.astype(float) * step + offset
    expected = np.where(codes == FILL, np.nan, expected)
    actual = assert_decoders(path, variable, expected, codes, step, offset)
    for coordinate in ("time", "time_bnds", "forecast_reference_time"):
        if coordinate in ds:
            np.testing.assert_array_equal(
                actual[coordinate], normalize_dataset(ds)[coordinate]
            )
    if name == "reflectivity":
        np.testing.assert_array_equal(actual.detection_status, ds.detection_status)


@pytest.mark.parametrize("state, code", [(0.0, 0), (np.nan, -32768)])
def test_all_dry_and_all_missing_are_not_interchangeable(state, code, tmp_path):
    """Verify all dry and all missing are not interchangeable."""
    ds, variable = source(values=np.full((2, 3), state))
    path = io.export_netcdf_cf(ds, tmp_path / "uniform.nc", encoding=packing(variable))
    codes = np.full(ds[variable].shape, code, dtype="int16")
    expected = np.full(ds[variable].shape, state)
    assert_decoders(path, variable, expected, codes, 0.1)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_rounding_uses_ties_to_even_without_a_half_up_bias(dtype, tmp_path):
    # Binary-exact step and half steps make the tie expectations unambiguous.
    """Verify rounding uses ties to even without a half up bias."""
    ds, variable = source(
        values=[[0.125, 0.375, 0.625], [0.875, 0.0, np.nan]], dtype=dtype
    )
    path = io.export_netcdf_cf(
        ds, tmp_path / "ties.nc", encoding=packing(variable, 0.25)
    )
    codes = np.broadcast_to(
        np.array([[0, 2, 2], [4, 0, -32768]], dtype="int16"), ds[variable].shape
    )
    expected = np.where(codes == FILL, np.nan, codes.astype(float) * 0.25)
    assert_decoders(path, variable, expected, codes, 0.25)


def test_values_either_side_of_a_tie_and_upper_range(tmp_path):
    """Verify values either side of a tie and upper range."""
    ds, variable = source(
        values=[
            [np.nextafter(0.375, -np.inf), 0.375, np.nextafter(0.375, np.inf)],
            [8191.75, 8191.8, np.nan],
        ]
    )
    path = io.export_netcdf_cf(
        ds, tmp_path / "edges.nc", encoding=packing(variable, 0.25)
    )
    codes = np.broadcast_to(
        np.array([[1, 2, 2], [32767, 32767, -32768]], dtype="int16"), ds[variable].shape
    )
    expected = np.where(codes == FILL, np.nan, codes.astype(float) * 0.25)
    assert_decoders(path, variable, expected, codes, 0.25)


@pytest.mark.parametrize("value", [3276.7, 3276.74])
def test_range_check_uses_rounded_code_before_int16_cast(value, tmp_path):
    """Verify range check uses rounded code before int16 cast."""
    ds, variable = source(values=np.full((2, 3), value))
    path = io.export_netcdf_cf(ds, tmp_path / "maximum.nc", encoding=packing(variable))
    with netCDF4.Dataset(path) as raw:
        var = raw[variable]
        var.set_auto_maskandscale(False)
        np.testing.assert_array_equal(
            var[:], np.full(ds[variable].shape, 32767, dtype="int16")
        )
    np.testing.assert_allclose(io.import_netcdf_cf(path)[variable], 3276.7, atol=1e-12)


@pytest.mark.parametrize(
    "value, name, message",
    [
        (3276.75, "instantaneous_rate", "overflow"),
        (3276.8, "instantaneous_rate", "overflow"),
        (1e100, "instantaneous_rate", "overflow"),
        (-0.001, "instantaneous_rate", "physical"),
        (-3276.8, "reflectivity", "collide"),
        (-3276.9, "reflectivity", "overflow"),
        (np.inf, "instantaneous_rate", "physical"),
        (-np.inf, "reflectivity", "physical"),
    ],
)
def test_overflow_fill_collision_and_invalid_rain_never_wrap(
    value, name, message, tmp_path
):
    """Verify overflow fill collision and invalid rain never wrap."""
    ds, variable = source(name, np.full((2, 3), value))
    path = tmp_path / "protected.nc"
    path.write_bytes(b"previous completed forecast")
    with pytest.raises(ValueError, match=message):
        io.export_netcdf_cf(ds, path, encoding=packing(variable), overwrite=True)
    assert path.read_bytes() == b"previous completed forecast"
    assert not list(tmp_path.glob(".*incomplete*"))


@pytest.mark.parametrize(
    "options, message",
    [
        ({"dtype": "int16"}, "explicit scale_factor"),
        ({"dtype": "int16", "scale_factor": 0}, "Invalid scale_factor"),
        ({"dtype": "int16", "scale_factor": -0.1}, "Invalid scale_factor"),
        ({"dtype": "int16", "scale_factor": np.nan}, "Invalid scale_factor"),
        ({"dtype": "int16", "scale_factor": np.inf}, "Invalid scale_factor"),
        (
            {"dtype": "int16", "scale_factor": 0.1, "add_offset": np.inf},
            "Invalid add_offset",
        ),
        (
            {"dtype": "int16", "scale_factor": 0.1, "_FillValue": 65535},
            "Unrepresentable fill",
        ),
        (
            {"dtype": "int16", "scale_factor": 0.1, "_FillValue": -32768.5},
            "Unrepresentable fill",
        ),
        (
            {"dtype": "int16", "scale_factor": 0.1, "_FillValue": None},
            "reserved _FillValue",
        ),
        ({"dtype": "int32", "scale_factor": 0.1}, "requires int16"),
        ({"dtype": "uint16", "scale_factor": 0.1}, "require int16"),
        ({"dtype": "float32", "scale_factor": 0.1}, "require int16"),
    ],
)
def test_invalid_packing_schema_leaves_no_file(options, message, tmp_path):
    """Verify invalid packing schema leaves no file."""
    ds, variable = source()
    with pytest.raises(ValueError, match=message):
        io.export_netcdf_cf(ds, tmp_path / "invalid.nc", encoding={variable: options})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fill, offset", [(0, 0.0), (-9999, 1000.0)])
def test_custom_fill_collisions_are_checked_after_quantization(fill, offset, tmp_path):
    """Verify custom fill collisions are checked after quantization."""
    values = [[0.0, 0.04, 0.06], [0.1, 1.26, np.nan]]
    ds, variable = source(values=values)
    with pytest.raises(ValueError, match="collide"):
        io.export_netcdf_cf(
            ds,
            tmp_path / "collision.nc",
            encoding=packing(variable, _FillValue=fill, add_offset=offset),
        )
    assert not list(tmp_path.iterdir())


def test_fill_zero_is_legal_when_valid_codes_do_not_use_zero(tmp_path):
    """Verify fill zero is legal when valid codes do not use zero."""
    ds, variable = source(values=[[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]])
    path = io.export_netcdf_cf(
        ds,
        tmp_path / "zero-fill.nc",
        encoding=packing(variable, _FillValue=0, add_offset=2.0),
    )
    codes = np.broadcast_to(
        np.array([[-20, -20, -19], [-18, -7, 0]], dtype="int16"), ds[variable].shape
    )
    expected = np.where(codes == 0, np.nan, codes.astype(float) * 0.1 + 2.0)
    assert_decoders(path, variable, expected, codes, 0.1, offset=2.0, fill=0)


def test_unpack_repack_and_float_reexport_do_not_double_scale(tmp_path):
    """Verify unpack repack and float reexport do not double scale."""
    ds, variable = source(values=[[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]])
    first = io.export_netcdf_cf(ds, tmp_path / "first.nc", encoding=packing(variable))
    imported = io.import_netcdf_cf(first)
    second = io.export_netcdf_cf(
        imported, tmp_path / "second.nc", encoding=packing(variable)
    )
    with netCDF4.Dataset(first) as a, netCDF4.Dataset(second) as b:
        a[variable].set_auto_maskandscale(False)
        b[variable].set_auto_maskandscale(False)
        np.testing.assert_array_equal(a[variable][:], b[variable][:])
    floating = io.export_netcdf_cf(imported, tmp_path / "float.nc")
    with netCDF4.Dataset(floating) as raw:
        assert raw[variable].dtype == np.dtype("float64")
        assert "scale_factor" not in raw[variable].ncattrs()
    np.testing.assert_allclose(
        io.import_netcdf_cf(floating)[variable], imported[variable], equal_nan=True
    )


@pytest.mark.parametrize("fill_view", [0.0, -9999.0])
def test_finite_fill_view_is_restored_before_packing(fill_view, tmp_path):
    """Verify finite fill view is restored before packing."""
    ds, variable = source(values=[[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]])
    ds.to_netcdf(tmp_path / "input.nc")
    view = io.import_netcdf_pysteps(
        tmp_path / "input.nc", fillna=fill_view, onerror="raise"
    )
    path = io.export_netcdf_cf(view, tmp_path / "view.nc", encoding=packing(variable))
    with netCDF4.Dataset(path) as raw:
        raw[variable].set_auto_maskandscale(False)
        np.testing.assert_array_equal(raw[variable][:] == FILL, np.isnan(ds[variable]))
        assert raw[variable][0, 0, 0] == 0


def test_supplied_detection_flags_preserve_unrounded_classification(tmp_path):
    """Verify supplied detection flags preserve unrounded classification."""
    ds, variable = source(values=[[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]])
    status = np.where(
        np.isnan(ds[variable]), 9, np.where(ds[variable] > 0, 1, 0)
    ).astype("int8")
    ds["source_detection"] = (
        ds[variable].dims,
        status,
        {
            "long_name": "source rain detection before quantization",
            "units": "1",
            "flag_values": np.array([0, 1, 9], dtype="int8"),
            "flag_meanings": "dry detected missing",
        },
    )
    ds[variable].attrs["ancillary_variables"] = "source_detection"
    path = io.export_netcdf_cf(
        ds, tmp_path / "classification.nc", encoding=packing(variable)
    )
    actual = io.import_netcdf_cf(path)
    np.testing.assert_array_equal(actual.source_detection, status)
    assert actual[variable][0, 0, 1] == 0 and actual.source_detection[0, 0, 1] == 1


def test_unsigned_source_marker_cannot_reinterpret_signed_dbz(tmp_path):
    """Verify unsigned source marker cannot reinterpret signed dbz."""
    ds, variable = source("reflectivity")
    ds[variable].attrs["_Unsigned"] = "true"
    path = io.export_netcdf_cf(ds, tmp_path / "signed.nc", encoding=packing(variable))
    actual = io.import_netcdf_cf(path)
    assert actual[variable][0, 0, 0] == -10
    assert actual[variable][0, 1, 1] == -32
    assert actual[variable].isnull()[0, 1, 2]


@pytest.mark.parametrize("name", ["instantaneous_rate", "reflectivity"])
def test_quantization_error_bound_across_every_nonmissing_int16_code(name, tmp_path):
    """Verify quantization error bound across every nonmissing int16 code."""
    rng = np.random.default_rng(8192)
    integers = (
        np.arange(32768) if name == "instantaneous_rate" else np.arange(-32767, 32768)
    )
    values = integers.astype(float) * 0.1 + rng.uniform(-0.049, 0.049, len(integers))
    if name == "instantaneous_rate":
        values[0] = 0
        shape, padding = (128, 257), 128
    else:
        shape, padding = (256, 256), 1
    values = np.concatenate([values, np.full(padding, np.nan)])
    base, variable = source(name)
    attrs = base[variable].attrs.copy()
    attrs.pop("ancillary_variables", None)
    ds = base.drop_vars(["lat", "lon"]).isel(time=slice(0, 1)).drop_dims(["x", "y"])
    ds = ds.assign_coords(
        x=(
            "x",
            1600000.0 + np.arange(shape[1]) * 1000,
            {
                "standard_name": "projection_x_coordinate",
                "units": "m",
                "stepsize": 1000.0,
            },
        ),
        y=(
            "y",
            5900000.0 + np.arange(shape[0]) * 1000,
            {
                "standard_name": "projection_y_coordinate",
                "units": "m",
                "stepsize": 1000.0,
            },
        ),
    )
    ds[variable] = (("time", "y", "x"), values.reshape(1, *shape), attrs)
    path = io.export_netcdf_cf(
        ds, tmp_path / "all-codes.nc", encoding=packing(variable)
    )
    actual = io.import_netcdf_cf(path)[variable].values.ravel()
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(values))
    assert np.nanmax(np.abs(actual - values)) <= 0.05 + 1e-11
    with netCDF4.Dataset(path) as raw:
        raw[variable].set_auto_maskandscale(False)
        expected = np.concatenate(
            [integers.astype("int16"), np.full(padding, FILL, dtype="int16")]
        )
        np.testing.assert_array_equal(raw[variable][:].ravel(), expected)


@pytest.mark.parametrize("mode", [None, "timestep", "member"])
def test_legacy_region_writes_pack_once_and_preserve_missing(mode, tmp_path):
    """Verify legacy region writes pack once and preserve missing."""
    values = np.broadcast_to(
        np.array([[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]]), (2, 3, 2, 3)
    ).copy()
    metadata = dict(
        projection="EPSG:2193",
        x1=1600000.0,
        x2=1603000.0,
        y1=5900000.0,
        y2=5902000.0,
        cartesian_unit="m",
        yorigin="upper",
        unit="mm/h",
    )
    exporter = io.initialize_forecast_exporter_netcdf(
        str(tmp_path),
        "legacy",
        datetime(2023, 1, 1),
        7.5,
        [1, 2, 4],
        (2, 3),
        metadata,
        n_ens_members=2,
        datatype="int16",
        fill_value=-32768,
        scale_factor=0.1,
        offset=2.0,
        incremental=mode,
        ensemble_ids=[0, 7],
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
    codes = np.broadcast_to(
        np.array([[-20, -20, -19], [-18, -7, -32768]], dtype="int16"), values.shape
    )
    expected = np.where(codes == FILL, np.nan, codes.astype(float) * 0.1 + 2.0)
    actual = assert_decoders(
        tmp_path / "legacy.nc", "precip_intensity", expected, codes, 0.1, 2.0
    )
    np.testing.assert_array_equal(actual.ens_number, [0, 7])


@pytest.mark.parametrize(
    "declared", [{"valid_range": [0.0, 2.0]}, {"valid_min": 0.0, "valid_max": 2.0}]
)
def test_validity_attributes_are_in_packed_domain(declared, tmp_path):
    """Verify validity attributes are in packed domain."""
    ds, variable = source(values=[[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]])
    ds[variable].attrs.update(declared)
    path = io.export_netcdf_cf(
        ds, tmp_path / "validity.nc", encoding=packing(variable, add_offset=2.0)
    )
    with netCDF4.Dataset(path) as raw:
        for key in declared:
            assert np.asarray(raw[variable].getncattr(key)).dtype == np.dtype("int16")
        if "valid_range" in declared:
            np.testing.assert_array_equal(raw[variable].valid_range, [-20, 0])
        else:
            assert raw[variable].valid_min == -20 and raw[variable].valid_max == 0
        assert np.ma.getmaskarray(raw[variable][:])[0, 1, 2]
        assert not np.ma.getmaskarray(raw[variable][:])[0, 0, 0]
    actual = io.import_netcdf_cf(path)
    np.testing.assert_array_equal(actual[variable].isnull(), ds[variable].isnull())


def test_invalid_declared_physical_range_cannot_erase_valid_values_on_read(tmp_path):
    """Verify invalid declared physical range cannot erase valid values on read."""
    ds, variable = source(values=[[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]])
    ds[variable].attrs["valid_range"] = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="declared valid range"):
        io.export_netcdf_cf(ds, tmp_path / "range.nc", encoding=packing(variable))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("gain", [np.nan, np.inf])
def test_invalid_source_gain_is_not_accepted_as_an_all_missing_field(gain, tmp_path):
    """Verify invalid source gain is not accepted as an all missing field."""
    ds, variable = source()
    path = tmp_path / "invalid-source.nc"
    ds.to_netcdf(path)
    with netCDF4.Dataset(path, "a") as raw:
        raw[variable].scale_factor = gain
    with pytest.raises(ValueError, match="finite scalars"):
        io.import_netcdf_cf(path)


def test_source_scale_and_offset_must_have_one_numeric_type(tmp_path):
    """Reject ambiguous CF unpacking precision in an externally packed file."""
    ds, variable = source()
    path = io.export_netcdf_cf(
        ds, tmp_path / "mixed-packing-types.nc", encoding=packing(variable)
    )
    with netCDF4.Dataset(path, "a") as raw:
        var = raw[variable]
        var.delncattr("scale_factor")
        var.setncattr("scale_factor", np.float32(0.1))
    with pytest.raises(ValueError, match="must have the same numeric type"):
        io.import_netcdf_cf(path)


def test_source_validity_attributes_must_use_packed_dtype(tmp_path):
    """Reject validity limits that cannot be interpreted in the packed domain."""
    ds, variable = source(values=[[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]])
    ds[variable].attrs["valid_range"] = np.array([0.0, 2.0])
    path = io.export_netcdf_cf(
        ds, tmp_path / "invalid-validity-type.nc", encoding=packing(variable)
    )
    with netCDF4.Dataset(path, "a") as raw:
        var = raw[variable]
        var.delncattr("valid_range")
        var.setncattr("valid_range", np.array([0.0, 20.0], dtype="float64"))
    with pytest.raises(ValueError, match="valid_range must have the same numeric type"):
        io.import_netcdf_cf(path)


@pytest.mark.parametrize("compressed", [False, True])
def test_compression_chunks_shuffle_and_checksum_do_not_change_codes(
    compressed, tmp_path
):
    """Verify compression chunks shuffle and checksum do not change codes."""
    ds, variable = source(values=[[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]])
    path = io.export_netcdf_cf(
        ds,
        tmp_path / "filters.nc",
        encoding=packing(
            variable,
            zlib=compressed,
            complevel=1,
            shuffle=False,
            fletcher32=True,
            chunksizes=(1, 2, 3),
        ),
    )
    with netCDF4.Dataset(path) as raw:
        var = raw[variable]
        filters = var.filters()
        assert filters["zlib"] == compressed
        assert filters["fletcher32"] and not filters["shuffle"]
        assert var.chunking() == [1, 2, 3]
    codes = np.broadcast_to(
        np.array([[0, 0, 1], [2, 13, -32768]], dtype="int16"), ds[variable].shape
    )
    expected = np.where(codes == FILL, np.nan, codes.astype(float) * 0.1)
    assert_decoders(path, variable, expected, codes, 0.1)


@pytest.mark.parametrize("option", ["zlib", "shuffle", "fletcher32"])
def test_filter_options_are_explicit_booleans(option, tmp_path):
    """Verify filter options are explicit booleans."""
    ds, variable = source()
    with pytest.raises(ValueError, match="must be a boolean"):
        io.export_netcdf_cf(
            ds, tmp_path / "invalid.nc", encoding=packing(variable, **{option: "false"})
        )
    assert not list(tmp_path.iterdir())


def test_rate_units_normalize_before_applying_the_requested_storage_step(tmp_path):
    """Verify rate units normalize before applying the requested storage step."""
    ds, variable = source("unit_alias_rate")
    path = io.export_netcdf_cf(ds, tmp_path / "units.nc", encoding=packing(variable))
    with netCDF4.Dataset(path) as raw:
        assert raw[variable].units == "mm h-1"
        raw[variable].set_auto_maskandscale(False)
        np.testing.assert_array_equal(raw[variable][0], [[0, 0, 1], [13, 1000, -32768]])


def test_multiple_variables_choose_their_own_precision_and_keep_time_axes(tmp_path):
    """Verify multiple variables choose their own precision and keep time axes."""
    ds = reference_cases()["multiple_variables"]
    ds.attrs["precip_var"] = "precip_accum"
    path = io.export_netcdf_cf(
        ds,
        tmp_path / "multi.nc",
        encoding={
            "precip_intensity": {"dtype": "int16", "scale_factor": 0.1},
            "precip_accum": {"dtype": "int16", "scale_factor": 0.01},
        },
    )
    actual = io.import_netcdf_cf(path)
    assert actual.precip_intensity.dims == ("time_aux", "y", "x")
    assert actual.precip_accum.dims == ("time", "y", "x")
    with netCDF4.Dataset(path) as raw:
        assert raw["precip_intensity"].scale_factor == 0.1
        assert raw["precip_accum"].scale_factor == 0.01
        assert raw["precip_accum"].units == "mm"


@pytest.mark.parametrize("mode", [None, "timestep", "member"])
def test_legacy_writes_honour_compression_and_chunk_controls(mode, tmp_path):
    """Verify legacy writes honour compression and chunk controls."""
    metadata = dict(
        projection="EPSG:2193",
        x1=1600000.0,
        x2=1603000.0,
        y1=5900000.0,
        y2=5902000.0,
        cartesian_unit="m",
        yorigin="upper",
        unit="mm/h",
    )
    exporter = io.initialize_forecast_exporter_netcdf(
        str(tmp_path),
        "filters",
        datetime(2023, 1, 1),
        5,
        3,
        (2, 3),
        metadata,
        n_ens_members=2,
        datatype="int16",
        fill_value=-32768,
        scale_factor=0.1,
        incremental=mode,
        zlib=True,
        complevel=2,
        shuffle=False,
        fletcher32=True,
        chunksizes=(1, 1, 2, 3),
    )
    values = np.ones((2, 3, 2, 3))
    if mode is None:
        io.export_forecast_dataset(values, exporter)
    elif mode == "timestep":
        for index in range(3):
            io.export_forecast_dataset(values[:, index], exporter)
    else:
        for index in range(2):
            io.export_forecast_dataset(values[index], exporter)
    io.close_forecast_files(exporter)
    with netCDF4.Dataset(tmp_path / "filters.nc") as raw:
        var = raw["precip_intensity"]
        assert var.chunking() == [1, 1, 2, 3]
        assert var.filters()["complevel"] == 2
        assert var.filters()["fletcher32"] and not var.filters()["shuffle"]
        np.testing.assert_array_equal(var[:], values)


@pytest.mark.parametrize("mode", [None, "timestep", "member"])
@pytest.mark.parametrize("hidden_value", [0, -32768])
def test_masked_integer_callback_input_never_loses_its_mask(
    mode, hidden_value, tmp_path
):
    """Verify masked integer callback input never loses its mask."""
    values = np.ones((2, 3, 2, 3), dtype="int16")
    values[..., 1, 2] = hidden_value
    mask = np.zeros(values.shape, dtype=bool)
    mask[..., 1, 2] = True
    values = np.ma.array(values, mask=mask)
    metadata = dict(
        projection="EPSG:2193",
        x1=1600000.0,
        x2=1603000.0,
        y1=5900000.0,
        y2=5902000.0,
        cartesian_unit="m",
        yorigin="upper",
        unit="mm/h",
    )
    exporter = io.initialize_forecast_exporter_netcdf(
        str(tmp_path),
        "masked",
        datetime(2023, 1, 1),
        5,
        3,
        (2, 3),
        metadata,
        n_ens_members=2,
        datatype="int16",
        fill_value=-32768,
        scale_factor=0.1,
        incremental=mode,
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
    with netCDF4.Dataset(tmp_path / "masked.nc") as raw:
        var = raw["precip_intensity"]
        np.testing.assert_array_equal(np.ma.getmaskarray(var[:]), mask)
        var.set_auto_maskandscale(False)
        np.testing.assert_array_equal(
            var[:], np.where(mask, -32768, 10).astype("int16")
        )


@pytest.mark.parametrize("target", ["x", "time", "detection_status"])
def test_rain_packing_options_do_not_rescale_coordinates_or_flags(target, tmp_path):
    """Verify rain packing options do not rescale coordinates or flags."""
    ds, _ = source("reflectivity")
    with pytest.raises(ValueError, match="only for physical fields"):
        io.export_netcdf_cf(
            ds, tmp_path / "invalid.nc", encoding={target: {"scale_factor": 0.1}}
        )
    assert not list(tmp_path.iterdir())


def test_custom_zero_fill_with_negative_codes_declares_explicit_valid_range(tmp_path):
    """Verify custom zero fill with negative codes declares explicit valid range."""
    ds, variable = source(values=[[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]])
    path = io.export_netcdf_cf(
        ds,
        tmp_path / "negative-codes.nc",
        encoding=packing(variable, add_offset=2.0, _FillValue=0),
    )
    with netCDF4.Dataset(path) as raw:
        var = raw[variable]
        np.testing.assert_array_equal(var.valid_range, [-20, -7])
        assert np.asarray(var.valid_range).dtype == np.dtype("int16")
        assert var._FillValue > var.valid_range[1]
        var.set_auto_maskandscale(False)
        codes = var[:]
        # Emulate a generic packed-domain validity check before decoding.
        missing = (
            (codes == var._FillValue)
            | (codes < var.valid_range[0])
            | (codes > var.valid_range[1])
        )
        np.testing.assert_array_equal(missing, np.isnan(ds[variable]))


def test_positive_fill_below_valid_codes_declares_explicit_valid_range(tmp_path):
    """Verify positive fill below valid codes declares explicit valid range."""
    ds, variable = source(values=[[2.0, 3.0, 4.0], [5.0, 6.0, np.nan]])
    path = io.export_netcdf_cf(
        ds, tmp_path / "positive-fill.nc", encoding=packing(variable, _FillValue=10)
    )
    with netCDF4.Dataset(path) as raw:
        np.testing.assert_array_equal(raw[variable].valid_range, [20, 60])
        assert raw[variable]._FillValue < raw[variable].valid_range[0]
    np.testing.assert_allclose(
        io.import_netcdf_cf(path)[variable], ds[variable], equal_nan=True
    )


@pytest.mark.parametrize("declared", [False, True])
def test_custom_fill_inside_valid_code_extent_is_rejected(declared, tmp_path):
    """Verify custom fill inside valid code extent is rejected."""
    ds, variable = source(values=[[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]])
    if declared:
        ds[variable].attrs["valid_range"] = np.array([0.0, 2.0])
    with pytest.raises(ValueError, match="outside the stored valid range"):
        io.export_netcdf_cf(
            ds,
            tmp_path / "invalid.nc",
            encoding=packing(variable, add_offset=2.0, _FillValue=-10),
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "value, scale, offset, message",
    [
        (1.7e308, 1e308, 0.0, "Decoded physical values overflow"),
        (0.0, 0.1, -0.05, "negative physical rain"),
    ],
)
def test_encoding_cannot_create_infinite_or_negative_decoded_rain(
    value, scale, offset, message, tmp_path
):
    """Verify encoding cannot create infinite or negative decoded rain."""
    ds, variable = source(values=np.full((2, 3), value))
    with pytest.raises(ValueError, match=message):
        io.export_netcdf_cf(
            ds,
            tmp_path / "invalid.nc",
            encoding=packing(variable, scale, add_offset=offset),
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("all_missing", [False, True])
def test_derived_actual_range_is_not_carried_stale_into_packed_output(
    all_missing, tmp_path
):
    """Verify derived actual range is not carried stale into packed output."""
    ds, variable = source(
        values=(
            np.full((2, 3), np.nan)
            if all_missing
            else [[0.0, 0.04, 0.06], [0.25, 1.26, np.nan]]
        )
    )
    ds[variable].attrs["actual_range"] = np.array([0.0, 1.26])
    path = io.export_netcdf_cf(ds, tmp_path / "range.nc", encoding=packing(variable))
    with netCDF4.Dataset(path) as raw:
        assert "actual_range" not in raw[variable].ncattrs()
    np.testing.assert_array_equal(ds[variable].attrs["actual_range"], [0.0, 1.26])
