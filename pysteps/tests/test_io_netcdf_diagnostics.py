"""Exercise user-facing CF diagnostics and failure-safe output handling."""

import netCDF4
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError

from pysteps import io
from pysteps.io.netcdf_cf import _stored_values
from pysteps.tests.cf_netcdf_reference import reference_cases


@pytest.mark.parametrize(
    "options, detail",
    [
        ({"scale_factor": "small"}, "scale_factor"),
        ({"scale_factor": [0.1, 0.2]}, "scale_factor"),
        ({"scale_factor": 0.1 + 0.2j}, "scale_factor"),
        ({"add_offset": "zero"}, "add_offset"),
        ({"_FillValue": [-1, -2]}, "fill value"),
        ({"_FillValue": "missing"}, "fill value"),
        ({"_FillValue": None}, "reserved _FillValue"),
        ({"chunksizes": 3}, "chunksizes"),
        ({"dtype": "unknown"}, "dtype"),
        ({"dtype": "int32"}, "int16"),
        ({"unrecognized": True}, "unrecognized"),
    ],
)
def test_malformed_encoding_identifies_field_and_preserves_output(
    options, detail, tmp_path
):
    """Malformed user options raise useful errors without replacing an output."""
    ds = reference_cases()["instantaneous_rate"]
    path = tmp_path / "forecast.nc"
    path.write_bytes(b"previous complete forecast")
    with pytest.raises(ValueError) as caught:
        io.export_netcdf_cf(
            ds,
            path,
            overwrite=True,
            encoding={
                "precip_intensity": {"dtype": "int16", "scale_factor": 0.1, **options}
            },
        )
    message = str(caught.value)
    assert "precip_intensity" in message and detail in message
    assert path.read_bytes() == b"previous complete forecast"
    assert not list(tmp_path.glob(".*incomplete*"))


def test_source_packing_type_conflict_identifies_both_types(tmp_path):
    """A malformed external file exposes the actual conflicting packing types."""
    ds = reference_cases()["instantaneous_rate"]
    path = io.export_netcdf_cf(
        ds,
        tmp_path / "mixed-types.nc",
        encoding={"precip_intensity": {"dtype": "int16", "scale_factor": 0.1}},
    )
    with netCDF4.Dataset(path, "a") as raw:
        raw["precip_intensity"].setncattr("add_offset", np.float32(0))
    with pytest.raises(ValueError) as caught:
        io.import_netcdf_cf(path)
    for detail in ("precip_intensity", "float32", "float64", "Store both"):
        assert detail in str(caught.value)


@pytest.mark.parametrize("missing", [True, False])
def test_validity_mask_errors_distinguish_missing_from_misaligned(missing, tmp_path):
    """The remedy for an absent mask differs from a transposed mask."""
    ds = reference_cases()["instantaneous_rate"]
    ds.precip_intensity.attrs["pysteps_validity_mask"] = "observed"
    if not missing:
        ds["observed"] = xr.ones_like(ds.precip_intensity).transpose("y", "x", "time")
    with pytest.raises(ValueError) as caught:
        io.export_netcdf_cf(ds, tmp_path / "invalid.nc")
    message = str(caught.value)
    assert "observed" in message and "precip_intensity" in message
    assert ("Missing" if missing else "Misaligned") in message
    if not missing:
        assert "mask dims=" in message and "field dims=" in message
    assert not list(tmp_path.iterdir())


def test_finite_projected_grid_can_have_undefined_inverse(tmp_path):
    """Finite x/y outside an orthographic globe need a transformation diagnostic."""
    ds = reference_cases()["instantaneous_rate"].drop_vars(["lat", "lon"])
    crs = CRS.from_proj4("+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84 +units=m")
    ds["crs"] = xr.DataArray(np.int32(0), attrs=crs.to_cf())
    for axis in ("x", "y"):
        ds = ds.assign_coords({axis: (axis, ds[axis].values + 1e8, ds[axis].attrs)})
        assert np.isfinite(ds[axis]).all()
    lon, lat = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True).transform(
        *np.meshgrid(ds.x, ds.y)
    )
    assert not (np.isfinite(lon).all() and np.isfinite(lat).all())
    path = tmp_path / "forecast.nc"
    path.write_bytes(b"previous complete forecast")
    with pytest.raises(ValueError) as caught:
        io.export_netcdf_cf(ds, path, include_latlon=True, overwrite=True)
    assert isinstance(caught.value.__cause__, ProjError)
    for detail in ("longitude/latitude", "CRS", "coordinate units", "grid extent"):
        assert detail in str(caught.value)
    assert path.read_bytes() == b"previous complete forecast"
    assert not list(tmp_path.glob(".*incomplete*"))


@pytest.mark.parametrize("values", [np.array(["rain"]), np.array([np.inf])])
def test_physical_array_errors_identify_dtype_or_infinity(values):
    """Bad array types and infinite numeric values provide different diagnoses."""
    with pytest.raises(ValueError) as caught:
        _stored_values(values, {}, name="rain")
    message = str(caught.value)
    assert "rain" in message
    assert ("dtype=" if values.dtype.kind == "U" else "infinite values") in message


def test_omitted_fill_option_records_reserved_code_and_preserves_dry(tmp_path):
    """The explicit-fill policy preserves observed zero and missing separately."""
    ds = reference_cases()["instantaneous_rate"]
    path = io.export_netcdf_cf(
        ds,
        tmp_path / "default-fill.nc",
        encoding={"precip_intensity": {"dtype": "int16", "scale_factor": 0.1}},
    )
    with netCDF4.Dataset(path) as raw:
        var = raw["precip_intensity"]
        assert "_FillValue" in var.ncattrs()
        assert var._FillValue == np.int16(-32768)
        var.set_auto_maskandscale(False)
        codes = var[:]
        np.testing.assert_array_equal(codes == -32768, ds.precip_intensity.isnull())
        assert np.all(codes[ds.precip_intensity.values == 0] == 0)
