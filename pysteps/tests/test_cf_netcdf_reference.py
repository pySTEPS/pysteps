"""Check stage-1 reference semantics, without claiming production compliance."""

import netCDF4
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from pysteps.tests.cf_netcdf_reference import (
    CASE_IDS,
    reference_cases,
    write_reference_files,
)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_reference_file_preserves_physical_values_and_masks(case_id, tmp_path):
    """Verify reference file preserves physical values and masks."""
    cases = reference_cases()
    # Each case is serialized independently from the future production exporter.
    path = tmp_path / f"{case_id}.nc"
    cases[case_id].to_netcdf(path, engine="netcdf4")
    with xr.open_dataset(path, decode_coords="all") as restored:
        xr.testing.assert_allclose(cases[case_id], restored)
        original_crs = CRS.from_cf(cases[case_id].crs.attrs)
        assert original_crs.equals(CRS.from_cf(restored.crs.attrs))
    with netCDF4.Dataset(path) as raw:
        assert raw.Conventions == "CF-1.8"
        assert raw["crs"].dimensions == ()
        assert raw["x"].dtype == np.dtype("float64")
        assert "_FillValue" not in raw["x"].ncattrs()


def test_duration_is_distinct_from_cadence_and_supports_fractional_minutes():
    """Verify duration is distinct from cadence and supports fractional minutes."""
    cases = reference_cases()
    mean = cases["mean_rate"]
    durations = np.diff(mean.time_bnds.values, axis=1) / np.timedelta64(1, "s")
    np.testing.assert_array_equal(durations.ravel(), [600, 600, 600])
    np.testing.assert_array_equal(
        np.diff(mean.time) / np.timedelta64(1, "s"), [450, 450]
    )
    depth = cases["interval_depth"]
    durations = np.diff(depth.time_bnds.values, axis=1) / np.timedelta64(1, "s")
    rates = depth.precip_accum.values / durations[:, :, None] * 3600
    np.testing.assert_allclose(rates[:, 0, 1], [12, 12, 12])
    assert "accutime" not in depth.precip_accum.attrs


def test_cumulative_amounts_require_differencing_not_summing():
    """Verify cumulative amounts require differencing not summing."""
    ds = reference_cases()["cumulative_depth"]
    np.testing.assert_array_equal(ds.precip_cumulative[:, 0, 1], [1.5, 3.0, 6.0])
    np.testing.assert_array_equal(np.diff(ds.precip_cumulative[:, 0, 1]), [1.5, 3.0])
    assert np.all(ds.time_bnds[:, 0].values == ds.forecast_reference_time.values)


def test_dbz_negative_zero_no_detection_and_missing_are_distinct():
    """Verify dbz negative zero no detection and missing are distinct."""
    ds = reference_cases()["reflectivity"]
    assert "transform" not in ds.reflectivity.attrs
    assert ds.reflectivity[0, 0, 0] == -10
    assert ds.reflectivity[0, 0, 1] == 0
    assert ds.detection_status[0, 1, 1] == 0
    assert np.isnan(ds.reflectivity[0, 1, 2])
    assert (10 ** (float(ds.reflectivity[0, 0, 2]) / 10) / 200) ** (
        1 / 1.6
    ) == pytest.approx(0.6484197773)


def test_multiple_variables_keep_their_own_time_support_and_member_ids():
    """Verify multiple variables keep their own time support and member ids."""
    cases = reference_cases()
    multi = cases["multiple_variables"]
    assert multi.precip_intensity.dims == ("time", "y", "x")
    assert multi.precip_accum.dims == ("time_depth", "y", "x")
    assert multi.time_depth.attrs["bounds"] == "time_depth_bnds"
    assert "bounds" not in multi.time.attrs
    np.testing.assert_array_equal(cases["ensemble_rate"].ens_number, [0, 7])


def test_units_scale_numerically_and_missing_is_not_dry():
    """Verify units scale numerically and missing is not dry."""
    cases = reference_cases()
    np.testing.assert_allclose(
        cases["unit_alias_rate"].precip_intensity * 3600000,
        cases["instantaneous_rate"].precip_intensity,
        equal_nan=True,
    )
    assert np.isnan(cases["all_missing"].precip_intensity).all()
    assert cases["instantaneous_rate"].precip_intensity[0, 0, 0] == 0


def test_reference_inventory(tmp_path):
    """Verify reference inventory."""
    inventory = write_reference_files(tmp_path)
    assert tuple(inventory) == CASE_IDS
    assert all(
        len(item["sha256"]) == 64 and item["bytes"] > 0 for item in inventory.values()
    )
