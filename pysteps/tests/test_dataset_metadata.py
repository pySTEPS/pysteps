"""Scientific regression checks for stage 2 shared physical metadata."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from pysteps.dataset_metadata import (
    cadence_seconds,
    cf_grid_mapping,
    cumulative_to_intervals,
    normalize_dataset,
    time_bounds,
    unit_spec,
)
from pysteps.tests.cf_netcdf_reference import CASE_IDS, reference_cases
from pysteps.utils.conversion import to_raindepth, to_rainrate, to_reflectivity
from pysteps.utils.dimension import aggregate_fields_time
from pysteps.xarray_helpers import (
    convert_input_to_xarray_dataset,
    convert_output_to_xarray_dataset,
)


def case(name, variable=None):
    ds = reference_cases()[name]
    if variable is None:
        variable = (
            "precip_accum"
            if name == "interval_depth"
            else (
                "precip_cumulative"
                if name == "cumulative_depth"
                else (
                    "precip_mean_rate"
                    if name == "mean_rate"
                    else (
                        "reflectivity" if name == "reflectivity" else "precip_intensity"
                    )
                )
            )
        )
    return normalize_dataset(ds, variable)


@pytest.mark.parametrize("name", CASE_IDS)
def test_reference_normalization_keeps_values_geometry_and_missing(name):
    source = reference_cases()[name]
    snapshot = source.copy(deep=True)
    result = case(name)
    variable = result.attrs["precip_var"]
    expected = source[variable] * (3600000 if name == "unit_alias_rate" else 1)
    np.testing.assert_allclose(result[variable], expected, equal_nan=True)
    xr.testing.assert_identical(source, snapshot)
    assert CRS.from_cf(result.crs.attrs).equals(CRS.from_cf(source.crs.attrs))
    assert result[variable].attrs["grid_mapping"] == "crs"
    assert result.x.dtype == np.float64
    assert (
        result.y.attrs["stepsize"] < 0
        if name != "ensemble_rate"
        else result.y.attrs["stepsize"] > 0
    )


def test_selecting_second_time_axis_preserves_unselected_fields_and_links():
    original = reference_cases()["multiple_variables"]
    result = normalize_dataset(original, "precip_accum")
    assert result.precip_accum.dims == ("time", "y", "x")
    assert result.precip_intensity.dims == ("time_aux", "y", "x")
    np.testing.assert_array_equal(result.precip_intensity, original.precip_intensity)
    assert result.time.attrs["bounds"] in result
    assert result.precip_accum.attrs["cell_methods"] == "time: sum"
    assert result.precip_intensity.attrs["cell_methods"] == "time_aux: point"
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_dataset(original)


def test_io_validation_does_not_invent_model_parameters():
    result = case("all_missing")
    attrs = result.precip_intensity.attrs
    assert "threshold" not in attrs and "zerovalue" not in attrs
    with pytest.raises(ValueError, match="explicit scalar threshold"):
        normalize_dataset(result, require_model=True)


@pytest.mark.parametrize(
    "name,mutate,match",
    [
        (
            "instantaneous_rate",
            lambda ds: ds.crs.attrs.update(false_easting=1),
            "conflicts",
        ),
        ("instantaneous_rate", lambda ds: ds.x.attrs.update(stepsize=12), "spacing"),
        (
            "instantaneous_rate",
            lambda ds: ds.precip_intensity.attrs.update(ancillary_variables="absent"),
            "Dangling",
        ),
        (
            "mean_rate",
            lambda ds: ds.precip_mean_rate.attrs.update(accutime=7.5),
            "accutime",
        ),
        (
            "instantaneous_rate",
            lambda ds: ds.time.attrs.update(stepsize=600),
            "stepsize",
        ),
        (
            "instantaneous_rate",
            lambda ds: ds.precip_intensity.attrs.update(units="kg m-2 s-1"),
            "Unsupported",
        ),
    ],
)
def test_contradictory_metadata_fails_without_guessing(name, mutate, match):
    ds = reference_cases()[name]
    mutate(ds)
    with pytest.raises(ValueError, match=match):
        normalize_dataset(ds)


@pytest.mark.parametrize(
    "value,allowed", [(-0.001, False), (np.inf, False), (np.nan, True), (0.04, True)]
)
def test_rain_physical_values(value, allowed):
    ds = reference_cases()["instantaneous_rate"]
    ds.precip_intensity.data[0, 0, 0] = value
    if allowed:
        normalize_dataset(ds)
    else:
        with pytest.raises(ValueError, match="physical"):
            normalize_dataset(ds)


def test_midpoint_labels_keep_source_times_and_use_interval_end():
    ds = reference_cases()["mean_rate"]
    original = ds.time.values.copy()
    ds = ds.assign_coords(
        time=("time", original - np.timedelta64(300, "s"), ds.time.attrs)
    )
    result = normalize_dataset(ds)
    np.testing.assert_array_equal(result.time, original)
    np.testing.assert_array_equal(
        result.time_source, original - np.timedelta64(300, "s")
    )


def test_crs_coordinate_kilometres_are_scaled_with_mapping_offsets():
    ds = reference_cases()["instantaneous_rate"]
    ds = ds.assign_coords(
        x=("x", ds.x.values / 1000, {**ds.x.attrs, "units": "km"}),
        y=("y", ds.y.values / 1000, {**ds.y.attrs, "units": "km"}),
    )
    result = normalize_dataset(ds)
    np.testing.assert_allclose(result.x, [1600000, 1601000, 1602000])
    assert result.crs.attrs["false_easting"] == 1600000
    assert result.x.attrs["units"] == "m"
    # A genuinely kilometre-based projected CRS must also normalize its WKT.
    projection = CRS(
        "+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +units=km"
    )
    ds.crs.attrs = cf_grid_mapping(projection)
    result = normalize_dataset(ds)
    assert result.crs.attrs["false_easting"] == pytest.approx(1600000)
    assert CRS.from_cf(result.crs.attrs).axis_info[0].unit_conversion_factor == 1


def test_projection_parameters_preserve_sphere_and_polar_origin():
    attrs = cf_grid_mapping(
        "+proj=stere +lat_0=-90 +lon_0=20 +lat_ts=-60 +R=6371000 +units=m"
    )
    assert attrs["latitude_of_projection_origin"] == -90
    assert CRS.from_cf(attrs).ellipsoid.semi_major_metre == 6371000


def test_legacy_input_helper_handles_epsg_single_pixels_and_timezone():
    metadata = dict(
        unit="mm/h",
        x1=1600000.0,
        x2=1601000.0,
        y1=5900000.0,
        y2=5901000.0,
        projection="EPSG:2193",
        cartesian_unit="m",
        yorigin="upper",
        institution="synthetic",
        transform=None,
    )
    aware = datetime(2023, 1, 1, 11, tzinfo=timezone(timedelta(hours=11)))
    ds = convert_input_to_xarray_dataset(np.ones((1, 1, 1)), None, metadata, aware, 450)
    assert ds.time[0] == np.datetime64("2023-01-01T00:00:00")
    assert "transform" not in ds.precip_intensity.attrs
    assert ds.precip_intensity.attrs["standard_name"] == "lwe_precipitation_rate"
    assert ds.transverse_mercator.attrs["crs_wkt"]


def test_depth_mean_conversion_uses_each_interval_and_preserves_missing():
    original = case("interval_depth")
    rates = to_rainrate(original)
    np.testing.assert_allclose(rates.precip_intensity[:, 0, 1], [12, 12, 12])
    assert "accutime" not in rates.precip_intensity.attrs
    assert rates.precip_intensity.attrs["cell_methods"] == "time: mean"
    depths = to_raindepth(rates)
    np.testing.assert_allclose(
        depths.precip_accum, original.precip_accum, equal_nan=True
    )
    np.testing.assert_array_equal(depths.time_bnds, original.time_bnds)


def test_variable_duration_threshold_is_a_linked_field_not_a_wrong_scalar():
    ds = case("interval_depth")
    ds.precip_accum.attrs.update(threshold=0.1, zerovalue=0.0)
    result = to_rainrate(ds)
    assert "threshold" not in result.precip_intensity.attrs
    link = result.precip_intensity.attrs["ancillary_variables"]
    np.testing.assert_allclose(result[link], [0.8, 0.8, 0.4])
    assert result[link].attrs["units"] == "mm/h"
    assert result.precip_intensity.attrs["zerovalue"] == 0


def test_point_integration_is_explicit_and_does_not_use_cadence():
    ds = case("instantaneous_rate")
    with pytest.raises(ValueError, match="integration"):
        to_raindepth(ds)
    with pytest.raises(ValueError, match="duration_seconds"):
        to_raindepth(ds, integration="constant")
    result = to_raindepth(ds, integration="constant", duration_seconds=600)
    assert result.precip_accum.attrs["accutime"] == 10
    assert result.time.attrs["stepsize"] == 450
    np.testing.assert_allclose(
        result.precip_accum, ds.precip_intensity / 6, equal_nan=True
    )


def test_legacy_duration_is_opt_in_and_bounds_win():
    ds = case("interval_depth").drop_vars("time_bnds")
    ds.time.attrs.pop("bounds")
    ds.precip_accum.attrs["accutime"] = 7.5
    with pytest.raises(ValueError, match="bounds"):
        to_rainrate(ds)
    result = to_rainrate(ds, legacy=True)
    assert result.precip_intensity[0, 0, 1] == 12
    np.testing.assert_array_equal(result.time_bnds[:, 1], result.time)
    assert result.time[0] - result.time_bnds[0, 0] == np.timedelta64(450, "s")
    assert "Legacy accutime" in result.attrs["history"]
    normalize_dataset(result)


@pytest.mark.parametrize("marker", [None, "dB"])
def test_dbz_representation_is_not_an_optional_statistical_transform(marker):
    ds = case("reflectivity")
    ds.reflectivity.attrs["transform"] = marker
    result = to_rainrate(ds)
    assert float(result.precip_intensity[0, 0, 2]) == pytest.approx(0.6484197773255048)
    assert result.precip_intensity[0, 1, 1] == 0
    assert np.isnan(result.precip_intensity[0, 1, 2])
    assert result.precip_intensity.attrs["zr_source"] == "a:default,b:default"
    assert result.precip_intensity[0, 0, 0] > 0  # -10 dBZ is a valid detection


def test_conversion_does_not_overwrite_other_products_or_reuse_packing():
    ds = case("multiple_variables", "precip_accum")
    with pytest.raises(ValueError, match="overwrite"):
        to_rainrate(ds)
    ds.precip_accum.encoding.update(scale_factor=0.1, add_offset=0, dtype="int16")
    result = to_rainrate(ds, name="depth_mean_rate")
    assert "scale_factor" not in result.depth_mean_rate.encoding
    np.testing.assert_array_equal(result.precip_intensity, ds.precip_intensity)


def test_reflectivity_identity_and_zr_trace_values():
    ds = case("reflectivity")
    result = to_reflectivity(ds)
    np.testing.assert_array_equal(result.reflectivity, ds.reflectivity)
    rates = case("instantaneous_rate")
    rates.precip_intensity.attrs["threshold"] = 0.1
    result = to_reflectivity(rates)
    expected_trace = 10 * np.log10(200 * 0.04**1.6)
    assert result.reflectivity[0, 0, 1] == pytest.approx(expected_trace)
    roundtrip = to_rainrate(result)
    np.testing.assert_allclose(
        roundtrip.precip_intensity, rates.precip_intensity, equal_nan=True
    )


def test_aggregation_updates_duration_and_uses_weighted_means():
    depths = case("interval_depth")
    aggregated = aggregate_fields_time(depths, 15)
    np.testing.assert_allclose(aggregated.precip_accum[:, 0, 1], [3, 3])
    assert aggregated.precip_accum.attrs["accutime"] == 15
    np.testing.assert_allclose(
        to_rainrate(aggregated).precip_intensity[:, 0, 1], [12, 12]
    )
    means = to_rainrate(depths)
    means.precip_intensity.data[1] *= 2
    result = aggregate_fields_time(means, 15)
    np.testing.assert_allclose(result.precip_intensity[:, 0, 1], [18, 12])
    assert result.precip_intensity.attrs["cell_methods"] == "time: mean"


def test_aggregation_differences_cumulative_totals_and_rejects_resets():
    cumulative = case("cumulative_depth")
    intervals = cumulative_to_intervals(cumulative)
    np.testing.assert_allclose(intervals.precip_cumulative[:, 0, 1], [1.5, 1.5, 3])
    result = aggregate_fields_time(cumulative, 15)
    np.testing.assert_allclose(result.precip_cumulative[:, 0, 1], [3, 3])
    cumulative.precip_cumulative.data[1] = 0
    with pytest.raises(ValueError, match="decrease"):
        aggregate_fields_time(cumulative, 15)


def test_aggregation_missing_coverage_stays_missing_or_is_explicit():
    ds = case("interval_depth")
    ds.precip_accum.data[0, 0, 1] = np.nan
    default = aggregate_fields_time(ds, 15)
    assert np.isnan(default.precip_accum[0, 0, 1])
    partial = aggregate_fields_time(ds, 15, ignore_nan=True)
    assert partial.precip_accum[0, 0, 1] == 1.5
    assert partial.precip_accum_coverage_fraction[0, 0, 1] == 0.5
    assert np.isnan(partial.precip_accum[0, 1, 2])
    assert partial.precip_accum_coverage_fraction[0, 1, 2] == 0
    assert partial.time.attrs["standard_name"] == "time"
    assert partial.time.attrs["bounds"] == "time_bnds"
    assert partial.time.encoding["dtype"] == "float64"
    assert partial.time_bnds.encoding["dtype"] == "float64"


def test_overlapping_mean_rates_and_implicit_point_integration_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        aggregate_fields_time(case("mean_rate"), 15)
    with pytest.raises(ValueError, match="integration"):
        aggregate_fields_time(case("instantaneous_rate"), 15)
    result = aggregate_fields_time(
        case("instantaneous_rate"), 22.5, integration="constant"
    )
    assert result.time_bnds.shape == (1, 2)


def test_aggregation_retains_other_time_series_without_aggregating_them():
    ds = case("multiple_variables", "precip_accum")
    result = aggregate_fields_time(ds, 30)
    assert result.precip_accum.shape[0] == 1
    np.testing.assert_array_equal(result.precip_intensity, ds.precip_intensity)


def test_forecast_sparse_leads_keep_duration_and_have_valid_cf_links():
    ds = case("mean_rate")
    result = convert_output_to_xarray_dataset(ds, [1, 2, 4], np.ones((3, 2, 3)))
    duration = np.diff(result.time_bnds.values, axis=1) / np.timedelta64(1, "s")
    np.testing.assert_array_equal(duration.ravel(), [600, 600, 600])
    assert "stepsize" not in result.time.attrs
    assert (
        result.forecast_reference_time.attrs["standard_name"]
        == "forecast_reference_time"
    )
    assert result.forecast_reference_time == ds.time[-1]
    for name in result.variables:
        link = result[name].attrs.get("bounds")
        assert link is None or link in result


def test_forecast_variable_durations_require_a_product_definition():
    ds = case("interval_depth")
    with pytest.raises(ValueError, match="timestep"):
        convert_output_to_xarray_dataset(ds, 2, np.ones((2, 2, 3)))
    with pytest.raises(ValueError, match="interval_seconds"):
        convert_output_to_xarray_dataset(
            ds, 2, np.ones((2, 2, 3)), timestep_seconds=450
        )
    result = convert_output_to_xarray_dataset(
        ds, [1, 3], np.ones((2, 2, 3)), timestep_seconds=450, accumulation="cumulative"
    )
    assert np.all(result.time_bnds[:, 0].values == ds.time[-1].values)


def test_forecast_preserves_member_ids_and_drops_unforecast_detection_flags():
    ds = case("ensemble_rate")
    result = convert_output_to_xarray_dataset(ds, 1, np.ones((2, 1, 2, 3)))
    np.testing.assert_array_equal(result.ens_number, [0, 7])
    dbz = case("reflectivity")
    result = convert_output_to_xarray_dataset(dbz, 1, np.ones((1, 2, 3)))
    assert "detection_status" not in result
    assert "ancillary_variables" not in result.reflectivity.attrs


def test_normalization_is_serializable_and_does_not_change_physical_values(tmp_path):
    ds = case("interval_depth")
    path = tmp_path / "normalized.nc"
    ds.to_netcdf(path)
    with xr.open_dataset(path, decode_coords="all") as restored:
        np.testing.assert_array_equal(restored.time_bnds, ds.time_bnds)
        np.testing.assert_allclose(
            restored.precip_accum, ds.precip_accum, equal_nan=True
        )


def test_varying_threshold_fields_convert_back_with_the_quantity():
    ds = case("interval_depth")
    ds.precip_accum.attrs.update(threshold=0.1, zerovalue=0.0)
    means = to_rainrate(ds)
    result = to_raindepth(means)
    parameter = result.precip_accum.attrs["ancillary_variables"]
    np.testing.assert_allclose(result[parameter], [0.1, 0.1, 0.1])
    assert result[parameter].attrs["units"] == "mm"
    assert result[parameter].attrs["pysteps_parameter_of"] == "precip_accum"


def test_declared_dbz_floor_is_dry_but_an_unflagged_negative_detection_is_not():
    ds = case("reflectivity").drop_vars("detection_status")
    ds.reflectivity.attrs.pop("ancillary_variables")
    ds.reflectivity.attrs["zerovalue"] = -32.0
    result = to_rainrate(ds)
    assert result.precip_intensity[0, 1, 1] == 0
    assert result.precip_intensity[0, 0, 0] > 0
    assert np.isnan(result.precip_intensity[0, 1, 2])


def test_legacy_partial_mapping_uses_declared_sphere_instead_of_default_datum():
    ds = reference_cases()["polar_sphere"]
    full = ds.crs.attrs.copy()
    ds.crs.attrs = {
        key: full[key]
        for key in [
            "grid_mapping_name",
            "latitude_of_projection_origin",
            "straight_vertical_longitude_from_pole",
            "standard_parallel",
            "false_easting",
            "false_northing",
        ]
    }
    ds.attrs["projection"] = (
        "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +R=6371000 +units=m"
    )
    result = normalize_dataset(ds, legacy=True)
    assert CRS.from_cf(result.crs.attrs).ellipsoid.semi_major_metre == 6371000
    ds.crs.attrs["false_easting"] = 1000
    with pytest.raises(ValueError, match="conflicts"):
        normalize_dataset(ds, legacy=True)


def test_legacy_names_and_explicit_duration_normalize_with_provenance():
    ds = (
        reference_cases()["interval_depth"]
        .isel(time=slice(0, 2))
        .drop_vars("time_bnds")
    )
    ds.time.attrs.pop("bounds")
    ds.precip_accum.attrs.update(
        standard_name="accumulated_precipitation", accutime=7.5
    )
    with pytest.raises(ValueError):
        normalize_dataset(ds)
    result = normalize_dataset(ds, legacy=True)
    assert (
        result.precip_accum.attrs["standard_name"]
        == "lwe_thickness_of_precipitation_amount"
    )
    assert (
        result.precip_accum.attrs["pysteps_source_standard_name"]
        == "accumulated_precipitation"
    )
    assert "Legacy accutime" in result.attrs["history"]


@pytest.mark.parametrize(
    "problem", ["object", "infinite_threshold", "multi_origin", "too_precise"]
)
def test_unsupported_metadata_and_precision_are_rejected(problem):
    ds = reference_cases()["instantaneous_rate"]
    if problem == "object":
        ds.attrs["unsupported"] = {"nested": "object"}
    elif problem == "infinite_threshold":
        ds.precip_intensity.attrs["threshold"] = np.inf
    elif problem == "multi_origin":
        ds["forecast_reference_time"] = ds.time.copy()
    else:
        ds = ds.assign_coords(
            time=(
                "time",
                ds.time.values.astype("datetime64[ns]") + np.timedelta64(1, "ns"),
                ds.time.attrs,
            )
        )
    with pytest.raises(ValueError):
        normalize_dataset(ds)


def test_coordinate_aliases_singletons_and_none_metadata_are_normalized(tmp_path):
    ds = reference_cases()["ensemble_rate"].rename(
        x="easting", y="northing", ens_number="member"
    )
    ds = ds.isel(time=slice(0, 1), easting=slice(0, 1), northing=slice(0, 1))
    ds.time.attrs["stepsize"] = 450.0
    ds.easting.attrs["stepsize"] = ds.northing.attrs["stepsize"] = 1000.0
    ds.attrs["optional"] = None
    result = normalize_dataset(ds)
    assert result.precip_intensity.dims == ("ens_number", "time", "y", "x")
    assert result.time.attrs["stepsize"] == 450
    assert "optional" not in result.attrs
    result.to_netcdf(tmp_path / "singleton.nc")


def test_zero_initial_cumulative_amount_is_not_divided_by_zero():
    ds = case("cumulative_depth")
    # A new first total at the origin is zero; valid amounts thereafter are unchanged.
    origin = ds.forecast_reference_time.values
    first = ds.isel(time=slice(0, 1)).copy(deep=True)
    first = first.assign_coords(time=("time", [origin], ds.time.attrs.copy()))
    first.precip_cumulative.data[:] = 0
    first.time_bnds.data[:] = origin
    ds = xr.concat([first, ds], dim="time", data_vars="minimal", coords="minimal")
    ds.time.attrs.pop("stepsize", None)
    with pytest.raises(ValueError, match="zero-duration"):
        to_rainrate(ds)
    result = aggregate_fields_time(ds, 15)
    np.testing.assert_allclose(result.precip_cumulative[:, 0, 1], [3, 3])


def test_singleton_forecast_can_be_used_with_its_known_cadence():
    first = convert_output_to_xarray_dataset(
        case("instantaneous_rate"), 1, np.ones((1, 2, 3))
    )
    assert first.time.attrs["stepsize"] == 450
    second = convert_output_to_xarray_dataset(first, 1, np.ones((1, 2, 3)))
    assert second.time[0] - first.time[0] == np.timedelta64(450, "s")


def test_same_axis_inactive_product_is_retained_on_its_original_times():
    ds = case("interval_depth")
    ds["second_depth"] = ds.precip_accum * 2
    ds.second_depth.attrs = ds.precip_accum.attrs.copy()
    result = aggregate_fields_time(ds, 15)
    assert result.second_depth.dims == ("time_aux", "y", "x")
    np.testing.assert_array_equal(result.second_depth, ds.second_depth)
    assert result.time_aux.attrs["bounds"] == "time_aux_bnds"
    assert result.second_depth.attrs["cell_methods"] == "time_aux: sum"


def test_shared_time_bounds_keep_consistent_encoding_for_point_and_mean(tmp_path):
    ds = reference_cases()["mean_rate"]
    ds["point_rate"] = ds.precip_mean_rate.copy(deep=True)
    ds.point_rate.attrs["cell_methods"] = "time: point"
    result = normalize_dataset(ds, "precip_mean_rate")
    assert result.time.encoding["units"] == result.time_bnds.encoding["units"]
    result.to_netcdf(tmp_path / "shared.nc")


def test_geographic_auxiliary_coordinates_cannot_contradict_the_grid():
    ds = reference_cases()["instantaneous_rate"]
    ds.lat.data += 1
    with pytest.raises(ValueError, match="Geographic coordinates conflict"):
        normalize_dataset(ds)


def test_spatial_aggregation_recomputes_geolocation_at_new_centres():
    from pysteps.utils.dimension import aggregate_fields
    from pysteps.xarray_helpers import compute_lat_lon

    ds = case("instantaneous_rate")
    result = aggregate_fields(ds, 2, dim="y")
    lat, lon = compute_lat_lon(
        result.x.values, result.y.values, result.attrs["projection"]
    )
    np.testing.assert_allclose(result.lat, lat)
    np.testing.assert_allclose(result.lon, lon)
    # The enlarged singleton row has explicit spacing, so the strict shared
    # validator can still establish its footprint.
    normalize_dataset(result)
    with pytest.raises(ValueError, match="aggregate_fields_time"):
        aggregate_fields(ds, 3, dim="time")


def test_point_products_do_not_reinterpret_scan_duration_as_accumulation():
    ds = reference_cases()["instantaneous_rate"]
    ds.precip_intensity.attrs["accutime"] = 5
    result = normalize_dataset(ds)
    assert "accutime" not in result.precip_intensity.attrs
    assert result.precip_intensity.attrs["pysteps_source_accutime_minutes"] == 5
    assert "bounds" not in result.time.attrs


def test_legacy_exporter_accepts_normalized_wkt_and_keeps_full_crs(tmp_path):
    from pysteps.io.exporters import (
        initialize_forecast_exporter_netcdf,
        close_forecast_files,
    )
    from pysteps.xarray_helpers import geodata_from_dataset
    import netCDF4

    ds = case("instantaneous_rate")
    metadata = {**geodata_from_dataset(ds), "unit": "mm/h"}
    exporter = initialize_forecast_exporter_netcdf(
        str(tmp_path), "wkt", datetime(2023, 1, 1), 7.5, 1, (2, 3), metadata
    )
    close_forecast_files(exporter)
    with netCDF4.Dataset(tmp_path / "wkt.nc") as raw:
        mapping = raw[raw["precip_intensity"].grid_mapping]
        restored = CRS.from_cf(
            {key: mapping.getncattr(key) for key in mapping.ncattrs()}
        )
        assert restored.equals(CRS.from_epsg(2193))
        assert raw["precip_intensity"].standard_name == "lwe_precipitation_rate"


@pytest.mark.parametrize("duration", [450.0, 90030.0])
def test_bom_duration_extraction_preserves_fractional_minutes_and_days(
    duration, tmp_path
):
    import netCDF4
    from pysteps.io.importers import _import_bom_rf3_geodata

    path = tmp_path / "bom.nc"
    with netCDF4.Dataset(path, "w") as raw:
        for axis in ("x", "y"):
            raw.createDimension(axis, 2)
            coordinate = raw.createVariable(axis, "f8", (axis,))
            coordinate[:] = [0.0, 1000.0]
            coordinate.units = "m"
        projection = raw.createVariable("proj", "i4")
        projection.grid_mapping_name = "albers_conical_equal_area"
        projection.longitude_of_central_meridian = 144.752
        projection.latitude_of_projection_origin = -37.852
        projection.standard_parallel = [-18.0, -36.0]
        rainfall = raw.createVariable("precipitation", "f8", ("y", "x"))
        rainfall.units = "mm"
        for name, value in [("start_time", 0.0), ("valid_time", duration)]:
            time = raw.createVariable(name, "f8")
            time.units = "seconds since 2023-01-01 00:00:00"
            time[...] = value
    with netCDF4.Dataset(path) as raw:
        assert _import_bom_rf3_geodata(raw)["accutime"] == duration / 60.0


def test_compact_projected_dataset_can_be_padded_without_latlon():
    from pysteps.utils.dimension import square_domain

    ds = case("projected_without_latlon").isel(y=slice(0, 1))
    ds.precip_intensity.attrs["zerovalue"] = 0.0
    result = square_domain(ds, method="pad")
    assert result.sizes["x"] == result.sizes["y"]
    assert "lat" not in result and "lon" not in result


def test_midpoint_relabelling_cannot_move_shared_point_observations():
    ds = reference_cases()["mean_rate"]
    ds = ds.assign_coords(
        time=("time", ds.time.values - np.timedelta64(300, "s"), ds.time.attrs)
    )
    ds["point_rate"] = ds.precip_mean_rate.copy(deep=True)
    ds.point_rate.attrs["cell_methods"] = "time: point"
    with pytest.raises(ValueError, match="separate time axes"):
        normalize_dataset(ds, "precip_mean_rate")


@pytest.mark.parametrize("value", [True, 1 + 2j, np.ones((2, 2))])
def test_nonserializable_attribute_types_are_rejected(value):
    ds = reference_cases()["instantaneous_rate"]
    ds.attrs["unsupported"] = value
    with pytest.raises(ValueError, match="nonserializable metadata"):
        normalize_dataset(ds)


@pytest.mark.parametrize("target", ["time_bnds", "forecast_reference_time"])
def test_bounds_and_forecast_origin_cannot_silently_lose_precision(target):
    ds = reference_cases()["cumulative_depth"]
    original = ds[target]
    ds[target] = (
        original.dims,
        original.values.astype("datetime64[ns]") + np.timedelta64(1, "ns"),
        original.attrs,
    )
    with pytest.raises(ValueError, match="microseconds"):
        normalize_dataset(ds)


@pytest.mark.parametrize("unit", ["m s-1", "cm"])
def test_legacy_exporter_does_not_relabel_unscaled_values(unit, tmp_path):
    from pysteps.io.exporters import initialize_forecast_exporter_netcdf

    with pytest.raises(ValueError, match="normalize units before export"):
        initialize_forecast_exporter_netcdf(
            str(tmp_path),
            "unscaled",
            datetime(2023, 1, 1),
            5,
            1,
            (2, 3),
            {"unit": unit},
        )
    assert not (tmp_path / "unscaled.nc").exists()
