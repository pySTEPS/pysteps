"""Validate precipitation mass flux against independently specified depth rates."""

import netCDF4
import numpy as np
import pytest
import xarray as xr

from pysteps import io
from pysteps.dataset_metadata import normalize_dataset, physical_variables, unit_spec
from pysteps.tests.cf_netcdf_reference import reference_cases
from pysteps.utils.conversion import to_raindepth
from pysteps.xarray_helpers import convert_input_to_xarray_dataset

MASS_FLUX = np.array([[0.0, 0.0001, 0.001], [0.002, 0.01, np.nan]])
DEPTH_RATE = np.array([[0.0, 0.36, 3.6], [7.2, 36.0, np.nan]])


def mass_flux_source(product="instantaneous_rate", units="kg m-2 s-1"):
    """Construct a mass-flux source without using pySTEPS unit conversion."""
    ds = reference_cases()[product].rename(
        {"precip_mean_rate" if product == "mean_rate" else "precip_intensity": "pr"}
    )
    ds.attrs.pop("precip_var", None)
    ds["pr"] = (
        ds.pr.dims,
        np.broadcast_to(MASS_FLUX, ds.pr.shape).copy(),
        {
            **ds.pr.attrs,
            "standard_name": "precipitation_flux",
            "long_name": "total precipitation mass flux",
            "units": units,
            "threshold": 0.0001,
            "zerovalue": 0.0,
            "pysteps_no_detection_value": 0.0,
            "valid_range": np.array([0.0, 0.02]),
        },
    )
    return ds


@pytest.mark.parametrize(
    "units",
    [
        "kg m-2 s-1",
        "kg m^-2 s^-1",
        "kg m**-2 s**-1",
        "kg/m2/s",
        "kg/m^2/s",
        "kg m-2/s",
        "  kg   m-2   s-1  ",
    ],
)
def test_mass_flux_units_and_metadata_conversion(units):
    """Common spellings give known rates and leave the caller's source intact."""
    source = mass_flux_source(units=units)
    snapshot = source.copy(deep=True)
    assert unit_spec(units) == ("rate", "mm/h", 3600.0)
    assert physical_variables(source) == ["pr"]
    actual = normalize_dataset(source)
    np.testing.assert_allclose(actual.pr, np.broadcast_to(DEPTH_RATE, actual.pr.shape))
    np.testing.assert_array_equal(actual.pr.isnull(), source.pr.isnull())
    attrs = actual.pr.attrs
    assert attrs["units"] == "mm/h"
    assert attrs["standard_name"] == "lwe_precipitation_rate"
    assert attrs["threshold"] == pytest.approx(0.36)
    assert attrs["zerovalue"] == attrs["pysteps_no_detection_value"] == 0
    np.testing.assert_allclose(attrs["valid_range"], [0, 72])
    assert attrs["pysteps_source_units"] == units
    assert attrs["pysteps_source_standard_name"] == "precipitation_flux"
    assert attrs["pysteps_liquid_water_density_kg_m3"] == 1000.0
    assert attrs["long_name"] == "liquid water equivalent precipitation rate"
    assert attrs["pysteps_source_long_name"] == "total precipitation mass flux"
    assert "1000 kg m-3" in actual.attrs["history"]
    xr.testing.assert_identical(source, snapshot)
    xr.testing.assert_identical(normalize_dataset(actual), actual)


@pytest.mark.parametrize("product", ["instantaneous_rate", "mean_rate"])
@pytest.mark.parametrize("packed", [False, True])
def test_mass_flux_export_uses_normalized_precision(product, packed, tmp_path):
    """Output packing is in mm/h and preserves bounds, dry pixels and missing."""
    source = mass_flux_source(product)
    snapshot = source.copy(deep=True)
    path = io.export_netcdf_cf(
        source,
        tmp_path / "normalized.nc",
        variables="pr",
        encoding={"pr": {"dtype": "int16", "scale_factor": 0.1}} if packed else None,
    )
    expected = np.broadcast_to(DEPTH_RATE, source.pr.shape)
    expected = np.round(expected / 0.1) * 0.1 if packed else expected
    with xr.open_dataset(path, decode_coords="all") as independent:
        np.testing.assert_allclose(independent.pr, expected, atol=1e-12)
        np.testing.assert_array_equal(independent.pr.isnull(), source.pr.isnull())
        np.testing.assert_array_equal(independent.time, source.time)
        assert independent.pr.attrs["units"] == "mm h-1"
        assert independent.pr.attrs["standard_name"] == "lwe_precipitation_rate"
        assert independent.pr.attrs["cell_methods"] == source.pr.attrs["cell_methods"]
        if product == "mean_rate":
            bounds = source.time.attrs["bounds"]
            np.testing.assert_array_equal(independent[bounds], source[bounds])
    with netCDF4.Dataset(path) as raw:
        if packed:
            field = raw["pr"]
            field.set_auto_maskandscale(False)
            assert field.scale_factor == 0.1
            assert field._FillValue == -32768
            assert field[0, 0, 1] == 4  # 0.36 mm/h, rounded to 0.4 mm/h.
            assert field[0, 0, 0] == 0
            assert field[0, 1, 2] == -32768
    actual = io.import_netcdf_cf(path)
    np.testing.assert_allclose(actual.pr, expected, atol=1e-12)
    assert actual.pr.attrs["pysteps_source_standard_name"] == "precipitation_flux"
    xr.testing.assert_identical(source, snapshot)


@pytest.mark.parametrize("load", [True, False])
def test_external_packed_mass_flux_decodes_once_before_conversion(load, tmp_path):
    """External packing and raw validity masks precede the mass-to-depth conversion."""
    source = mass_flux_source("mean_rate")
    source.pr.values[0, 1, 1] = 0.0201  # Raw code 201 is outside [0, 200].
    source.pr.attrs["valid_range"] = np.array([0, 200], dtype="int16")
    path = tmp_path / "external.nc"
    source.to_netcdf(
        path,
        encoding={
            "pr": {
                "dtype": "int16",
                "scale_factor": 0.0001,
                "add_offset": 0.0,
                "_FillValue": np.int16(-32768),
            }
        },
    )
    with io.import_netcdf_cf(path, load=load) as actual:
        if not load:
            assert hasattr(actual.pr.data, "dask")
        expected = np.broadcast_to(DEPTH_RATE, actual.pr.shape).copy()
        expected[0, 1, 1] = np.nan
        np.testing.assert_allclose(actual.pr, expected, atol=1e-12)
        assert actual.pr.attrs["pysteps_source_scale_factor"] == 0.0001
        assert actual.pr.attrs["pysteps_source_units"] == "kg m-2 s-1"
        bounds = source.time.attrs["bounds"]
        np.testing.assert_array_equal(actual[bounds], source[bounds])


@pytest.mark.parametrize(
    "standard,units",
    [
        ("precipitation_flux", "mm h-1"),
        ("lwe_precipitation_rate", "kg m-2 s-1"),
        ("precipitation_flux", "mm"),
    ],
)
def test_mass_flux_rejects_contradictory_standard_name_and_units(standard, units):
    """Mass flux and depth rate are different dimensions in the CF metadata."""
    ds = mass_flux_source()
    ds.pr.attrs.update(standard_name=standard, units=units)
    with pytest.raises(ValueError, match="standard_name conflicts"):
        normalize_dataset(ds)


def test_mass_flux_mean_rate_integrates_using_actual_interval_bounds():
    """A mean flux converts to depth with interval duration, not sample cadence."""
    source = mass_flux_source("mean_rate")
    bounds = source[source.time.attrs["bounds"]].values
    hours = (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "h")
    actual = to_raindepth(source)
    np.testing.assert_allclose(
        actual.precip_accum, DEPTH_RATE[None, :, :] * hours[:, None, None]
    )


def test_legacy_array_mass_flux_records_density_and_converts_thresholds():
    """The shared unit helper also keeps legacy array metadata consistent."""
    metadata = dict(
        unit="kg m-2 s-1",
        x1=1600000.0,
        x2=1603000.0,
        y1=5900000.0,
        y2=5902000.0,
        projection="EPSG:2193",
        institution="synthetic test",
        cartesian_unit="m",
        yorigin="upper",
        threshold=0.0001,
        zerovalue=0.0,
    )
    actual = convert_input_to_xarray_dataset(MASS_FLUX.copy(), None, metadata)
    np.testing.assert_allclose(actual.precip_intensity, DEPTH_RATE)
    assert actual.precip_intensity.attrs["threshold"] == pytest.approx(0.36)
    assert actual.precip_intensity.attrs["pysteps_liquid_water_density_kg_m3"] == 1000


def test_other_mass_fluxes_are_not_recognized_as_precipitation():
    """Identical physical dimensions do not turn evaporation into precipitation."""
    source = mass_flux_source()
    source.pr.attrs["standard_name"] = "water_evaporation_flux"
    assert physical_variables(source) == []
