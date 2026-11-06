"""Check exported CF files with Iris, independently of pySTEPS and xarray.

Run with the optional acceptance dependencies scitools-iris and netCDF4::

    python -m pysteps.tests.cf_netcdf_independent_acceptance \
        --output /tmp/iris-validation.json /tmp/float-exports /tmp/packed-exports
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import cf_units
import iris
import netCDF4
import numpy as np
from pyproj import CRS, Transformer

QUANTITIES = {
    "lwe_precipitation_rate",
    "lwe_thickness_of_precipitation_amount",
    "equivalent_reflectivity_factor",
}


def check_file(path):
    """Assert independently decoded values, support, member labels and CRS."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        iris.loading.LOAD_PROBLEMS.reset()
        with iris.FUTURE.context(datum_support=False):
            cubes = iris.load(str(path))
    checked = {}
    with netCDF4.Dataset(path) as raw:
        fields = {
            name: var
            for name, var in raw.variables.items()
            if getattr(var, "standard_name", None) in QUANTITIES
        }
        for name, var in fields.items():
            matches = [c for c in cubes if c.var_name == name]
            assert len(matches) == 1, (path, name, "missing/ambiguous cube")
            cube = matches[0]
            assert cube.standard_name == var.standard_name
            assert str(cube.units) == var.units
            expected = np.ma.masked_invalid(var[:].astype(float))
            actual = np.ma.masked_invalid(cube.data.astype(float))
            np.testing.assert_array_equal(
                np.ma.getmaskarray(actual), np.ma.getmaskarray(expected)
            )
            np.testing.assert_allclose(
                actual.compressed(), expected.compressed(), rtol=0, atol=1e-10
            )
            assert cube.shape == var.shape
            for dim, size in zip(var.dimensions, var.shape):
                assert cube.coord_dims(cube.coord(var_name=dim)) == (
                    var.dimensions.index(dim),
                )
            coordinate_names = list(var.dimensions)
            coordinate_names += getattr(var, "coordinates", "").split()
            for coord_name in coordinate_names:
                source = raw[coord_name]
                if getattr(source, "standard_name", "") not in {
                    "time",
                    "forecast_reference_time",
                    "realization",
                    "projection_x_coordinate",
                    "projection_y_coordinate",
                }:
                    continue
                coord = cube.coord(var_name=coord_name)
                units = cf_units.Unit(
                    source.units, calendar=getattr(source, "calendar", None)
                )
                points = coord.units.convert(coord.points, units)
                np.testing.assert_allclose(
                    points, np.atleast_1d(source[:]), rtol=0, atol=1e-6
                )
                bounds = getattr(source, "bounds", None)
                assert coord.has_bounds() == bool(bounds)
                if bounds:
                    np.testing.assert_allclose(
                        coord.units.convert(coord.bounds, units),
                        raw[bounds][:],
                        rtol=0,
                        atol=1e-6,
                    )
            for method in ("point", "mean", "sum"):
                if ": " + method in getattr(var, "cell_methods", ""):
                    assert any(m.method == method for m in cube.cell_methods)
            x, y = (
                cube.coord("projection_x_coordinate"),
                cube.coord("projection_y_coordinate"),
            )
            assert x.coord_system is not None and x.coord_system == y.coord_system
            independent = CRS.from_user_input(x.coord_system.as_cartopy_crs())
            declared = CRS.from_cf(raw[var.grid_mapping].__dict__)
            xx, yy = np.meshgrid(x.points[[0, -1]], y.points[[0, -1]])
            # Compare geolocation, allowing equivalent parameterizations and datums.
            lonlat = [
                Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True).transform(
                    xx, yy
                )
                for crs in (independent, declared)
            ]
            np.testing.assert_allclose(lonlat[0], lonlat[1], rtol=0, atol=1e-7)
            checked[name] = {
                "standard_name": cube.standard_name,
                "units": str(cube.units),
                "shape": list(cube.shape),
                "missing_pixels": int(np.ma.getmaskarray(actual).sum()),
                "values_masks_coordinates_bounds_members_cell_methods_crs": "pass",
            }
        assert fields, (path, "no physical fields")
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fields": checked,
        "reader_warnings": sorted(set(str(w.message) for w in caught)),
        "reader_load_problems": [
            str(problem) for problem in iris.loading.LOAD_PROBLEMS.problems
        ],
    }


def main():
    """Parse command-line options and run the documented file checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = {}
    for directory in args.directories:
        for path in sorted(directory.glob("*.nc")):
            results[str(path)] = check_file(path)
    assert results
    report = {
        "reader": "scitools-iris",
        "version": iris.__version__,
        "files_checked": len(results),
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Iris {iris.__version__}: {len(results)} files passed")


if __name__ == "__main__":
    main()
