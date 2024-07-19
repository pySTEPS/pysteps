# -*- coding: utf-8 -*-
"""
pysteps.converters
==================

Module with data converter functions.

.. autosummary::
    :toctree: ../generated/

    convert_to_xarray_dataset
"""

import numpy as np
import numpy.typing as npt
import pyproj
import xarray as xr

from pysteps.utils.conversion import cf_parameters_from_unit

# TODO(converters): Write methods for converting Proj.4 projection definitions
# into CF grid mapping attributes. Currently this has been implemented for
# the stereographic projection.
# The conversions implemented here are take from:
# https://github.com/cf-convention/cf-convention.github.io/blob/master/wkt-proj-4.md


def _convert_proj4_to_grid_mapping(proj4str):
    tokens = proj4str.split("+")

    d = {}
    for t in tokens[1:]:
        t = t.split("=")
        if len(t) > 1:
            d[t[0]] = t[1].strip()

    params = {}
    # TODO(exporters): implement more projection types here
    if d["proj"] == "stere":
        grid_mapping_var_name = "polar_stereographic"
        grid_mapping_name = "polar_stereographic"
        v = d["lon_0"] if d["lon_0"][-1] not in ["E", "W"] else d["lon_0"][:-1]
        params["straight_vertical_longitude_from_pole"] = float(v)
        v = d["lat_0"] if d["lat_0"][-1] not in ["N", "S"] else d["lat_0"][:-1]
        params["latitude_of_projection_origin"] = float(v)
        if "lat_ts" in list(d.keys()):
            params["standard_parallel"] = float(d["lat_ts"])
        elif "k_0" in list(d.keys()):
            params["scale_factor_at_projection_origin"] = float(d["k_0"])
        params["false_easting"] = float(d["x_0"])
        params["false_northing"] = float(d["y_0"])
    elif d["proj"] == "aea":  # Albers Conical Equal Area
        grid_mapping_var_name = "proj"
        grid_mapping_name = "albers_conical_equal_area"
        params["false_easting"] = float(d["x_0"]) if "x_0" in d else float(0)
        params["false_northing"] = float(d["y_0"]) if "y_0" in d else float(0)
        v = d["lon_0"] if "lon_0" in d else float(0)
        params["longitude_of_central_meridian"] = float(v)
        v = d["lat_0"] if "lat_0" in d else float(0)
        params["latitude_of_projection_origin"] = float(v)
        v1 = d["lat_1"] if "lat_1" in d else float(0)
        v2 = d["lat_2"] if "lat_2" in d else float(0)
        params["standard_parallel"] = (float(v1), float(v2))
    else:
        print("unknown projection", d["proj"])
        return None, None, None

    return grid_mapping_var_name, grid_mapping_name, params


def compute_lat_lon(
    x_r: npt.ArrayLike, y_r: npt.ArrayLike, projection: str
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    x_2d, y_2d = np.meshgrid(x_r, y_r)
    pr = pyproj.Proj(projection)
    lon, lat = pr(x_2d.flatten(), y_2d.flatten(), inverse=True)
    return lat.reshape(x_2d.shape), lon.reshape(x_2d.shape)


def convert_to_xarray_dataset(
    precip: np.ndarray,
    quality: np.ndarray | None,
    metadata: dict[str, str | float | None],
) -> xr.Dataset:
    """
    Read a precip, quality, metadata tuple as returned by the importers
    (:py:mod:`pysteps.io.importers`) and return an xarray dataset containing
    this data.

    Parameters
    ----------
    precip: array
        2D array containing imported precipitation data.
    quality: array, None
        2D array containing the quality values of the imported precipitation
        data, can be None.
    metadata: dict
        Metadata dictionary containing the attributes described in the
        documentation of :py:mod:`pysteps.io.importers`.

    Returns
    -------
    out: Dataset
        A CF compliant xarray dataset, which contains all data and metadata.

    """
    var_name, attrs = cf_parameters_from_unit(metadata["unit"])
    h, w = precip.shape
    x_r = np.linspace(metadata["x1"], metadata["x2"], w + 1)[:-1]
    x_r += 0.5 * (x_r[1] - x_r[0])
    y_r = np.linspace(metadata["y1"], metadata["y2"], h + 1)[:-1]
    y_r += 0.5 * (y_r[1] - y_r[0])

    # flip yr vector if yorigin is upper
    if metadata["yorigin"] == "upper":
        y_r = np.flip(y_r)

    lat, lon = compute_lat_lon(x_r, y_r, metadata["projection"])

    (
        grid_mapping_var_name,
        grid_mapping_name,
        grid_mapping_params,
    ) = _convert_proj4_to_grid_mapping(metadata["projection"])

    data_vars = {
        var_name: (
            ["y", "x"],
            precip,
            {
                "units": attrs["units"],
                "standard_name": attrs["standard_name"],
                "long_name": attrs["long_name"],
                "grid_mapping": "projection",
                "transform": metadata["transform"],
                "accutime": metadata["accutime"],
                "threshold": metadata["threshold"],
                "zerovalue": metadata["zerovalue"],
                "zr_a": metadata["zr_a"],
                "zr_b": metadata["zr_b"],
            },
        )
    }
    if quality is not None:
        data_vars["quality"] = (
            ["y", "x"],
            quality,
            {
                "units": "1",
                "standard_name": "quality_flag",
                "grid_mapping": "projection",
            },
        )
    coords = {
        "y": (
            ["y"],
            y_r,
            {
                "axis": "Y",
                "long_name": "y-coordinate in Cartesian system",
                "standard_name": "projection_y_coordinate",
                "units": metadata["cartesian_unit"],
            },
        ),
        "x": (
            ["x"],
            x_r,
            {
                "axis": "X",
                "long_name": "x-coordinate in Cartesian system",
                "standard_name": "projection_x_coordinate",
                "units": metadata["cartesian_unit"],
            },
        ),
        "lon": (
            ["y", "x"],
            lon,
            {
                "long_name": "longitude coordinate",
                "standard_name": "longitude",
                # TODO(converters): Don't hard-code the unit.
                "units": "degrees_east",
            },
        ),
        "lat": (
            ["y", "x"],
            lat,
            {
                "long_name": "latitude coordinate",
                "standard_name": "latitude",
                # TODO(converters): Don't hard-code the unit.
                "units": "degrees_north",
            },
        ),
    }
    if grid_mapping_var_name is not None:
        coords[grid_mapping_name] = (
            [],
            None,
            {"grid_mapping_name": grid_mapping_name, **grid_mapping_params},
        )
    attrs = {
        "Conventions": "CF-1.7",
        "institution": metadata["institution"],
        "projection": metadata["projection"],
        "precip_var": var_name,
    }
    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    return dataset.sortby(["y", "x"])
