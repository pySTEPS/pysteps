# -*- coding: utf-8 -*-
"""
pysteps.converters
==================

Module with xarray helper functions.

.. autosummary::
    :toctree: ../generated/

    convert_to_xarray_dataset
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
import pyproj
import xarray as xr


# TODO(converters): Write methods for converting Proj.4 projection definitions
# into CF grid mapping attributes. Currently this has been implemented for
# the stereographic projection.
# The conversions implemented here are take from:
# https://github.com/cf-convention/cf-convention.github.io/blob/master/wkt-proj-4.md


def cf_parameters_from_unit(unit: str) -> tuple[str, dict[str, str | None]]:
    if unit == "mm/h":
        var_name = "precip_intensity"
        var_standard_name = "instantaneous_precipitation_rate"
        var_long_name = "instantaneous precipitation rate"
        var_unit = "mm/h"
    elif unit == "mm":
        var_name = "precip_accum"
        var_standard_name = "accumulated_precipitation"
        var_long_name = "accumulated precipitation"
        var_unit = "mm"
    elif unit == "dBZ":
        var_name = "reflectivity"
        var_long_name = "equivalent reflectivity factor"
        var_standard_name = "equivalent_reflectivity_factor"
        var_unit = "dBZ"
    else:
        raise ValueError(f"unknown unit {unit}")

    return var_name, {
        "standard_name": var_standard_name,
        "long_name": var_long_name,
        "units": var_unit,
    }


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


def convert_input_to_xarray_dataset(
    precip: np.ndarray,
    quality: np.ndarray | None,
    metadata: dict[str, str | float | None],
    startdate: datetime | None = None,
    timestep: int | None = None,
) -> xr.Dataset:
    """
    Read a precip, quality, metadata tuple as returned by the importers
    (:py:mod:`pysteps.io.importers`) and return an xarray dataset containing
    this data.

    Parameters
    ----------
    precip: array
        ND array containing imported precipitation data.
    quality: array, None
        ND array containing the quality values of the imported precipitation
        data, can be None.
    metadata: dict
        Metadata dictionary containing the attributes described in the
        documentation of :py:mod:`pysteps.io.importers`.
    startdate: datetime, None
        Datetime object containing the start date and time for the nowcast
    timestep: int, None
        The timestep in seconds between 2 consecutive fields, mandatory if
        the precip has 3 or more dimensions

    Returns
    -------
    out: Dataset
        A CF compliant xarray dataset, which contains all data and metadata.

    """
    var_name, attrs = cf_parameters_from_unit(metadata["unit"])

    dims = None
    timesteps = None
    ens_number = None

    if precip.ndim == 4:
        ens_number, timesteps, h, w = precip.shape
        dims = ["ens_number", "time", "y", "x"]

        if startdate is None:
            raise Exception("startdate missing")
        if timestep is None:
            raise Exception("timestep missing")

    elif precip.ndim == 3:
        timesteps, h, w = precip.shape
        dims = ["time", "y", "x"]

        if startdate is None:
            raise Exception("startdate missing")
        if timestep is None:
            raise Exception("timestep missing")

    elif precip.ndim == 2:
        h, w = precip.shape
        dims = ["y", "x"]
    else:
        raise Exception(f"Precip field shape: {precip.shape} not supported")

    x_r = np.linspace(metadata["x1"], metadata["x2"], w + 1)[:-1]
    x_r += 0.5 * (x_r[1] - x_r[0])
    y_r = np.linspace(metadata["y1"], metadata["y2"], h + 1)[:-1]
    y_r += 0.5 * (y_r[1] - y_r[0])

    if "xpixelsize" in metadata:
        xpixelsize = metadata["xpixelsize"]
    else:
        xpixelsize = x_r[1] - x_r[0]

    if "ypixelsize" in metadata:
        ypixelsize = metadata["ypixelsize"]
    else:
        ypixelsize = y_r[1] - y_r[0]

    if x_r[1] - x_r[0] != xpixelsize:
        # XR: This should be an error, but the importers don't always provide correct pixelsizes
        warnings.warn(
            "xpixelsize does not match x1, x2 and array shape, using xpixelsize for pixel size"
        )
    if y_r[1] - y_r[0] != ypixelsize:
        # XR: This should be an error, but the importers don't always provide correct pixelsizes
        warnings.warn(
            "ypixelsize does not match y1, y2 and array shape, using ypixelsize for pixel size"
        )

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
            dims,
            precip,
            {
                "units": attrs["units"],
                "standard_name": attrs["standard_name"],
                "long_name": attrs["long_name"],
                "grid_mapping": "projection",
            },
        )
    }

    # XR: accutime vs timestep, what should be optional and what required?
    optional_metadata_keys = ["transform", "accutime", "zr_a", "zr_b"]

    required_metadata_keys = ["threshold", "zerovalue"]

    for metadata_field in optional_metadata_keys:
        if metadata_field in metadata:
            data_vars[var_name][2][metadata_field] = metadata[metadata_field]

    for metadata_field in required_metadata_keys:
        data_vars[var_name][2][metadata_field] = metadata[metadata_field]

    if quality is not None:
        data_vars["quality"] = (
            dims,
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
                "stepsize": ypixelsize,
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
                "stepsize": xpixelsize,
            },
        ),
        "lon": (
            ["y", "x"],
            lon,
            {
                "long_name": "longitude coordinate",
                "standard_name": "longitude",
                "units": "degrees_east",
            },
        ),
        "lat": (
            ["y", "x"],
            lat,
            {
                "long_name": "latitude coordinate",
                "standard_name": "latitude",
                "units": "degrees_north",
            },
        ),
    }

    if ens_number is not None:
        coords["ens_number"] = (
            ["ens_number"],
            list(range(1, ens_number + 1, 1)),
            {
                "long_name": "ensemble member",
                "standard_name": "realization",
                "units": "",
            },
        )

    if timesteps is not None:
        startdate_str = datetime.strftime(startdate, "%Y-%m-%d %H:%M:%S")

        coords["time"] = (
            ["time"],
            [
                startdate + timedelta(seconds=float(second))
                for second in np.arange(timesteps) * timestep
            ],
            {"long_name": "forecast time", "stepsize": timestep},
            {"units": "seconds since %s" % startdate_str},
        )
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
    return dataset.sortby(dims)


def convert_output_to_xarray_dataset(
    dataset: xr.Dataset, timesteps: int | list[int], output: np.ndarray
) -> xr.Dataset:
    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs

    last_timestamp = (
        dataset["time"][-1].values.astype("datetime64[us]").astype(datetime)
    )
    time_metadata = dataset["time"].attrs
    time_encoding = dataset["time"].encoding
    timestep_seconds = dataset["time"].attrs["stepsize"]
    dataset = dataset.drop_vars([precip_var]).drop_dims(["time"])
    if isinstance(timesteps, int):
        timesteps = list(range(1, timesteps + 1))
    next_timestamps = [
        last_timestamp + timedelta(seconds=timestep_seconds * i) for i in timesteps
    ]
    dataset = dataset.assign_coords(
        {"time": (["time"], next_timestamps, time_metadata, time_encoding)}
    )

    if output.ndim == 4:
        dataset = dataset.assign_coords(
            {
                "ens_number": (
                    ["ens_number"],
                    list(range(1, output.shape[0] + 1)),
                    {
                        "long_name": "ensemble member",
                        "standard_name": "realization",
                        "units": "",
                    },
                )
            }
        )
        dataset[precip_var] = (["ens_number", "time", "y", "x"], output, metadata)
    else:
        dataset[precip_var] = (["time", "y", "x"], output, metadata)

    return dataset


def verify_xarray_dataset(dataset: xr.Dataset) -> bool:
    """
    Verify that the input xarray dataset contains all the required 
    coordinate.

    Paramaters
    ----------
    dataset: xr.Dataset
        xarray dataset to be cheched.
    
    Returns
    -------
    bool:
        True if the dataset contains the required coordinates and false otherwise.  
    """
    is_valid=True

    condition_1=len(dataset.sizes)==2 and set(dataset.sizes)==set(['x','y'])
    condition_2=len(dataset.sizes)==3 and set(dataset.sizes)==set(['x','y','time'])
    condition_3=len(dataset.sizes)==4 and set(dataset.sizes)==set(['x','y','time','ens_number'])

    variables_checklist=['precip_intensity','precip_accum','reflectivity']
    attribute_checklist=['projection','institution','precip_var']
    precip_var_checklist=['units','threshold','zerovalue']
    coord_var_checklist=['units','stepsize']
    velocity_checklist=['velocity_x','velocity_y' ] 

    variables_names=list(dataset.variables.keys())

    if not any([condition_1,condition_2,condition_3]):
        print("input specification incorrect.\n"
                f"Dimension mismatch "
                "expected for (x,y) for 2 dimension,\n"
                "(x,y,time) for three dimension\n or"
                "(x,y,time,ens_number) for four dimension."
                )
        is_valid=False
        
    elif  len(set(variables_names).intersection(set(variables_checklist)))!=1:
        print(
                    "input specification incorrect.\n"
                    f"Either None or more that one of : {', '.join(variables_checklist)},  detected\n"
                    f"expected exactly one of : {', '.join(variables_checklist)}"
                )
        is_valid=False
         
    elif len(set(variables_names).intersection(set(velocity_checklist)))==1:
        print(
                    "input specification incorrect.\n"
                    "Either velocity_x or velocity_y present\n"
                    "expected both velocity_x and velocity_y or none"
                )
        is_valid=False
    
    elif not all(attribute in dataset.attrs for attribute in attribute_checklist):
        print(
                    "input specification incorrect.\n"
                    f"One of the :{', '.join(attribute_checklist)} missing\n"
                    "expected all to be present"
                )
        is_valid=False
    
    elif not all(attribute in dataset.variables['precipitation'].attrs for attribute in precip_var_checklist):
        print(
                    "input specification incorrect.\n"
                    f"One of the : {', '.join(precip_var_checklist)} missing\n"
                    "expected all to be present"
        )
        is_valid=False
    
    elif dataset.coords is not None:
        for var in dataset.coords:
            if not all(attribute in dataset.coords[var].attrs for attribute in coord_var_checklist):
                print("Input specification incorrect\n"
                    "At least one attribute not present in a coordinate variable\n"
                    f"Expected all the coordinate variable to have:  {', '.join(coord_var_checklist)}, to be present ")
                is_valid=False
                break
                
    

    return is_valid
     
     

