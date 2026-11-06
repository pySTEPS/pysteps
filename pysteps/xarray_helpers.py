# -*- coding: utf-8 -*-
"""
pysteps.xarray_helpers
======================

Convert legacy input arrays and forecast output to physical xarray datasets.

.. autosummary::
    :toctree: ../generated/

    cf_parameters_from_unit
    validate_dataset
    compute_lat_lon
    geodata_from_dataset
    convert_input_to_xarray_dataset
    convert_output_to_xarray_dataset
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
import pyproj
import xarray as xr

from pysteps.dataset_metadata import (
    cadence_seconds,
    cf_grid_mapping,
    checked_times,
    dataset_crs,
    linked,
    normalize_dataset,
    replace_time_series,
    set_time_metadata,
    temporal_method,
    time_bounds,
    unit_spec,
    utc_datetime,
    variable_metadata,
)


def cf_parameters_from_unit(unit):
    """
    Return a default variable name and canonical CF attributes.

    Parameters
    ----------
    unit : str
        Supported physical units.

    Returns
    -------
    name : str
        Suggested variable name.
    attrs : dict
        Canonical units, standard_name, and long_name.
    """
    return variable_metadata(unit)


def _convert_proj4_to_grid_mapping(projection):
    """Adapt a supported CRS definition to the legacy grid-mapping tuple."""
    attrs = cf_grid_mapping(projection, projected=False)
    name = attrs.pop("grid_mapping_name")
    return name, name, attrs


def validate_dataset(dataset, **kwargs):
    """
    Validate and normalize a physical dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    **kwargs : dict
        Options forwarded to pysteps.dataset_metadata.normalize_dataset,
        including variable, legacy, require_model, and validate_values.

    Returns
    -------
    out : xarray.Dataset
        Dataset with updated values and metadata.
    """
    return normalize_dataset(dataset, **kwargs)


def compute_lat_lon(
    x_r: npt.ArrayLike, y_r: npt.ArrayLike, projection: str, coordinate_unit=None
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Compute geographic coordinates at rectilinear grid centres.

    Parameters
    ----------
    x_r, y_r : array_like
        One-dimensional grid-centre coordinates along x and y.
    projection : str or pyproj.CRS
        Source CRS definition accepted by pyproj.
    coordinate_unit : {None, 'm', 'km'}, optional
        Units of projected coordinates. Default to the units of the source CRS.

    Returns
    -------
    lat, lon : numpy.ndarray
        Two-dimensional latitude and longitude arrays in degrees, of shape
        (len(y_r), len(x_r)).
    """
    x_2d, y_2d = np.meshgrid(x_r, y_r)
    crs = pyproj.CRS.from_user_input(projection)
    if coordinate_unit is not None and not crs.is_geographic:
        source = crs.source_crs if crs.is_bound else crs
        scale = {"m": 1.0, "km": 1000.0}[coordinate_unit] / source.axis_info[
            0
        ].unit_conversion_factor
        x_2d, y_2d = x_2d * scale, y_2d * scale
    lon, lat = pyproj.Transformer.from_crs(
        crs, crs.geodetic_crs, always_xy=True
    ).transform(x_2d, y_2d)
    return lat, lon


def geodata_from_dataset(dataset):
    """
    Build plotting geodata from a dataset with grid metadata.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.

    Returns
    -------
    out : dict
        Projection, pixel sizes, domain edges, cartesian units, and y origin for
        plot_precip_field.
    """
    x = dataset.x.values
    y = dataset.y.values
    dx = dataset.x.attrs["stepsize"]
    dy = dataset.y.attrs["stepsize"]
    yorigin = "upper" if dy < 0 else "lower"
    y1, y2 = y[0] - dy * 0.5, y[-1] + dy * 0.5
    return {
        "projection": dataset.attrs["projection"],
        "x1": x[0] - dx * 0.5,
        "x2": x[-1] + dx * 0.5,
        "y1": min(y1, y2),
        "y2": max(y1, y2),
        "yorigin": yorigin,
        "xpixelsize": dx,
        "ypixelsize": abs(dy),
        "cartesian_unit": dataset.x.attrs["units"],
    }


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
    timestep: float, optional
        Sampling interval in seconds, including fractional seconds. Required
        when precip has a time dimension. Accumulation duration is supplied
        separately by metadata['accutime'] in minutes.

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

    dx = (metadata["x2"] - metadata["x1"]) / w
    dy = (metadata["y2"] - metadata["y1"]) / h
    if dx <= 0 or dy <= 0:
        raise ValueError("Grid outer edges must define positive pixel sizes")
    x_r = metadata["x1"] + (np.arange(w) + 0.5) * dx
    y_r = metadata["y1"] + (np.arange(h) + 0.5) * dy

    sizes = {"xpixelsize": dx, "ypixelsize": dy}
    for key, actual in list(sizes.items()):
        if key in metadata:
            if np.isclose(metadata[key], actual, rtol=1e-7, atol=1e-6):
                sizes[key] = float(metadata[key])
            else:
                warnings.warn(
                    f"{key} conflicts with grid edges and shape; using coordinate spacing",
                    UserWarning,
                )
    xpixelsize, ypixelsize = sizes["xpixelsize"], sizes["ypixelsize"]

    # flip yr vector if yorigin is upper
    if metadata["yorigin"] == "upper":
        y_r = np.flip(y_r)

    crs = pyproj.CRS.from_user_input(metadata["projection"])
    native_unit = (
        (crs.source_crs if crs.is_bound else crs).axis_info[0].unit_conversion_factor
    )
    coordinate_scale = (
        1.0
        if crs.is_geographic
        else {"m": 1.0, "km": 1000.0}[metadata["cartesian_unit"]] / native_unit
    )
    lat, lon = compute_lat_lon(x_r * coordinate_scale, y_r * coordinate_scale, crs)

    (
        grid_mapping_var_name,
        grid_mapping_name,
        grid_mapping_params,
    ) = _convert_proj4_to_grid_mapping(metadata["projection"])

    precip_attrs = {
        "units": attrs["units"],
        "standard_name": attrs["standard_name"],
        "long_name": attrs["long_name"],
    }
    if grid_mapping_name is not None:
        precip_attrs["grid_mapping"] = grid_mapping_name

    _, canonical_unit, multiplier = unit_spec(metadata["unit"])
    if multiplier != 1 and metadata.get("transform") not in (None,):
        raise ValueError(
            "Normalize physical units before applying a modelling transform"
        )
    precip_attrs["units"] = canonical_unit
    precip_attrs["cell_methods"] = "time: " + metadata.get(
        "temporal_method", "sum" if canonical_unit == "mm" else "point"
    )
    data_vars = {
        var_name: (
            dims,
            precip * multiplier if multiplier != 1 else precip,
            precip_attrs,
        )
    }

    # XR: accutime vs timestep, what should be optional and what required?
    optional_metadata_keys = ["transform", "accutime", "zr_a", "zr_b"]

    required_metadata_keys = ["threshold", "zerovalue"]

    for metadata_field in optional_metadata_keys:
        if metadata.get(metadata_field) is not None:
            data_vars[var_name][2][metadata_field] = metadata[metadata_field]

    for metadata_field in required_metadata_keys:
        if metadata.get(metadata_field) is not None:
            data_vars[var_name][2][metadata_field] = (
                metadata[metadata_field] * multiplier
            )

    if quality is not None:
        quality_attrs = {"units": "1", "long_name": "precipitation quality score"}
        if grid_mapping_name is not None:
            quality_attrs["grid_mapping"] = grid_mapping_name
        data_vars["quality"] = (dims, quality, quality_attrs)
    coords = {
        "y": (
            ["y"],
            y_r,
            {
                "axis": "Y",
                "long_name": "y-coordinate in Cartesian system",
                "standard_name": "projection_y_coordinate",
                "units": metadata["cartesian_unit"],
                "stepsize": (
                    -ypixelsize if metadata["yorigin"] == "upper" else ypixelsize
                ),
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
                "units": "1",
            },
        )

    if timesteps is not None:
        startdate = utc_datetime(startdate)
        startdate_str = str(startdate).replace("T", " ")

        coords["time"] = (
            ["time"],
            [
                startdate + np.timedelta64(int(round(float(second) * 1e6)), "us")
                for second in np.arange(timesteps) * timestep
            ],
            {
                "standard_name": "time",
                "long_name": "valid time",
                "stepsize": float(timestep),
            },
            {"units": "seconds since %s" % startdate_str},
        )
    if grid_mapping_var_name is not None:
        coords[grid_mapping_name] = (
            [],
            np.int32(0),
            {"grid_mapping_name": grid_mapping_name, **grid_mapping_params},
        )
    attrs = {
        "Conventions": "CF-1.8",
        "institution": metadata["institution"],
        "projection": metadata["projection"],
        "precip_var": var_name,
    }
    if crs.is_geographic:
        coords["x"][2].update(standard_name="longitude", units="degrees_east")
        coords["y"][2].update(standard_name="latitude", units="degrees_north")
    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    if timesteps is not None and temporal_method(dataset, var_name) != "point":
        bounds = time_bounds(dataset, var_name, legacy=True)
        if bounds is not None:
            dataset = set_time_metadata(dataset, var_name, bounds)
            dataset.attrs["history"] = (
                "Legacy importer duration interpreted as an interval ending at each supplied valid time"
            )
    return dataset


def convert_output_to_xarray_dataset(
    dataset,
    timesteps,
    output,
    *,
    timestep_seconds=None,
    interval_seconds=None,
    accumulation="interval",
    ensemble_ids=None,
):
    """
    Label forecast values with valid times and explicit temporal support.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    timesteps : int or sequence of float
        Number of forecast steps, or positive increasing leads in multiples of
        the forecast timestep.
    output : numpy.ndarray
        Forecast values of shape (time, y, x) or (member, time, y, x), in the
        active variable units.
    timestep_seconds : float, optional
        Forecast step length in seconds. Default to regular input cadence; an
        explicit value must agree with known input cadence.
    interval_seconds : float or array_like, optional
        Positive duration of each forecast interval in seconds. Default to a
        known fixed input duration; supply explicitly for varying durations.
    accumulation : {'interval', 'cumulative'}, optional
        Declare interval depths or already computed running totals from
        initialization. Default is interval. Cumulative support requires depth
        values.
    ensemble_ids : array_like, optional
        Member labels matching the output ensemble axis. Reuse matching input
        labels or generate sequential labels by default.

    Returns
    -------
    out : xarray.Dataset
        Dataset with updated values and metadata.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.

    Notes
    -----
    Use the last observation time as forecast initialization. This function labels
    values without integrating them. Point forecasts have no interval bounds.
    Unforecast physical products and their ancillary time series are removed.
    """
    variable = dataset.attrs["precip_var"]
    cadence = cadence_seconds(dataset.time)
    if timestep_seconds is None:
        timestep_seconds = cadence
    elif cadence is not None and not np.isclose(timestep_seconds, cadence):
        raise ValueError("Forecast timestep conflicts with input cadence")
    if (
        timestep_seconds is None
        or not np.isfinite(timestep_seconds)
        or timestep_seconds <= 0
    ):
        raise ValueError(
            "Forecast requires a positive timestep_seconds or regular input cadence"
        )
    if isinstance(timesteps, (int, np.integer)):
        leads = np.arange(1, timesteps + 1, dtype=float)
    else:
        leads = np.asarray(timesteps, dtype=float)
    if (
        leads.ndim != 1
        or not len(leads)
        or not np.isfinite(leads).all()
        or np.any(leads <= 0)
        or np.any(np.diff(leads) <= 0)
    ):
        raise ValueError("Forecast lead times must be positive and strictly increasing")
    output = np.asarray(output)
    if output.ndim not in (3, 4) or output.shape[-3:] != (
        len(leads),
        dataset.sizes["y"],
        dataset.sizes["x"],
    ):
        raise ValueError("Forecast output shape conflicts with leads and spatial grid")
    if accumulation not in ("interval", "cumulative"):
        raise ValueError("Unknown forecast accumulation type")
    origin = checked_times(dataset.time)[-1]
    times = origin + np.rint(leads * float(timestep_seconds) * 1e6).astype(
        "timedelta64[us]"
    )
    method = temporal_method(dataset, variable)
    bounds = None
    if method != "point":
        if accumulation == "cumulative":
            if unit_spec(dataset[variable].attrs["units"])[0] != "depth":
                raise ValueError("Cumulative forecast support requires depth values")
            bounds = np.column_stack([np.repeat(origin, len(times)), times])
        else:
            if interval_seconds is None:
                source = time_bounds(dataset, variable)
                if source is None:
                    raise ValueError(
                        "Forecast interval duration must be supplied explicitly"
                    )
                durations = (source[:, 1] - source[:, 0]) / np.timedelta64(1, "s")
                if not np.allclose(durations, durations[-1]) or durations[-1] <= 0:
                    raise ValueError(
                        "Variable input durations require interval_seconds"
                    )
                interval_seconds = float(durations[-1])
            duration = np.broadcast_to(interval_seconds, (len(times),))
            if not np.isfinite(duration).all() or np.any(duration <= 0):
                raise ValueError("Forecast intervals must have positive durations")
            bounds = np.column_stack(
                [times - np.rint(duration * 1e6).astype("timedelta64[us]"), times]
            )
    elif interval_seconds is not None or accumulation != "interval":
        raise ValueError("Point forecasts do not have accumulation intervals")
    # Construct only the selected forecast product; observations of other
    # quantities and their detection/quality fields have not been forecast.
    base = dataset.drop_vars(
        [
            name
            for name in dataset.data_vars
            if name != variable
            and any(dim not in ("x", "y") for dim in dataset[name].dims)
        ]
    )
    base = base.drop_vars(
        [
            name
            for name in ("quality", "velocity_x", "velocity_y", "velocity_quality")
            if name in base
        ]
    )
    if output.ndim == 4:
        if ensemble_ids is None:
            ensemble_ids = (
                dataset.ens_number.values
                if "ens_number" in dataset
                and len(dataset.ens_number) == output.shape[0]
                else np.arange(1, output.shape[0] + 1)
            )
        if len(ensemble_ids) != output.shape[0] or len(np.unique(ensemble_ids)) != len(
            ensemble_ids
        ):
            raise ValueError(
                "Forecast ensemble identifiers must be unique and match output"
            )
        if "ens_number" in base.dims:
            base = base.isel(ens_number=0, drop=True)
        dims = ("ens_number", "time", "y", "x")
    else:
        if "ens_number" in base.dims:
            base = base.isel(ens_number=0, drop=True)
        dims = ("time", "y", "x")
    attrs = dataset[variable].attrs.copy()
    attrs.pop("ancillary_variables", None)
    data = xr.DataArray(output, dims=dims)
    result = replace_time_series(base, variable, data, times, bounds, keep_other=False)
    result[variable].attrs = {**attrs, **result[variable].attrs}
    result[variable].attrs.pop("ancillary_variables", None)
    result = set_time_metadata(result, variable, bounds, method)
    if len(times) == 1:
        result.time.attrs["stepsize"] = float(timestep_seconds)
    if output.ndim == 4:
        result = result.assign_coords(
            ens_number=(
                "ens_number",
                ensemble_ids,
                {"standard_name": "realization", "units": "1"},
            )
        )
    result = result.assign_coords(
        forecast_reference_time=xr.DataArray(
            origin,
            attrs={
                "standard_name": "forecast_reference_time",
                "long_name": "forecast reference time",
            },
        )
    )
    result.forecast_reference_time.encoding = {
        "units": "seconds since " + str(origin).replace("T", " "),
        "calendar": "proleptic_gregorian",
        "dtype": "float64",
        "_FillValue": None,
    }
    return result
