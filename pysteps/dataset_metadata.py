"""
pysteps.dataset_metadata
========================

Validate and normalize physical metadata for the shared dataset contract.

``normalize_dataset`` is the I/O-level validator. Model-specific requirements
are checked with ``require_model=True``. Numerical conversions and
the existing importer adapters also use the smaller helpers in this module.
"""

import re
from datetime import datetime, timezone

import numpy as np
import pyproj
import xarray as xr

STANDARD_NAMES = {
    "rate": "lwe_precipitation_rate",
    "depth": "lwe_thickness_of_precipitation_amount",
    "reflectivity": "equivalent_reflectivity_factor",
}
UNITS = {"rate": "mm/h", "depth": "mm", "reflectivity": "dBZ"}
_RATE_UNITS = {
    "mm/h": 1.0,
    "mm h-1": 1.0,
    "mm hr-1": 1.0,
    "mm/hour": 1.0,
    "mm s-1": 3600.0,
    "mm/s": 3600.0,
    "m s-1": 3600000.0,
    "m/s": 3600000.0,
    "m h-1": 1000.0,
}
# Mass flux becomes a liquid-water-equivalent depth rate using this conventional
# reference density, independent of the actual precipitation phase or density.
_LIQUID_WATER_DENSITY_KG_M3 = 1000.0
_MASS_FLUX_UNITS = {
    "kg m-2 s-1",
    "kg m^-2 s^-1",
    "kg m**-2 s**-1",
    "kg/m2/s",
    "kg/m^2/s",
    "kg m-2/s",
}
_DEPTH_UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
_LENGTH_UNITS = {
    "m": 1.0,
    "metre": 1.0,
    "meter": 1.0,
    "km": 1000.0,
    "kilometre": 1000.0,
    "kilometer": 1000.0,
    "1000 metre": 1000.0,
}
_REFERENCES = ("bounds", "grid_mapping", "coordinates", "ancillary_variables")
_PACKING = {
    "scale_factor",
    "add_offset",
    "_FillValue",
    "missing_value",
    "dtype",
    "valid_min",
    "valid_max",
    "valid_range",
}


def unit_spec(unit):
    """
    Return the quantity, pySTEPS normalized unit, and conversion multiplier.

    Parameters
    ----------
    unit : str
        Physical units of the input field.

    Returns
    -------
    quantity : str
        One of 'rate', 'depth', or 'reflectivity'.
    normalized_unit : str
        pySTEPS normalized units: 'mm/h', 'mm', or 'dBZ'. These may differ
        from the CF standard-name table's canonical units.
    multiplier : float
        Factor multiplying input values to obtain pySTEPS normalized values.
        Precipitation mass flux in kg m-2 s-1 uses a reference liquid-water
        density of 1000 kg m-3, giving a multiplier of 3600 to mm/h.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.
    """
    if not isinstance(unit, str):
        raise ValueError("A physical units string is required")
    unit = " ".join(unit.strip().split())
    if unit in _RATE_UNITS:
        return "rate", "mm/h", _RATE_UNITS[unit]
    if unit in _MASS_FLUX_UNITS:
        return "rate", "mm/h", 1000.0 * 3600.0 / _LIQUID_WATER_DENSITY_KG_M3
    if unit in _DEPTH_UNITS:
        return "depth", "mm", _DEPTH_UNITS[unit]
    if unit == "dBZ":
        return "reflectivity", "dBZ", 1.0
    raise ValueError(f"Unsupported physical units {unit!r}")


def variable_metadata(unit, method=None):
    """
    Return a default variable name and physical CF attributes.

    Parameters
    ----------
    unit : str
        Supported physical units.
    method : str, optional
        Temporal method used to describe the rate. Defaults to 'sum' for depth
        and 'point' otherwise.

    Returns
    -------
    name : str
        Suggested name; variable names do not determine physical meaning.
    attrs : dict
        pySTEPS normalized units, standard_name, and long_name attributes.
    """
    quantity, normalized_unit, _ = unit_spec(unit)
    method = method or ("sum" if quantity == "depth" else "point")
    if quantity == "rate":
        name = "precip_intensity"
        long_name = (
            "mean precipitation rate"
            if method == "mean"
            else "instantaneous precipitation rate"
        )
    elif quantity == "depth":
        name, long_name = "precip_accum", "accumulated precipitation depth"
    else:
        name, long_name = "reflectivity", "equivalent reflectivity factor"
    attrs = {
        "units": normalized_unit,
        "standard_name": STANDARD_NAMES[quantity],
        "long_name": long_name,
    }
    if " ".join(unit.strip().split()) in _MASS_FLUX_UNITS:
        attrs.update(
            pysteps_source_units=unit,
            pysteps_liquid_water_density_kg_m3=_LIQUID_WATER_DENSITY_KG_M3,
        )
    return name, attrs


def linked(variable, key):
    """
    Return a CF link from attributes or decoded encoding.

    Parameters
    ----------
    variable : xarray.DataArray or xarray.Variable
        Variable whose metadata contain the link.
    key : str
        Link attribute, such as 'bounds' or 'grid_mapping'.

    Returns
    -------
    out : str or None
        Link value, preferring attributes over encoding, or None if absent.
    """
    return variable.attrs.get(key, variable.encoding.get(key))


def clear_packing(variable):
    """
    Remove source packing metadata in place after changing values.

    Parameters
    ----------
    variable : xarray.DataArray
        Variable whose attributes and encoding are updated.
    """
    variable.encoding = {
        k: v for k, v in variable.encoding.items() if k not in _PACKING
    }
    for key in _PACKING:
        variable.attrs.pop(key, None)


def temporal_axis(dataset, variable):
    """
    Return the unique time dimension of a physical variable.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str
        Name of the physical variable in the dataset.

    Returns
    -------
    out : str
        Dimension identified by its time standard_name or its name.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.
    """
    candidates = [
        dim
        for dim in dataset[variable].dims
        if dim in dataset.coords
        and (dataset[dim].attrs.get("standard_name") == "time" or dim == "time")
    ]
    if len(candidates) != 1:
        raise ValueError(f"{variable!r} requires one identifiable time coordinate")
    return candidates[0]


def temporal_method(dataset, variable, time=None):
    """
    Return the supported temporal cell method for a physical variable.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str
        Name of the physical variable in the dataset.
    time : str, optional
        Time dimension name. Infer it from the variable, falling back to 'time'.

    Returns
    -------
    out : str
        Temporal method: 'point', 'mean', or 'sum'.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.

    Notes
    -----
    Absent cell_methods imply 'sum' for depth and 'point' otherwise.
    """
    quantity = unit_spec(dataset[variable].attrs["units"])[0]
    try:
        time = time or temporal_axis(dataset, variable)
    except ValueError:
        time = "time"
    methods = re.findall(
        rf"\b{re.escape(time)}\s*:\s*(\w+)",
        dataset[variable].attrs.get("cell_methods", ""),
    )
    if len(methods) > 1:
        raise ValueError("Multiple temporal operations are not supported")
    method = methods[0] if methods else ("sum" if quantity == "depth" else "point")
    allowed = {"rate": {"point", "mean"}, "depth": {"sum"}, "reflectivity": {"point"}}
    if method not in allowed[quantity]:
        raise ValueError(f"Unsupported temporal method {method!r} for {quantity}")
    return method


def utc_datetime(value):
    """
    Convert a time value to a timezone-free UTC datetime64.

    Parameters
    ----------
    value : datetime or datetime64 or str
        Time value. Convert timezone-aware datetimes to UTC before removing the
        timezone.

    Returns
    -------
    out : numpy.datetime64
        Time with microsecond resolution.

    Raises
    ------
    ValueError
        If the converted time is NaT.
    """
    if isinstance(value, datetime) and value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    result = np.datetime64(value, "us")
    if np.isnat(result):
        raise ValueError("Missing valid time")
    return result


def checked_times(coordinate):
    """
    Validate and return strictly increasing microsecond timestamps.

    Parameters
    ----------
    coordinate : xarray.DataArray
        Nonempty one-dimensional datetime64 coordinate using a supported
        Gregorian calendar.

    Returns
    -------
    out : numpy.ndarray
        Time values with dtype datetime64[us], preserving input precision.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.
    """
    values = np.asarray(coordinate.values)
    if (
        values.ndim != 1
        or not len(values)
        or not np.issubdtype(values.dtype, np.datetime64)
    ):
        raise ValueError(
            "Time must be a nonempty one-dimensional datetime64 coordinate"
        )
    if np.isnat(values).any():
        raise ValueError("Time contains NaT")
    if len(values) > 1 and np.any(np.diff(values) <= np.timedelta64(0, "us")):
        raise ValueError("Time coordinates must be strictly increasing")
    calendar = coordinate.encoding.get("calendar", coordinate.attrs.get("calendar"))
    if calendar not in (None, "standard", "gregorian", "proleptic_gregorian"):
        raise ValueError(f"Unsupported calendar {calendar!r}")
    normalized = values.astype("datetime64[us]")
    if not np.array_equal(normalized.astype(values.dtype), values):
        raise ValueError("Time precision finer than microseconds is not supported")
    if calendar in ("standard", "gregorian") and normalized.min() < np.datetime64(
        "1582-10-15"
    ):
        raise ValueError("Historical mixed-calendar dates need an explicit extension")
    return normalized


def cadence_seconds(coordinate, *, require=False):
    """
    Return the regular sampling cadence in seconds, when available.

    Parameters
    ----------
    coordinate : xarray.DataArray
        Time coordinate; a singleton may declare stepsize in seconds.
    require : bool, optional
        Raise an error if no regular cadence can be determined. Default is
        False.

    Returns
    -------
    out : float or None
        Cadence in seconds, or None for irregular or unspecified singleton
        sampling.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.
    """
    values = checked_times(coordinate)
    cached = coordinate.attrs.get("stepsize")
    delta = np.diff(values) / np.timedelta64(1, "s")
    regular = len(delta) and np.allclose(delta, delta[0], rtol=1e-9, atol=1e-6)
    if cached is not None and (
        not np.isscalar(cached) or not np.isfinite(cached) or cached <= 0
    ):
        raise ValueError("Time stepsize must be positive seconds")
    if (
        len(delta)
        and cached is not None
        and (not regular or not np.isclose(cached, delta[0], rtol=1e-9, atol=1e-6))
    ):
        raise ValueError("Time stepsize conflicts with the actual time coordinate")
    result = (
        float(delta[0])
        if regular
        else (float(cached) if not len(delta) and cached is not None else None)
    )
    if require and result is None:
        raise ValueError("A regular cadence or explicit singleton stepsize is required")
    return result


def time_bounds(dataset, variable, *, legacy=False, validate_values=True):
    """
    Return validated interval bounds without inferring them from cadence.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str
        Name of the physical variable in the dataset.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.
    validate_values : bool, optional
        Check that a zero-duration initial cumulative field contains only zero
        or missing values. Default is True.

    Returns
    -------
    out : numpy.ndarray or None
        Decoded datetime64[us] bounds of shape (time, 2), or None if absent.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.

    Notes
    -----
    Only legacy=True allows accutime in minutes to supply absent bounds. Existing
    bounds take precedence and must agree with any accutime cache.
    """
    time = temporal_axis(dataset, variable)
    values = checked_times(dataset[time])
    name = linked(dataset[time], "bounds")
    if name is None:
        # Legacy duration is in minutes; cadence does not define bounds.
        duration = dataset[variable].attrs.get("accutime")
        if not legacy or duration is None:
            return None
        if not np.isscalar(duration) or not np.isfinite(duration) or duration <= 0:
            raise ValueError("Legacy accutime must be positive minutes")
        duration = np.timedelta64(int(round(float(duration) * 60e6)), "us")
        return np.column_stack([values - duration, values])
    if name not in dataset:
        raise ValueError(f"Dangling time bounds reference {name!r}")
    bound = dataset[name]
    if bound.ndim != 2 or bound.dims[0] != time or bound.shape != (len(values), 2):
        raise ValueError("Time bounds must have shape (time, 2)")
    if not np.issubdtype(bound.dtype, np.datetime64):
        raise ValueError("Time bounds must be decoded datetime64 values")
    original = bound.values
    bounds = original.astype("datetime64[us]")
    if not np.array_equal(bounds.astype(original.dtype), original):
        raise ValueError("Time bounds finer than microseconds are not supported")
    if np.isnat(bounds).any() or np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError("Time bounds must be finite and ordered")
    if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
        raise ValueError("Valid time must lie within its bounds")
    duration = (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s")
    zero = duration == 0
    if zero.any():
        data = (
            dataset[variable].isel({time: np.flatnonzero(zero)}).values
            if validate_values
            else None
        )
        if (
            zero.sum() != 1
            or not zero[0]
            or (validate_values and np.any(np.isfinite(data) & (data != 0)))
        ):
            raise ValueError(
                "Only an initial zero cumulative amount can have zero duration"
            )
    cached = dataset[variable].attrs.get("accutime")
    if cached is not None and not np.allclose(
        duration, float(cached) * 60, rtol=1e-9, atol=1e-6
    ):
        raise ValueError("accutime conflicts with interval bounds")
    return bounds


def interval_durations(dataset, variable, *, legacy=False):
    """
    Return positive interval durations in seconds.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str
        Name of the physical variable in the dataset.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.

    Returns
    -------
    out : xarray.DataArray or float
        Durations aligned with the time coordinate, or a scalar for legacy
        fields without a time axis.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.
    """
    try:
        time = temporal_axis(dataset, variable)
    except ValueError:
        duration = dataset[variable].attrs.get("accutime")
        if not legacy or duration is None or not np.isfinite(duration) or duration <= 0:
            raise ValueError("Explicit interval bounds are required")
        return float(duration) * 60
    bounds = time_bounds(dataset, variable, legacy=legacy)
    if bounds is None:
        raise ValueError(
            "Explicit interval bounds are required; cadence is not duration"
        )
    durations = (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s")
    if np.any(durations <= 0):
        raise ValueError(
            "Cannot convert a zero-duration cumulative initial value to a rate"
        )
    return xr.DataArray(durations, dims=time, coords={time: dataset[time]})


def is_cumulative(bounds):
    """
    Check whether multiple intervals share a common starting time.

    Parameters
    ----------
    bounds : numpy.ndarray
        Interval bounds of shape (time, 2).

    Returns
    -------
    out : bool
        True if at least two intervals have the same origin.
    """
    return len(bounds) > 1 and np.all(bounds[:, 0] == bounds[0, 0])


def set_time_metadata(dataset, variable, bounds=None, method=None):
    """
    Update time links, serialization, and duration caches.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str
        Name of the physical variable in the dataset.
    bounds : numpy.ndarray, optional
        Decoded interval bounds of shape (time, 2). Omit to retain existing
        bounds.
    method : str, optional
        Temporal cell method to replace or append.

    Returns
    -------
    out : xarray.Dataset
        Dataset with updated values and metadata.

    Notes
    -----
    Updates metadata on the supplied dataset and returns it. Cadence is stored in
    seconds and constant positive accutime in minutes. No intervals are inferred
    from sampling cadence.
    """
    time = temporal_axis(dataset, variable)
    attrs = dataset[time].attrs.copy()
    attrs.update(standard_name="time", axis="T")
    if dataset.sizes[time] > 1:
        attrs.pop("stepsize", None)
    dataset[time].attrs = attrs
    cadence = cadence_seconds(dataset[time])
    if cadence is not None:
        dataset[time].attrs["stepsize"] = cadence
    origin = checked_times(dataset[time])[0]
    existing_bounds = linked(dataset[time], "bounds")
    if bounds is None and existing_bounds in dataset:
        origin = min(origin, dataset[existing_bounds].values.min())
    if bounds is not None:
        origin = min(origin, np.asarray(bounds).min())
    encoding = {
        "units": "seconds since " + str(origin).replace("T", " "),
        "calendar": "proleptic_gregorian",
        "dtype": "float64",
        "_FillValue": None,
    }
    dataset[time].encoding = encoding.copy()
    if bounds is not None:
        name = linked(dataset[time], "bounds") or time + "_bnds"
        dataset[name] = ((time, "nv"), np.asarray(bounds, dtype="datetime64[us]"))
        dataset = dataset.set_coords(name)
        dataset[name].encoding = encoding.copy()
        dataset[time].attrs["bounds"] = name
        dataset[time].encoding.pop("bounds", None)
        duration = (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s")
        if (
            len(duration)
            and duration[0] > 0
            and np.allclose(duration, duration[0], rtol=1e-9, atol=1e-6)
        ):
            dataset[variable].attrs["accutime"] = float(duration[0]) / 60
        else:
            dataset[variable].attrs.pop("accutime", None)
    if method:
        previous = dataset[variable].attrs.get("cell_methods", "")
        previous = re.sub(rf"\b{re.escape(time)}\s*:\s*\w+", "", previous).strip()
        dataset[variable].attrs["cell_methods"] = " ".join(
            filter(None, [previous, f"{time}: {method}"])
        )
    return dataset


def cf_grid_mapping(projection, *, projected=True):
    """
    Build complete CF grid mapping attributes from a CRS.

    Parameters
    ----------
    projection : str or pyproj.CRS
        Any CRS definition accepted by pyproj.CRS.from_user_input.
    projected : bool, optional
        Require a projected CRS. If False, also allow geographic CRS
        definitions. Default is True.

    Returns
    -------
    out : dict
        CF projection attributes including WKT and earth-model information.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.
    """
    crs = pyproj.CRS.from_user_input(projection)
    source = crs.source_crs if crs.is_bound else crs
    if not source.is_projected and (projected or not source.is_geographic):
        raise ValueError("Only projected rectilinear grids are supported")
    attrs = crs.to_cf()
    if "grid_mapping_name" not in attrs:
        raise ValueError("CRS has no supported CF grid mapping")
    if (
        attrs["grid_mapping_name"] == "polar_stereographic"
        and "latitude_of_projection_origin" not in attrs
    ):
        parallel = attrs.get("standard_parallel")
        if parallel is None or parallel == 0:
            raise ValueError("Cannot identify the polar stereographic origin")
        attrs["latitude_of_projection_origin"] = float(np.copysign(90, parallel))
    return {"long_name": "coordinate reference system", **attrs}


def _check_mapping_attrs(attrs, crs):
    """Check supplied CF projection attributes against the declared CRS."""
    expected = cf_grid_mapping(crs)
    for key, value in attrs.items():
        if key not in expected or key in ("crs_wkt", "long_name"):
            continue
        target = expected[key]
        if isinstance(target, (float, int, tuple, list)):
            if not np.allclose(value, target, rtol=1e-9, atol=1e-6):
                raise ValueError(f"CRS definition conflicts with {key}")
        elif key == "grid_mapping_name" and target != value:
            raise ValueError("CRS definition conflicts with grid_mapping_name")
        elif (
            key == "horizontal_datum_name"
            and str(value).lower() != "unknown"
            and str(target).lower() != "unknown"
            and target != value
        ):
            raise ValueError("CRS WKT conflicts with horizontal_datum_name")


def _mapping_crs(attrs):
    """Reconstruct a CRS and reject conflicting CF or WKT definitions."""
    if (
        "crs_wkt" not in attrs
        and "earth_radius" not in attrs
        and not (
            "semi_major_axis" in attrs
            and ("semi_minor_axis" in attrs or "inverse_flattening" in attrs)
        )
    ):
        raise ValueError(
            "CRS earth model is missing; supply full WKT or explicit ellipsoid/sphere metadata"
        )
    crs = pyproj.CRS.from_cf(attrs)
    if "crs_wkt" in attrs:
        _check_mapping_attrs(attrs, crs)
    return crs


def dataset_crs(dataset, variable, *, legacy=False):
    """
    Return the CRS of a physical variable after checking its definition.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str
        Name of the physical variable in the dataset.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.

    Returns
    -------
    out : pyproj.CRS
        Validated CRS from the CF grid mapping or explicit legacy projection.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.
    """
    name = linked(dataset[variable], "grid_mapping")
    projection = dataset.attrs.get("projection")
    if name:
        if name not in dataset or dataset[name].ndim != 0:
            raise ValueError("A scalar grid_mapping variable is required")
        if legacy and projection and "crs_wkt" not in dataset[name].attrs:
            # A declared legacy CRS can complete an old partial mapping. Check
            # every supplied parameter instead of first defaulting its datum.
            crs = pyproj.CRS.from_user_input(projection)
            _check_mapping_attrs(dataset[name].attrs, crs)
        else:
            crs = _mapping_crs(dataset[name].attrs)
        if projection and not crs.equals(
            pyproj.CRS.from_user_input(projection), ignore_axis_order=True
        ):
            raise ValueError("Legacy projection conflicts with the grid mapping")
    elif legacy and projection:
        crs = pyproj.CRS.from_user_input(projection)
    else:
        raise ValueError("A CF grid_mapping is required")
    cf_grid_mapping(crs)
    return crs


def rename_with_links(dataset, mapping):
    """
    Rename variables and update CF links and cell methods.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    mapping : dict
        Mapping from existing variable or dimension names to new names.

    Returns
    -------
    out : xarray.Dataset
        Dataset with updated values and metadata.
    """
    dataset = dataset.rename(mapping)
    for name in dataset.variables:
        for store in (dataset[name].attrs, dataset[name].encoding):
            for key in _REFERENCES:
                if isinstance(store.get(key), str):
                    store[key] = " ".join(
                        mapping.get(token, token) for token in store[key].split()
                    )
        text = dataset[name].attrs.get("cell_methods")
        if text:
            for old, new in mapping.items():
                text = re.sub(rf"\b{re.escape(old)}(?=\s*:)", new, text)
            dataset[name].attrs["cell_methods"] = text
    if dataset.attrs.get("precip_var") in mapping:
        dataset.attrs["precip_var"] = mapping[dataset.attrs["precip_var"]]
    return dataset


def physical_variables(dataset, *, legacy=False):
    """
    List supported physical fields using their CF standard names.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.

    Returns
    -------
    out : list of str
        Physical data variable names in dataset order; legacy mode also
        recognizes documented historical variable names.
    """
    candidates = []
    for name, variable in dataset.data_vars.items():
        standard = variable.attrs.get("standard_name")
        if standard in (*STANDARD_NAMES.values(), "precipitation_flux") or (
            legacy
            and name
            in (
                "precip_intensity",
                "precip_mean_rate",
                "precip_accum",
                "precip_cumulative",
                "reflectivity",
            )
        ):
            candidates.append(name)
    return candidates


def select_variable(dataset, variable=None, *, legacy=False):
    """
    Select an explicitly named or unambiguous physical field.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str, optional
        Selected name. Default to dataset.attrs["precip_var"], then the sole
        supported physical field.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.

    Returns
    -------
    out : str
        Selected physical variable name.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.
    """
    candidates = physical_variables(dataset, legacy=legacy)
    variable = variable or dataset.attrs.get("precip_var")
    if variable is not None:
        if variable not in dataset.data_vars or variable not in candidates:
            raise ValueError(f"Invalid precipitation selector {variable!r}")
        return variable
    if len(candidates) != 1:
        raise ValueError(
            "Select a precipitation variable explicitly; input is ambiguous"
        )
    return candidates[0]


def replace_time_series(
    dataset, variable, values, times, bounds=None, *, related=None, keep_other=True
):
    """
    Replace a time series while preserving other fields on their own axis.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str
        Name of the physical variable in the dataset.
    values : xarray.DataArray or numpy.ndarray
        Replacement physical values with the new time-axis length.
    times : array_like
        Replacement valid times, converted to datetime64[us].
    bounds : numpy.ndarray, optional
        Replacement interval bounds of shape (time, 2).
    related : dict, optional
        Replacement ancillary DataArrays keyed by variable name.
    keep_other : bool, optional
        Retain unprocessed time series on an auxiliary axis. Default is True.

    Returns
    -------
    out : xarray.Dataset
        Dataset with updated values and metadata.

    Notes
    -----
    Replace values positionally to avoid alignment with old timestamps. Remove stale
    packing and links to ancillary time series that were not replaced.
    """
    time = temporal_axis(dataset, variable)
    related = related or {}
    dependencies = set((linked(dataset[variable], "ancillary_variables") or "").split())
    dependencies.update({"quality", "velocity_x", "velocity_y", "velocity_quality"})
    drops = {variable, *dependencies}
    base = dataset.drop_vars([name for name in drops if name in dataset])
    other = [
        name
        for name in base.data_vars
        if time in base[name].dims and name != linked(dataset[time], "bounds")
    ]
    if keep_other and other:
        auxiliary = time + "_aux"
        while auxiliary in base:
            auxiliary += "_aux"
        mapping = {time: auxiliary}
        old_bounds = linked(dataset[time], "bounds")
        if old_bounds:
            mapping[old_bounds] = auxiliary + "_bnds"
        base = rename_with_links(base, mapping)
    else:
        base = base.drop_dims(time)
        if not keep_other:
            for name in physical_variables(base, legacy=True):
                base = base.drop_vars(name)
            for name in list(base.coords):
                if name not in ("x", "y", "lat", "lon", "crs") and "time" in name:
                    base = base.drop_vars(name)
    attrs = dataset[time].attrs.copy()
    for key in ("bounds", "stepsize"):
        attrs.pop(key, None)
    base = base.assign_coords(
        {time: (time, np.asarray(times, dtype="datetime64[us]"), attrs)}
    )
    base[time].encoding = {
        k: v
        for k, v in dataset[time].encoding.items()
        if k not in ("bounds", "original_shape", "source")
    }
    if isinstance(values, xr.DataArray):
        # The new coordinate is authoritative; never align by old timestamps.
        base[variable] = (values.dims, values.data, dataset[variable].attrs.copy())
    else:
        base[variable] = (
            dataset[variable].dims,
            values,
            dataset[variable].attrs.copy(),
        )
    for name, data in related.items():
        base[name] = (data.dims, data.data, data.attrs.copy())
    links = (base[variable].attrs.get("ancillary_variables") or "").split()
    links = [
        name
        for name in links
        if name in related or name in base and time not in base[name].dims
    ]
    if links:
        base[variable].attrs["ancillary_variables"] = " ".join(links)
    else:
        base[variable].attrs.pop("ancillary_variables", None)
    clear_packing(base[variable])
    base[variable].attrs.pop("accutime", None)
    return set_time_metadata(base, variable, bounds)


def cumulative_to_intervals(dataset, variable=None):
    """
    Difference cumulative depths using explicit common-origin bounds.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str, optional
        Physical depth variable. Default to the active or sole physical field.

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
    Return a shallow copy unchanged when bounds already describe nonoverlapping
    intervals. Reject decreasing totals, resets, and unsupported overlaps.
    """
    dataset = dataset.copy(deep=False)
    variable = variable or dataset.attrs.get("precip_var") or select_variable(dataset)
    if unit_spec(dataset[variable].attrs["units"])[0] != "depth":
        raise ValueError("Cumulative differencing requires rainfall depths")
    axis = temporal_axis(dataset, variable)
    bounds = time_bounds(dataset, variable)
    if bounds is None:
        raise ValueError("Cumulative differencing requires explicit bounds")
    if not is_cumulative(bounds):
        if len(bounds) > 1 and np.any(bounds[1:, 0] < bounds[:-1, 1]):
            raise ValueError(
                "Accumulation resets or overlapping intervals need an explicit policy"
            )
        return dataset
    amounts = dataset[variable] - dataset[variable].shift({axis: 1}, fill_value=0)
    if np.any(amounts.values < 0):
        raise ValueError("Cumulative amounts decrease; resolve resets explicitly")
    intervals = bounds.copy()
    intervals[1:, 0] = bounds[:-1, 1]
    dataset[variable] = amounts.assign_attrs(dataset[variable].attrs)
    dataset[variable].attrs.pop("accutime", None)
    dataset = set_time_metadata(dataset, variable, intervals, "sum")
    clear_packing(dataset[variable])
    dataset[variable].attrs["long_name"] = "interval accumulated precipitation depth"
    dataset.attrs["history"] = (
        dataset.attrs.get("history", "")
        + "\nDifferenced cumulative rainfall using explicit bounds"
    ).strip()
    return dataset


def normalize_units(dataset, variable):
    """
    Normalize physical values and metadata to pySTEPS units.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str
        Name of the physical variable in the dataset.

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
    Update the supplied dataset without changing variable names or temporal support.
    Scale thresholds and dry values consistently and discard stale source packing.
    Precipitation mass flux is converted to liquid-water-equivalent rate using
    1000 kg m-3. Source units/standard name and this density are recorded as
    provenance. This conversion does not identify the precipitation as rain.
    """
    attrs = dataset[variable].attrs.copy()
    quantity, normalized_unit, scale = unit_spec(attrs.get("units"))
    standard = attrs.get("standard_name")
    mass_flux = " ".join(attrs["units"].strip().split()) in _MASS_FLUX_UNITS
    expected_standard = "precipitation_flux" if mass_flux else STANDARD_NAMES[quantity]
    if standard is not None and standard != expected_standard:
        raise ValueError(
            f"{variable!r} standard_name conflicts with its units: "
            f"standard_name={standard!r}, units={attrs['units']!r}; "
            f"expected standard_name={expected_standard!r}. "
            "Correct the source standard name or units; a change of physical "
            "quantity is not inferred."
        )
    transform = attrs.get("transform")
    if transform is not None and not (quantity == "reflectivity" and transform == "dB"):
        raise ValueError(
            "Physical metadata normalization requires an untransformed field"
        )
    attrs.pop("transform", None)
    if mass_flux:
        attrs.setdefault("pysteps_source_units", attrs["units"])
        if standard is not None:
            attrs.setdefault("pysteps_source_standard_name", standard)
        if "long_name" in attrs:
            attrs.setdefault("pysteps_source_long_name", attrs["long_name"])
        attrs["long_name"] = "liquid water equivalent precipitation rate"
        attrs["pysteps_liquid_water_density_kg_m3"] = _LIQUID_WATER_DENSITY_KG_M3
        dataset.attrs["history"] = (
            dataset.attrs.get("history", "")
            + f"\nConverted precipitation mass flux in {variable!r} from "
            f"{attrs['units']} to mm/h using reference liquid-water density "
            f"{_LIQUID_WATER_DENSITY_KG_M3:g} kg m-3"
        ).strip()
    if scale != 1:
        dataset[variable] = dataset[variable] * scale
        for key in ("threshold", "zerovalue", "pysteps_no_detection_value"):
            if key in attrs:
                attrs[key] = np.asarray(attrs[key]) * scale
                if attrs[key].ndim == 0:
                    attrs[key] = attrs[key].item()
        clear_packing(dataset[variable])
        for key in _PACKING:
            if key in attrs:
                if key.startswith("valid_"):
                    attrs[key] = np.asarray(attrs[key]) * scale
                else:
                    attrs.pop(key)
    attrs["units"] = normalized_unit
    attrs["standard_name"] = STANDARD_NAMES[quantity]
    dataset[variable].attrs = attrs
    return dataset


def normalize_dataset(
    dataset, variable=None, *, legacy=False, require_model=False, validate_values=True
):
    """
    Return a validated copy of a decoded physical CF dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    variable : str, optional
        Active physical field; use the dataset selector or sole supported field
        by default.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.
    require_model : bool, optional
        Also require metadata needed for modelling, including thresholds and dry
        values. Default is False.
    validate_values : bool, optional
        Scan physical values as well as metadata. Set False to defer these scans
        for lazy imports. Default is True.

    Returns
    -------
    out : xarray.Dataset
        Shallow copy with normalized units, coordinates, CF links, and active
        variable metadata.

    Raises
    ------
    ValueError
        If metadata are missing, inconsistent, or unsupported.

    Notes
    -----
    This function performs no file I/O, packing decoding, or regridding. It
    preserves missing values and requires explicit physical and temporal
    definitions.
    """
    dataset = dataset.copy(deep=False)
    active = select_variable(dataset, variable, legacy=legacy)
    variables = physical_variables(dataset, legacy=legacy)
    for target, standard in (
        ("x", "projection_x_coordinate"),
        ("y", "projection_y_coordinate"),
        ("ens_number", "realization"),
    ):
        if target not in dataset[active].dims:
            candidates = [
                dim
                for dim in dataset[active].dims
                if dim in dataset.coords
                and dataset[dim].attrs.get("standard_name") == standard
            ]
            if len(candidates) == 1:
                if target in dataset:
                    raise ValueError(f"Coordinate name {target!r} is already in use")
                dataset = rename_with_links(dataset, {candidates[0]: target})
    time = temporal_axis(dataset, active)
    if time != "time":
        if "time" in dataset:
            auxiliary = "time_aux"
            while auxiliary in dataset:
                auxiliary += "_aux"
            dataset = rename_with_links(dataset, {"time": auxiliary})
        dataset = rename_with_links(dataset, {time: "time"})
    dataset.attrs["precip_var"] = active
    crs = dataset_crs(dataset, active, legacy=legacy)
    source_crs = crs.source_crs if crs.is_bound else crs
    if any(
        not np.isclose(axis.unit_conversion_factor, 1) for axis in source_crs.axis_info
    ):
        # Express the same projection in metres while retaining the original
        # geographic datum/WKT rather than reconstructing a guessed datum.
        definition = source_crs.to_json_dict()
        for axis in definition["coordinate_system"]["axis"]:
            axis["unit"] = "metre"
        parameters = {
            item.name: item for item in source_crs.coordinate_operation.params
        }
        for item in definition["conversion"]["parameters"]:
            parameter = parameters[item["name"]]
            if parameter.unit_category == "linear":
                item["value"] *= parameter.unit_conversion_factor
                item["unit"] = "metre"
        metric = pyproj.CRS.from_json_dict(definition)
        if crs.is_bound:
            definition = crs.to_json_dict()
            definition["source_crs"] = metric.to_json_dict()
            metric = pyproj.CRS.from_json_dict(definition)
        crs = metric
    mapping_name = linked(dataset[active], "grid_mapping") or "crs"
    for axis, standard in (
        ("x", "projection_x_coordinate"),
        ("y", "projection_y_coordinate"),
    ):
        if axis not in dataset.coords or dataset[axis].dims != (axis,):
            raise ValueError(
                f"A one-dimensional {axis} pixel-centre coordinate is required"
            )
        coord = dataset[axis]
        if coord.attrs.get("standard_name") not in (None, standard):
            raise ValueError(f"Invalid standard_name on {axis}")
        try:
            scale = _LENGTH_UNITS[coord.attrs["units"]]
        except KeyError as error:
            raise ValueError(f"Unsupported {axis} coordinate units") from error
        values = np.asarray(coord.values, dtype=float) * scale
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Grid coordinates must be finite and nonempty")
        delta = np.diff(values)
        if len(delta):
            if (
                delta[0] == 0
                or (axis == "x" and delta[0] < 0)
                or not np.allclose(delta, delta[0], rtol=1e-7, atol=1e-6)
            ):
                raise ValueError(
                    "Grid must be regular, with increasing x and monotonic y"
                )
            step = float(delta[0])
        else:
            step = coord.attrs.get("stepsize")
            bound_name = linked(coord, "bounds")
            if step is None and bound_name in dataset:
                limits = dataset[bound_name].values
                if limits.shape != (1, 2):
                    raise ValueError(
                        "Singleton coordinate bounds must have shape (1, 2)"
                    )
                step = float(limits[0, 1] - limits[0, 0])
            if step is None:
                raise ValueError(
                    "Singleton grid coordinates need explicit spacing or bounds"
                )
            step = float(step) * scale
        if step == 0 or (axis == "x" and step < 0):
            raise ValueError("Invalid grid spacing")
        cached = coord.attrs.get("stepsize")
        if cached is not None and not np.isclose(
            float(cached) * scale, step, rtol=1e-7, atol=1e-6
        ):
            raise ValueError("Grid stepsize conflicts with coordinate spacing")
        attrs = {
            **coord.attrs,
            "units": "m",
            "standard_name": standard,
            "axis": axis.upper(),
            "stepsize": step,
        }
        dataset = dataset.assign_coords({axis: (axis, values, attrs)})
        dataset[axis].encoding = {"dtype": "float64", "_FillValue": None}
        bound_name = linked(coord, "bounds")
        if bound_name:
            if bound_name not in dataset:
                raise ValueError("Dangling grid bounds reference")
            limits = np.asarray(dataset[bound_name].values, dtype=float) * scale
            tolerance = max(1e-6, abs(step) * 1e-7)
            if (
                limits.shape != (len(values), 2)
                or not np.isfinite(limits).all()
                or not np.allclose(limits.mean(axis=1), values, rtol=0, atol=tolerance)
                or not np.allclose(
                    np.abs(np.diff(limits, axis=1)), abs(step), rtol=1e-7, atol=1e-6
                )
            ):
                raise ValueError("Grid bounds conflict with pixel centres or spacing")
            bound_attrs = dataset[bound_name].attrs.copy()
            if "units" in bound_attrs:
                if bound_attrs["units"] != coord.attrs["units"]:
                    raise ValueError("Grid bounds units conflict with their coordinate")
                bound_attrs["units"] = "m"
            dataset[bound_name] = (
                dataset[bound_name].dims,
                limits,
                bound_attrs,
            )
            dataset[bound_name].encoding = {"dtype": "float64", "_FillValue": None}
    for name in variables:
        other_crs = dataset_crs(dataset, name, legacy=legacy)
        original = dataset_crs(dataset, active, legacy=legacy)
        if not other_crs.equals(original, ignore_axis_order=True):
            raise ValueError("Multiple horizontal CRS definitions are not supported")
        old_standard = dataset[name].attrs.get("standard_name")
        old_names = {
            "instantaneous_precipitation_rate": "rate",
            "accumulated_precipitation": "depth",
        }
        if legacy and old_standard in old_names:
            quantity = unit_spec(dataset[name].attrs["units"])[0]
            if quantity != old_names[old_standard]:
                raise ValueError("Legacy standard name conflicts with its units")
            dataset[name].attrs["pysteps_source_standard_name"] = old_standard
            dataset[name].attrs["standard_name"] = STANDARD_NAMES[quantity]
        dataset = normalize_units(dataset, name)
        quantity = unit_spec(dataset[name].attrs["units"])[0]
        axis = temporal_axis(dataset, name)
        allowed_dimensions = {axis, "x", "y"}
        realization = [
            dim
            for dim in dataset[name].dims
            if dim in dataset.coords
            and dataset[dim].attrs.get("standard_name") == "realization"
        ]
        if "ens_number" in dataset[name].dims and "ens_number" not in realization:
            realization.append("ens_number")
        if (
            len(realization) > 1
            or set(dataset[name].dims) - allowed_dimensions - set(realization)
            or not {"x", "y"}.issubset(dataset[name].dims)
        ):
            raise ValueError(
                "Physical fields require a shared x/y grid and optional realization axis"
            )
        for key in ("threshold", "zerovalue"):
            value = dataset[name].attrs.get(key)
            if value is not None and (not np.isscalar(value) or not np.isfinite(value)):
                raise ValueError(
                    f"Supplied {key} must be a finite scalar; use linked fields when it varies"
                )
        checked_times(dataset[axis])
        cadence_seconds(dataset[axis])
        method = temporal_method(dataset, name, axis)
        had_bounds = linked(dataset[axis], "bounds") is not None
        bounds = time_bounds(
            dataset,
            name,
            legacy=legacy and method != "point",
            validate_values=validate_values,
        )
        if not had_bounds and bounds is not None:
            dataset.attrs["history"] = (
                dataset.attrs.get("history", "")
                + "\nLegacy accutime interpreted as interval ending at supplied valid time"
            ).strip()
        if method != "point" and bounds is None:
            raise ValueError(f"{name!r} requires interval bounds")
        if method == "point" and "accutime" in dataset[name].attrs:
            dataset[name].attrs["pysteps_source_accutime_minutes"] = dataset[
                name
            ].attrs.pop("accutime")
        if bounds is not None and method != "point":
            ends = bounds[:, 1]
            if not np.array_equal(dataset[axis].values, ends):
                if any(
                    other != name
                    and axis in dataset[other].dims
                    and temporal_method(dataset, other, axis) == "point"
                    for other in variables
                ):
                    raise ValueError(
                        "Relabelling shared time would move point observations; use separate time axes"
                    )
                source_name = axis + "_source"
                if source_name in dataset:
                    raise ValueError("Source time coordinate already exists")
                source_values = dataset[axis].values.copy()
                dataset = dataset.assign_coords(
                    {
                        source_name: (
                            axis,
                            source_values,
                            {"long_name": "original source time label"},
                        )
                    }
                )
                dataset[source_name].encoding = {
                    "units": "seconds since " + str(source_values[0]).replace("T", " "),
                    "calendar": "proleptic_gregorian",
                    "dtype": "float64",
                    "_FillValue": None,
                }
                dataset = dataset.assign_coords(
                    {axis: (axis, ends, dataset[axis].attrs)}
                )
        dataset = set_time_metadata(
            dataset, name, bounds if method != "point" else None, method
        )
        dataset[name].attrs["grid_mapping"] = mapping_name
        dataset[name].encoding.pop("grid_mapping", None)
        if validate_values:
            data = dataset[name].values
            if np.isinf(data).any() or (
                quantity != "reflectivity" and np.any(data < 0)
            ):
                raise ValueError(f"Invalid physical values in {name!r}")
        if require_model and name == active:
            for key in ("threshold", "zerovalue"):
                if (
                    key not in dataset[name].attrs
                    or not np.isscalar(dataset[name].attrs[key])
                    or not np.isfinite(dataset[name].attrs[key])
                ):
                    raise ValueError(f"Model input requires an explicit scalar {key}")
    dataset[mapping_name] = xr.DataArray(np.int32(0), attrs=cf_grid_mapping(crs))
    dataset = dataset.set_coords(mapping_name)
    dataset.attrs["projection"] = crs.to_wkt()
    geographic = [
        name
        for name in dataset.coords
        if dataset[name].attrs.get("standard_name") in ("latitude", "longitude")
    ]
    if geographic and validate_values:
        xx, yy = np.meshgrid(dataset.x.values, dataset.y.values)
        source = crs.source_crs if crs.is_bound else crs
        lon, lat = pyproj.Transformer.from_crs(
            crs, source.geodetic_crs, always_xy=True
        ).transform(xx, yy)
        for name in geographic:
            coordinate = dataset[name]
            if set(coordinate.dims) - {"x", "y"}:
                raise ValueError(
                    "Geographic coordinates must be spatial, without time/member duplication"
                )
            latitude = coordinate.attrs["standard_name"] == "latitude"
            units = coordinate.attrs.get("units")
            allowed = (
                ("degrees_north", "degree_north", "degree_N", "degrees_N")
                if latitude
                else ("degrees_east", "degree_east", "degree_E", "degrees_E")
            )
            if units not in allowed:
                raise ValueError("Invalid geographic coordinate units")
            expected = lat if latitude else lon
            actual = (
                coordinate.broadcast_like(xr.DataArray(expected, dims=("y", "x")))
                .transpose("y", "x")
                .values
            )
            difference = actual - expected
            if not latitude:
                difference = (difference + 180) % 360 - 180
            if not np.isfinite(actual).all() or not np.allclose(
                difference, 0, atol=1e-6, rtol=0
            ):
                raise ValueError(
                    "Geographic coordinates conflict with projected cell centres and CRS"
                )
    active_dims = dataset[active].dims
    if set(active_dims) not in ({"time", "y", "x"}, {"ens_number", "time", "y", "x"}):
        raise ValueError(
            "Active field must have time, y, x and optional ens_number dimensions"
        )
    dataset[active] = dataset[active].transpose(
        *(["ens_number"] if "ens_number" in active_dims else []), "time", "y", "x"
    )
    if "ens_number" in active_dims:
        labels = dataset.ens_number.values
        if (
            labels.ndim != 1
            or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all()
            or len(np.unique(labels)) != len(labels)
            or np.any(labels != np.floor(labels))
        ):
            raise ValueError("Ensemble identifiers must be unique finite integers")
        limit = np.iinfo(np.int32)
        if np.any(labels < limit.min) or np.any(labels > limit.max):
            raise ValueError(
                "Ensemble identifiers are outside the supported int32 profile"
            )
        attrs = {
            **dataset.ens_number.attrs,
            "standard_name": "realization",
            "units": "1",
        }
        dataset = dataset.assign_coords(
            ens_number=("ens_number", labels.astype("int32"), attrs)
        )
        dataset.ens_number.encoding = {"dtype": "int32", "_FillValue": None}
    if "forecast_reference_time" in dataset:
        coordinate = dataset.forecast_reference_time
        if (
            coordinate.ndim != 0
            or not np.issubdtype(coordinate.dtype, np.datetime64)
            or np.isnat(coordinate.values)
        ):
            raise ValueError(
                "forecast_reference_time must be one finite scalar datetime"
            )
        checked_times(coordinate.expand_dims("_reference_time_check"))
        dataset.forecast_reference_time.attrs["standard_name"] = (
            "forecast_reference_time"
        )
        dataset.forecast_reference_time.encoding = {
            "units": "seconds since " + str(coordinate.values).replace("T", " "),
            "calendar": "proleptic_gregorian",
            "dtype": "float64",
            "_FillValue": None,
        }
    dataset.attrs = {
        key: value for key, value in dataset.attrs.items() if value is not None
    }
    for owner in [dataset, *[dataset[name] for name in dataset.variables]]:
        for key, value in owner.attrs.items():
            if value is None:
                continue
            if (
                not isinstance(
                    value, (str, int, float, np.number, np.ndarray, list, tuple)
                )
                or np.asarray(value).dtype.kind not in "iufUS"
                or np.asarray(value).ndim > 1
            ):
                raise ValueError(
                    f"Unsupported object-valued or nonserializable metadata attribute {key!r}"
                )
    for name in dataset.variables:
        dataset[name].attrs = {
            k: v for k, v in dataset[name].attrs.items() if v is not None
        }
        for key in _REFERENCES:
            value = linked(dataset[name], key)
            if value and any(token not in dataset for token in value.split()):
                raise ValueError(f"Dangling {key} reference on {name!r}")
    return dataset
