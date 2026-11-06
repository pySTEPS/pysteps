"""
pysteps.utils.conversion
========================

Convert between physical rain rate, interval depth, and reflectivity.

Time support is retained when converting depths to interval mean rates. An
instantaneous rate requires an explicit constant-in-time integration rule to
produce a depth. ``legacy=True`` permits an explicit accutime duration for old
in-memory fields lacking CF bounds; cadence is never used as a duration.

.. autosummary::
    :toctree: ../generated/

    to_rainrate
    to_raindepth
    to_reflectivity
"""

import numpy as np
import xarray as xr

from pysteps.dataset_metadata import (
    cadence_seconds,
    clear_packing,
    interval_durations,
    linked,
    normalize_units,
    select_variable,
    set_time_metadata,
    temporal_axis,
    temporal_method,
    time_bounds,
    unit_spec,
    variable_metadata,
)
from . import transformation


def _prepare(dataset):
    """Copy the dataset, invert statistical transforms, and validate physical units."""
    dataset = dataset.copy(deep=True)
    variable = dataset.attrs.get("precip_var") or select_variable(dataset)
    dataset.attrs["precip_var"] = variable
    quantity = unit_spec(dataset[variable].attrs.get("units"))[0]
    marker = dataset[variable].attrs.get("transform")
    if marker is not None and not (quantity == "reflectivity" and marker == "dB"):
        inverse = {
            "dB": transformation.dB_transform,
            "BoxCox": transformation.boxcox_transform,
            "log": transformation.boxcox_transform,
            "sqrt": transformation.sqrt_transform,
            "NQT": transformation.NQ_transform,
        }.get(marker)
        if inverse is None:
            raise ValueError(f"Unknown transformation {marker!r}")
        dataset = inverse(dataset, inverse=True)
    dataset[variable].attrs.pop("transform", None)
    dataset = normalize_units(dataset, variable)
    try:
        time = temporal_axis(dataset, variable)
    except ValueError:
        pass
    else:
        cadence_seconds(dataset[time])
    values = dataset[variable].values
    if np.isinf(values).any() or (quantity != "reflectivity" and np.any(values < 0)):
        raise ValueError("Invalid physical precipitation values")
    return dataset, variable, quantity


def _interval_support(dataset, variable, legacy):
    """Return explicit durations and record legacy interval support when needed."""
    duration = interval_durations(dataset, variable, legacy=legacy)
    if legacy:
        try:
            axis = temporal_axis(dataset, variable)
        except ValueError:
            pass  # Explicit array-only compatibility has no dated support to store.
        else:
            if linked(dataset[axis], "bounds") is None:
                bounds = time_bounds(dataset, variable, legacy=True)
                dataset = set_time_metadata(dataset, variable, bounds)
                dataset.attrs["history"] = (
                    dataset.attrs.get("history", "")
                    + "\nLegacy accutime interpreted as an interval ending at supplied valid time"
                ).strip()
    return dataset, duration


def _coefficients(dataset, variable, zr_a, zr_b):
    """Validate Z-R coefficients and record their values and sources."""
    attrs = dataset[variable].attrs
    a = float(zr_a if zr_a is not None else attrs.get("zr_a", 200.0))
    b = float(zr_b if zr_b is not None else attrs.get("zr_b", 1.6))
    if not np.isfinite([a, b]).all() or a <= 0 or b <= 0:
        raise ValueError("Z-R coefficients must be positive finite numbers")
    sources = [
        "argument" if arg is not None else "metadata" if key in attrs else "default"
        for key, arg in (("zr_a", zr_a), ("zr_b", zr_b))
    ]
    attrs.update(zr_a=a, zr_b=b, zr_source="a:" + sources[0] + ",b:" + sources[1])
    return a, b


def _scale_metadata(dataset, variable, function, unit):
    """Keep varying thresholds as linked fields rather than misleading scalars."""
    attrs = dataset[variable].attrs.copy()
    existing_parameters = [
        name
        for name in (attrs.get("ancillary_variables") or "").split()
        if name in dataset
        and dataset[name].attrs.get("pysteps_parameter") in ("threshold", "zerovalue")
    ]
    for parameter in existing_parameters:
        metadata = dataset[parameter].attrs.copy()
        dataset[parameter] = function(dataset[parameter])
        dataset[parameter].attrs = {**metadata, "units": unit}
        clear_packing(dataset[parameter])
    for key in ("threshold", "zerovalue"):
        if key not in attrs:
            continue
        value = function(attrs[key])
        data = np.asarray(value)
        if data.size and np.allclose(data, data.flat[0], equal_nan=True):
            attrs[key] = float(data.flat[0])
        elif isinstance(value, xr.DataArray):
            name = variable + "_" + key
            if name in dataset:
                raise ValueError(f"Conversion would overwrite {name!r}")
            dataset[name] = value
            dataset[name].attrs = {
                "long_name": key + " for " + variable,
                "units": unit,
                "pysteps_parameter": key,
                "pysteps_parameter_of": variable,
            }
            attrs.pop(key)
            attrs["ancillary_variables"] = " ".join(
                filter(None, [attrs.get("ancillary_variables"), name])
            )
        else:
            raise ValueError(f"Cannot represent varying {key} without a coordinate")
    dataset[variable].attrs = attrs
    return dataset


def _finish(dataset, variable, values, unit, method=None, name=None):
    """Install converted values and refresh quantity, packing, and time metadata."""
    default, metadata = variable_metadata(unit, method)
    target = name or default
    if target != variable and target in dataset:
        raise ValueError(
            f"Conversion would overwrite {target!r}; select another result name"
        )
    attrs = {**dataset[variable].attrs, **metadata}
    if isinstance(values, xr.DataArray):
        dataset[variable] = (values.dims, values.data)
    else:
        dataset[variable] = (dataset[variable].dims, values)
    dataset[variable].attrs = attrs
    clear_packing(dataset[variable])
    if target != variable:
        dataset = dataset.rename_vars({variable: target})
    dataset.attrs["precip_var"] = target
    for auxiliary in (dataset[target].attrs.get("ancillary_variables") or "").split():
        if (
            auxiliary in dataset
            and dataset[auxiliary].attrs.get("pysteps_parameter_of") == variable
        ):
            dataset[auxiliary].attrs["pysteps_parameter_of"] = target
    try:
        bounds = time_bounds(dataset, target)
        dataset = set_time_metadata(dataset, target, bounds, method)
    except ValueError:
        # Legacy array-only conversion has no time axis. Do not suppress errors
        # in a dated field, especially malformed or conflicting bounds.
        if any(
            dim == "time" or dataset[dim].attrs.get("standard_name") == "time"
            for dim in dataset[target].dims
            if dim in dataset.coords
        ):
            raise
    return dataset


def to_rainrate(dataset, zr_a=None, zr_b=None, *, legacy=False, name=None):
    """
    Convert the active physical field to rain rate in mm/h.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    zr_a : float, optional
        Positive coefficient a in Z = a R**b. Use variable metadata, then 200,
        when omitted.
    zr_b : float, optional
        Positive exponent b in Z = a R**b. Use variable metadata, then 1.6, when
        omitted.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.
    name : str, optional
        Output variable name. Default to the canonical name for the target
        quantity; reject collisions with another field.

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
    Retain interval support when converting depth to mean rate. Preserve no-
    detection flags and floors as dry values and record the source of Z-R
    coefficients. Unprocessed physical fields remain in the result.
    """
    dataset, variable, quantity = _prepare(dataset)
    values = dataset[variable]
    method = temporal_method(dataset, variable)
    if quantity == "depth":
        dataset, duration = _interval_support(dataset, variable, legacy)
        factor = 3600 / duration
        dataset = _scale_metadata(dataset, variable, lambda x: x * factor, "mm/h")
        values = values * factor
        method = "mean"
    elif quantity == "reflectivity":
        a, b = _coefficients(dataset, variable, zr_a, zr_b)
        convert = lambda x: (10 ** (x / 10) / a) ** (1 / b)
        values = convert(values)
        attrs = dataset[variable].attrs
        floor = attrs.get(
            "pysteps_no_detection_value", attrs.get("zerovalue") if not legacy else None
        )
        if floor is not None:
            values = values.where(dataset[variable] != floor, 0)
        for auxiliary in (
            linked(dataset[variable], "ancillary_variables") or ""
        ).split():
            if auxiliary in dataset and dataset[auxiliary].attrs.get(
                "flag_meanings", ""
            ).split() == ["not_detected", "detected"]:
                codes = dataset[auxiliary].attrs.get("flag_values")
                if codes is not None and len(codes) == 2:
                    values = values.where(dataset[auxiliary] != codes[0], 0)
        if "threshold" in attrs:
            attrs["threshold"] = float(convert(attrs["threshold"]))
        if "zerovalue" in attrs or floor is not None:
            attrs["zerovalue"] = 0.0
        attrs.pop("pysteps_no_detection_value", None)
        method = "point"
    return _finish(dataset, variable, values, "mm/h", method, name)


def to_raindepth(
    dataset,
    zr_a=None,
    zr_b=None,
    *,
    integration=None,
    duration_seconds=None,
    legacy=False,
    name=None,
):
    """
    Convert the active physical field to precipitation depth in mm.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    zr_a : float, optional
        Positive coefficient a in Z = a R**b. Use variable metadata, then 200,
        when omitted.
    zr_b : float, optional
        Positive exponent b in Z = a R**b. Use variable metadata, then 1.6, when
        omitted.
    integration : {None, 'constant'}, optional
        For point rates, explicitly declare a constant rate over the preceding
        duration. Do not override support for interval mean rates.
    duration_seconds : float, optional
        Positive integration duration for point rates. Never infer this duration
        from sampling cadence.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.
    name : str, optional
        Output variable name. Default to the canonical name for the target
        quantity; reject collisions with another field.

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
    Use explicit interval support for mean rates. For legacy fields, accutime in
    minutes may supply missing durations. Point-rate integration ends at each valid
    time.
    """
    dataset, variable, quantity = _prepare(dataset)
    if quantity == "depth":
        return _finish(dataset, variable, dataset[variable], "mm", "sum", name)
    if quantity == "reflectivity":
        dataset = to_rainrate(dataset, zr_a, zr_b, legacy=legacy)
        variable = dataset.attrs["precip_var"]
    method = temporal_method(dataset, variable)
    if method == "point":
        if integration != "constant":
            raise ValueError(
                "Point rate to depth requires integration='constant' and an explicit duration"
            )
        duration = duration_seconds
        if duration is None and legacy:
            minutes = dataset[variable].attrs.get("accutime")
            duration = None if minutes is None else float(minutes) * 60
        if (
            duration is None
            or not np.isscalar(duration)
            or not np.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(
                "A positive duration_seconds is required for point integration"
            )
        dataset.attrs["history"] = (
            dataset.attrs.get("history", "")
            + f"\nIntegrated an explicitly constant rate over {duration:g} seconds ending at valid time"
        ).strip()
        try:
            time = temporal_axis(dataset, variable)
        except ValueError:
            if not legacy:
                raise
        else:
            ends = dataset[time].values.astype("datetime64[us]")
            bounds = np.column_stack(
                [ends - np.timedelta64(int(round(duration * 1e6)), "us"), ends]
            )
            dataset[variable].attrs.pop("accutime", None)
            dataset = set_time_metadata(dataset, variable, bounds)
    else:
        if integration is not None or duration_seconds is not None:
            raise ValueError(
                "Interval mean rate already has support; do not override its duration"
            )
        dataset, duration = _interval_support(dataset, variable, legacy)
    factor = duration / 3600
    dataset = _scale_metadata(dataset, variable, lambda x: x * factor, "mm")
    values = dataset[variable] * factor
    return _finish(dataset, variable, values, "mm", "sum", name)


def to_reflectivity(dataset, zr_a=None, zr_b=None, *, legacy=False, name=None):
    """
    Convert the active physical field to reflectivity in dBZ.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    zr_a : float, optional
        Positive coefficient a in Z = a R**b. Use variable metadata, then 200,
        when omitted.
    zr_b : float, optional
        Positive exponent b in Z = a R**b. Use variable metadata, then 1.6, when
        omitted.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.
    name : str, optional
        Output variable name. Default to the canonical name for the target
        quantity; reject collisions with another field.

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
    Preserve positive trace values. A positive rain threshold supplies a finite no-
    detection floor for dry zero; otherwise require an explicit dry-value policy.
    Treat physical dBZ separately from statistical log transforms of rain.
    """
    dataset, variable, quantity = _prepare(dataset)
    if quantity == "reflectivity":
        return _finish(dataset, variable, dataset[variable], "dBZ", "point", name)
    if quantity == "depth":
        dataset = to_rainrate(dataset, zr_a, zr_b, legacy=legacy)
        variable = dataset.attrs["precip_var"]
    method = temporal_method(dataset, variable)
    if method == "mean" and not legacy:
        raise ValueError(
            "A Z-R relation applied to interval mean rain does not recover instantaneous reflectivity"
        )
    a, b = _coefficients(dataset, variable, zr_a, zr_b)
    attrs = dataset[variable].attrs
    convert = lambda x: 10 * np.log10(a * x**b)
    threshold = attrs.get("threshold")
    floor = None
    if threshold is not None and threshold > 0:
        attrs["threshold"] = float(convert(threshold))
        floor = attrs["threshold"] - 5
        attrs["zerovalue"] = floor
        attrs["pysteps_no_detection_value"] = floor
        attrs["pysteps_no_detection_source"] = (
            "5 dBZ below converted supplied rain threshold"
        )
    if bool((dataset[variable] == 0).any()) and floor is None:
        raise ValueError(
            "Dry zero requires an explicit positive threshold/no-detection policy for finite dBZ"
        )
    values = convert(dataset[variable].where(dataset[variable] > 0))
    if floor is not None:
        values = values.where(dataset[variable] != 0, floor)
    # Interval-mean pseudo-reflectivity is available only through the explicit
    # compatibility path; retaining its interval prevents an instantaneous label.
    result = _finish(dataset, variable, values, "dBZ", None, name)
    if method == "mean":
        time = (
            temporal_axis(result, result.attrs["precip_var"])
            if "time" in result
            else "time"
        )
        result[result.attrs["precip_var"]].attrs["cell_methods"] = f"{time}: mean"
        result[result.attrs["precip_var"]].attrs[
            "pysteps_zr_mean_assumption"
        ] = "Z-R applied to mean rate"
    return result
