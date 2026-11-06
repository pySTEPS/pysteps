# -*- coding: utf-8 -*-
"""
pysteps.utils.dimension
=======================

Functions to manipulate array dimensions.

.. autosummary::
    :toctree: ../generated/

    aggregate_fields
    aggregate_fields_time
    aggregate_fields_space
    clip_domain
    square_domain
"""

from typing import Any, Callable

import numpy as np
import xarray as xr

from pysteps.xarray_helpers import compute_lat_lon
from pysteps.dataset_metadata import (
    cadence_seconds,
    cumulative_to_intervals,
    linked,
    normalize_units,
    replace_time_series,
    set_time_metadata,
    temporal_method,
    time_bounds,
    unit_spec,
)

_aggregation_methods: dict[str, Callable[..., Any]] = {
    "sum": np.sum,
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
    "nanmean": np.nanmean,
    "nansum": np.nansum,
    "nanmin": np.nanmin,
    "nanmax": np.nanmax,
}


def aggregate_fields_time(
    dataset, time_window_min, ignore_nan=False, *, integration=None, legacy=False
):
    """
    Aggregate contiguous physical intervals into complete time windows.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing decoded physical fields and their CF metadata.
    time_window_min : float or None
        Positive aggregation duration in minutes. Return the input unchanged if
        None.
    ignore_nan : bool, optional
        Allow partial coverage and attach a coverage fraction. Entirely missing
        windows remain missing. Default is False.
    integration : {None, 'constant'}, optional
        Required for point rates: declare each sample constant over the
        preceding sampling interval. Omit for interval products.
    legacy : bool, optional
        Allow explicit legacy names, projection metadata, or accutime durations
        where supported. Default is False.

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
    Sum interval depths and use duration-weighted means for interval rates.
    Difference cumulative totals first. Reject gaps, overlaps, incomplete windows,
    transformed values, and reflectivity.
    """
    if time_window_min is None:
        return dataset
    target = float(time_window_min) * 60
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Aggregation window must be positive minutes")
    dataset = dataset.copy(deep=False)
    variable = dataset.attrs["precip_var"]
    if dataset[variable].attrs.get("transform") is not None:
        raise ValueError(
            "Return the field to physical values before temporal aggregation"
        )
    dataset = normalize_units(dataset, variable)
    quantity = unit_spec(dataset[variable].attrs["units"])[0]
    if quantity not in ("rate", "depth"):
        raise ValueError("Temporal aggregation supports rain rates and depths, not dBZ")
    cadence_seconds(dataset.time)
    method = temporal_method(dataset, variable)
    if method == "point":
        if integration != "constant":
            raise ValueError("Point-rate aggregation requires integration='constant'")
        cadence = cadence_seconds(dataset.time, require=True)
        ends = dataset.time.values.astype("datetime64[us]")
        bounds = np.column_stack(
            [ends - np.timedelta64(int(round(cadence * 1e6)), "us"), ends]
        )
        dataset[variable].attrs.pop("accutime", None)
    else:
        if integration is not None:
            raise ValueError("Do not override the support of an interval product")
        bounds = time_bounds(dataset, variable, legacy=legacy)
        if bounds is None:
            raise ValueError("Temporal aggregation requires explicit interval bounds")
        dataset = set_time_metadata(dataset, variable, bounds, method)
        if quantity == "depth":
            dataset = cumulative_to_intervals(dataset, variable)
            bounds = time_bounds(dataset, variable)
    durations = (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s")
    first = 1 if durations[0] == 0 else 0
    if first == len(durations):
        raise ValueError("No nonzero interval is available for aggregation")
    bounds = bounds[first:]
    durations = durations[first:]
    if len(bounds) > 1 and not np.array_equal(bounds[1:, 0], bounds[:-1, 1]):
        raise ValueError(
            "Aggregation requires contiguous nonoverlapping intervals; resolve gaps/overlap first"
        )
    groups = []
    start, accumulated = 0, 0.0
    for index, duration in enumerate(durations):
        accumulated += duration
        if accumulated > target + 1e-6:
            raise ValueError("Aggregation window would split a source interval")
        if np.isclose(accumulated, target, rtol=0, atol=1e-6):
            groups.append((start + first, index + first + 1))
            start, accumulated = index + 1, 0.0
    if accumulated > 1e-6 or not groups:
        raise ValueError(
            "Aggregation requires complete windows; trailing coverage is incomplete"
        )
    outputs, coverage = [], []
    related_outputs = {}
    links = (linked(dataset[variable], "ancillary_variables") or "").split()
    for auxiliary in links:
        if auxiliary in dataset and "time" in dataset[auxiliary].dims:
            meanings = dataset[auxiliary].attrs.get("flag_meanings", "").split()
            if meanings != ["not_detected", "detected"]:
                raise ValueError(
                    f"No temporal aggregation policy for ancillary {auxiliary!r}"
                )
    related_names = [
        name
        for name in [*links, "quality", "velocity_x", "velocity_y"]
        if name in dataset and "time" in dataset[name].dims
    ]
    all_bounds = (
        np.asarray(time_bounds(dataset, variable))
        if method != "point"
        else np.column_stack(
            [
                dataset.time.values - np.timedelta64(int(round(cadence * 1e6)), "us"),
                dataset.time.values,
            ]
        )
    )
    for begin, end in groups:
        window = dataset[variable].isel(time=slice(begin, end))
        weights = xr.DataArray(
            (all_bounds[begin:end, 1] - all_bounds[begin:end, 0])
            / np.timedelta64(1, "s"),
            dims="time",
            coords={"time": window.time},
        )
        valid = window.notnull()
        valid_duration = weights.where(valid, 0).sum("time")
        if quantity == "rate":
            numerator = (window * weights).sum("time", skipna=ignore_nan)
            denominator = valid_duration if ignore_nan else target
            value = numerator / xr.where(denominator > 0, denominator, np.nan)
        else:
            value = window.sum("time", skipna=ignore_nan).where(valid.any("time"))
        outputs.append(value)
        coverage.append(valid_duration / target)
        for name in related_names:
            auxiliary = dataset[name].isel(time=slice(begin, end))
            if name.startswith("velocity_"):
                reduced = auxiliary.sum("time", skipna=False)
            elif name == "quality":
                reduced = auxiliary.min("time", skipna=ignore_nan)
            else:
                reduced = auxiliary.max("time", skipna=ignore_nan)
            related_outputs.setdefault(name, []).append(reduced)
    result_bounds = np.array(
        [[all_bounds[a, 0], all_bounds[b - 1, 1]] for a, b in groups]
    )
    result_time = result_bounds[:, 1]

    def combine(parts):
        """Stack aggregated intervals in the original physical dimension order."""
        combined = xr.concat(parts, dim=xr.IndexVariable("time", result_time))
        return combined.transpose(
            *[dim for dim in dataset[variable].dims if dim in combined.dims]
        )

    related = {
        name: combine(parts).assign_attrs(dataset[name].attrs)
        for name, parts in related_outputs.items()
    }
    result = replace_time_series(
        dataset, variable, combine(outputs), result_time, result_bounds, related=related
    )
    result = set_time_metadata(
        result, variable, result_bounds, "mean" if quantity == "rate" else "sum"
    )
    if ignore_nan:
        name = variable + "_coverage_fraction"
        if name in result:
            raise ValueError("Coverage variable already exists")
        combined = combine(coverage)
        result[name] = (combined.dims, combined.data)
        result[name].attrs = {
            "long_name": "fraction of aggregation interval with valid data",
            "units": "1",
        }
        attrs = result[variable].attrs
        attrs["ancillary_variables"] = " ".join(
            filter(None, [attrs.get("ancillary_variables"), name])
        )
        attrs["pysteps_missing_policy"] = (
            "ignore missing coverage; see coverage fraction"
        )
    return result


def aggregate_fields_space(
    dataset: xr.Dataset, space_window, ignore_nan=False
) -> xr.Dataset:
    """
    Upscale fields in space.

    It attempts to aggregate the given dataset in y and x direction in an integer
    number of sections of length = ``(window_size_y, window_size_x)``.
    If such a aggregation is not possible, an error is raised.
    The data is aggregated by computing the mean. Only datasets with precipitation
    data in the ``mm`` or ``mm/h`` unit are currently supported.

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset containing a single field or
        a time series of (ensemble) input fields as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    space_window: float, tuple or None
        The length of the space window that is used to upscale the fields.
        If a float is given, the same window size is used for the x- and
        y-directions. Separate window sizes are used for x- and y-directions if
        a two-element tuple is given (y, x). The space_window unit is the same
        as the unit of x and y in the input dataset. The space spanned by the
        n- and m-dimensions of the dataset content must be a multiple of space_window.
        If set to None, the function returns a copy of the original dataset.
    ignore_nan: bool, optional
        If True, ignore nan values.

    Returns
    -------
    dataset: xarray.Dataset
        The new dataset.

    See also
    --------
    pysteps.utils.dimension.aggregate_fields_time,
    pysteps.utils.dimension.aggregate_fields
    """

    if space_window is None:
        return dataset

    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs

    unit = metadata["units"]

    if np.isscalar(space_window):
        space_window = (space_window, space_window)

    ydelta = dataset["y"].attrs["stepsize"]
    xdelta = dataset["x"].attrs["stepsize"]

    if space_window[0] % ydelta > 1e-10 or space_window[1] % xdelta > 1e-10:
        raise ValueError("space_window does not equally split dataset")

    # specify the operator to be used to aggregate the values
    # within the space window
    if unit == "mm/h" or unit == "mm":
        method = "mean"
    else:
        raise ValueError(f"can only aggregate units of 'mm/h' or 'mm' not {unit}")

    if ignore_nan:
        method = "".join(("nan", method))

    window_size = (int(abs(space_window[0] / ydelta)), int(space_window[1] / xdelta))

    return aggregate_fields(dataset, window_size, ["y", "x"], method, "mean")


def aggregate_fields(
    dataset: xr.Dataset,
    window_size,
    dim="x",
    method="mean",
    velocity_method="mean",
    trim=False,
) -> xr.Dataset:
    """Aggregate fields along a given direction.

    It attempts to aggregate the given dataset dim in an integer number of sections
    of length = ``window_size``.
    If such a aggregation is not possible, an error is raised unless ``trim``
    set to True, in which case the dim is trimmed (from the end)
    to make it perfectly divisible".

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset containing the input fields as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    window_size: int or array-like of ints
        The length of the window that is used to aggregate the fields.
        If a single integer value is given, the same window is used for
        all the selected dim.

        If ``window_size`` is a 1D array-like,
        each element indicates the length of the window that is used
        to aggregate the fields along each dim. In this case,
        the number of elements of 'window_size' must be the same as the elements
        in the ``dim`` argument.
    dim: str or array-like of strs
        Dim or dims where to perform the aggregation.
        If this is an array-like of strs, the aggregation is performed over multiple
        dims, instead of a single dim
    method: string, optional
        Optional argument that specifies the operation to use
        to aggregate the precipitation values within the window.
        Default to mean operator.
    velocity_method: string, optional
        Optional argument that specifies the operation to use
        to aggregate the velocity values within the window.
        Default to mean operator.
    trim: bool
         In case that the ``data`` is not perfectly divisible by
         ``window_size`` along the selected dim:

         - trim=True: the data will be trimmed (from the end) along that
           dim to make it perfectly divisible.
         - trim=False: a ValueError exception is raised.

    Returns
    -------
    dataset: xarray.Dataset
        The new dataset.

    See also
    --------
    pysteps.utils.dimension.aggregate_fields_time,
    pysteps.utils.dimension.aggregate_fields_space
    """

    if np.ndim(dim) > 1:
        raise TypeError(
            "Only integers or integer 1D arrays can be used for the 'axis' argument."
        )

    if np.ndim(dim) == 0:
        dim = [dim]

    if any(
        d == "time"
        or d in dataset.coords
        and dataset[d].attrs.get("standard_name") == "time"
        for d in dim
    ):
        raise ValueError(
            "Use aggregate_fields_time to preserve temporal support and integration assumptions"
        )

    if np.ndim(window_size) == 0:
        window_size = [window_size for _ in dim]

    if len(window_size) != len(dim):
        raise TypeError("The length of window size does not to match the length of dim")

    if method not in _aggregation_methods:
        raise ValueError(
            "Aggregation method not recognized. "
            f"Available methods: {str(list(_aggregation_methods.keys()))}"
        )
    for ws in window_size:
        if ws <= 0:
            raise ValueError("'window_size' must be strictly positive")

    for d, ws in zip(dim, window_size):
        if (dataset.sizes[d] % ws) and (not trim):
            raise ValueError(
                f"Since 'trim' argument was set to False,"
                f"the 'window_size' {ws} must exactly divide"
                f"the dimension along the selected axis:"
                f"dataset.sizes[dim]={dataset.sizes[d]}"
            )

    dataset_ref = dataset
    dataset = (
        dataset.rolling(dict(zip(dim, window_size)))
        .reduce(_aggregation_methods[method])
        .isel(
            {
                d: slice(ws - 1, dataset.sizes[d] - dataset.sizes[d] % ws, ws)
                for d, ws in zip(dim, window_size)
            }
        )
    )
    if "velocity_x" in dataset_ref:
        dataset["velocity_x"] = (
            dataset_ref["velocity_x"]
            .rolling(dict(zip(dim, window_size)))
            .reduce(_aggregation_methods[velocity_method])
            .isel(
                {
                    d: slice(
                        ws - 1, dataset_ref.sizes[d] - dataset_ref.sizes[d] % ws, ws
                    )
                    for d, ws in zip(dim, window_size)
                }
            )
        )
    if "velocity_y" in dataset_ref:
        dataset["velocity_y"] = (
            dataset_ref["velocity_y"]
            .rolling(dict(zip(dim, window_size)))
            .reduce(_aggregation_methods[velocity_method])
            .isel(
                {
                    d: slice(
                        ws - 1, dataset_ref.sizes[d] - dataset_ref.sizes[d] % ws, ws
                    )
                    for d, ws in zip(dim, window_size)
                }
            )
        )
    if "quality" in dataset_ref:
        dataset["quality"] = (
            dataset_ref["quality"]
            .rolling(dict(zip(dim, window_size)))
            .reduce(_aggregation_methods["min"])
            .isel(
                {
                    d: slice(
                        ws - 1, dataset_ref.sizes[d] - dataset_ref.sizes[d] % ws, ws
                    )
                    for d, ws in zip(dim, window_size)
                }
            )
        )

    for d, ws in zip(dim, window_size):
        dataset[d] = (
            dataset_ref[d]
            .rolling({d: ws})
            .reduce(_aggregation_methods["mean"])
            .isel(
                {d: slice(ws - 1, dataset_ref.sizes[d] - dataset_ref.sizes[d] % ws, ws)}
            )
        )

    for d, ws in zip(dim, window_size):
        if "stepsize" in dataset[d].attrs:
            dataset[d].attrs["stepsize"] = dataset[d].attrs["stepsize"] * ws

    if {"x", "y"}.intersection(dim) and "lat" in dataset and "lon" in dataset:
        projection = dataset.attrs.get("projection")
        if projection is None:
            from pysteps.dataset_metadata import dataset_crs

            projection = dataset_crs(dataset, dataset.attrs["precip_var"]).to_wkt()
        lat, lon = compute_lat_lon(
            dataset.x.values,
            dataset.y.values,
            projection,
            coordinate_unit=dataset.x.attrs["units"],
        )
        dataset = dataset.assign_coords(
            lat=(("y", "x"), lat, dataset_ref.lat.attrs),
            lon=(("y", "x"), lon, dataset_ref.lon.attrs),
        )
    return dataset


def clip_domain(dataset: xr.Dataset, extent=None):
    """
    Clip the field domain by geographical coordinates.

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset containing the input fields as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    extent: scalars (left, right, bottom, top), optional
        The extent of the bounding box in data coordinates to be used to clip
        the data.
        Note that the direction of the vertical axis and thus the default
        values for top and bottom depend on origin. We follow the same
        convention as in the imshow method of matplotlib:
        https://matplotlib.org/tutorials/intermediate/imshow_extent.html

    Returns
    -------
    dataset: xarray.Dataset
        The clipped dataset
    """
    if extent is None:
        return dataset
    if dataset.y.stepsize < 0:
        return dataset.sel(x=slice(extent[0], extent[1]), y=slice(extent[3], extent[2]))
    return dataset.sel(x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3]))


def _pad_domain(
    dataset: xr.Dataset, dim_to_pad: str, idx_buffer: int, zerovalue: float
) -> xr.Dataset:
    """Pad one spatial dimension and refresh any geographic coordinates."""
    delta = dataset[dim_to_pad].attrs["stepsize"]
    end_values = (
        dataset[dim_to_pad].values[0] - delta * idx_buffer,
        dataset[dim_to_pad].values[-1] + delta * idx_buffer,
    )

    dataset_ref = dataset

    dataset = dataset_ref.pad({dim_to_pad: idx_buffer}, constant_values=zerovalue)
    dataset[dim_to_pad] = dataset_ref[dim_to_pad].pad(
        {dim_to_pad: idx_buffer},
        mode="linear_ramp",
        end_values={dim_to_pad: end_values},
    )
    if "lat" in dataset and "lon" in dataset:
        lat, lon = compute_lat_lon(
            dataset.x.values,
            dataset.y.values,
            dataset.attrs["projection"],
            coordinate_unit=dataset.x.attrs["units"],
        )
        dataset = dataset.assign_coords(
            lat=(("y", "x"), lat, dataset_ref.lat.attrs),
            lon=(("y", "x"), lon, dataset_ref.lon.attrs),
        )
    if "velocity_x" in dataset_ref:
        dataset["velocity_x"].data = (
            dataset_ref["velocity_x"]
            .pad({dim_to_pad: idx_buffer}, constant_values=0.0)
            .values
        )
    if "velocity_y" in dataset_ref:
        dataset["velocity_y"].data = (
            dataset_ref["velocity_y"]
            .pad({dim_to_pad: idx_buffer}, constant_values=0.0)
            .values
        )
    if "quality" in dataset_ref:
        dataset["quality"].data = (
            dataset_ref["quality"]
            .pad({dim_to_pad: idx_buffer}, constant_values=0.0)
            .values
        )
    return dataset


def square_domain(dataset: xr.Dataset, method="pad", inverse=False):
    """
    Either pad or crop a field to obtain a square domain.

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset containing the input fields as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    method: {'pad', 'crop'}, optional
        Either pad or crop.
        If pad, an equal number of pixels
        filled with the minimum value of the precipitation
        field is added to both ends of the precipitation fields shortest
        side in order to produce a square domain. The quality and velocity fields
        are always padded with zeros.
        If crop, an equal number of pixels is removed
        to both ends of its longest side in order to produce a square domain.
        Note that the crop method involves an irreversible loss of data.
    inverse: bool, optional
        Perform the inverse method to recover the original domain shape.
        After a crop, the inverse is performed by doing a pad.

    Returns
    -------
    dataset: xarray.Dataset
        the reshaped dataset
    """

    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    precip_data = dataset[precip_var].values

    x_len = len(dataset.x.values)
    y_len = len(dataset.y.values)

    if inverse:
        if "orig_domain" not in dataset.attrs or "square_method" not in dataset.attrs:
            raise ValueError("Attempting to inverse a non squared dataset")
        method = dataset.attrs.pop("square_method")
        orig_domain = dataset.attrs.pop("orig_domain")

        if method == "pad":
            if x_len > len(orig_domain[1]):
                extent = (
                    orig_domain[1].min(),
                    orig_domain[1].max(),
                    dataset.y.values.min(),
                    dataset.y.values.max(),
                )
            elif y_len > len(orig_domain[0]):
                extent = (
                    dataset.x.values.min(),
                    dataset.x.values.max(),
                    orig_domain[0].min(),
                    orig_domain[0].max(),
                )
            else:
                return dataset
            return clip_domain(dataset, extent)

        if method == "crop":
            if x_len < len(orig_domain[1]):
                dim_to_pad = "x"
                idx_buffer = int((len(orig_domain[1]) - x_len) / 2.0)
            elif y_len < len(orig_domain[0]):
                dim_to_pad = "y"
                idx_buffer = int((len(orig_domain[0]) - y_len) / 2.0)
            else:
                return dataset
            return _pad_domain(dataset, dim_to_pad, idx_buffer, np.nanmin(precip_data))

        raise ValueError(f"Unknown square method: {method}")

    else:
        if "orig_domain" in dataset.attrs and "square_method" in dataset.attrs:
            raise ValueError("Attempting to square an already squared dataset")
        dataset.attrs["orig_domain"] = (dataset.y.values, dataset.x.values)
        dataset.attrs["square_method"] = method

        if method == "pad":
            if x_len > y_len:
                dim_to_pad = "y"
                idx_buffer = int((x_len - y_len) / 2.0)
            elif y_len > x_len:
                dim_to_pad = "x"
                idx_buffer = int((y_len - x_len) / 2.0)
            else:
                return dataset
            return _pad_domain(dataset, dim_to_pad, idx_buffer, np.nanmin(precip_data))

        if method == "crop":
            if x_len > y_len:
                idx_buffer = int((x_len - y_len) / 2.0)
                extent = (
                    dataset.x.values[idx_buffer],
                    dataset.x.values[-idx_buffer - 1],
                    dataset.y.values.min(),
                    dataset.y.values.max(),
                )
            elif y_len > x_len:
                idx_buffer = int((y_len - x_len) / 2.0)
                extent = (
                    dataset.x.values.min(),
                    dataset.x.values.max(),
                    dataset.y.values[idx_buffer],
                    dataset.y.values[-idx_buffer - 1],
                )
            else:
                return dataset
            return clip_domain(dataset, extent)

        raise ValueError(f"Unknown square method: {method}")
