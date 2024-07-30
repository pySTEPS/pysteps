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
import numpy as np
import xarray as xr

from pysteps.converters import compute_lat_lon

_aggregation_methods = dict(
    sum=np.sum, mean=np.mean, nanmean=np.nanmean, nansum=np.nansum
)


def aggregate_fields_time(
    dataset: xr.Dataset, time_window_min, ignore_nan=False
) -> xr.Dataset:
    """Aggregate fields in time.

    It attempts to aggregate the given dataset in the time direction in an integer
    number of sections of length = ``time_window_min``.
    If such a aggregation is not possible, an error is raised.
    The data is aggregated by a method chosen based on the unit of the precipitation
    data in the dataset. ``mean`` is used when the unit is ``mm/h`` and ``sum``
    is used when the unit is ``mm``. For other units an error is raised.

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset containing
        a time series of (ensemble) input fields.
        They must be evenly spaced in time.
    time_window_min: float or None
        The length in minutes of the time window that is used to
        aggregate the fields.
        The time spanned by the t dimension of R must be a multiple of
        time_window_min.
        If set to None, it returns a copy of the original R and metadata.
    ignore_nan: bool, optional
        If True, ignore nan values.

    Returns
    -------
    dataset: xarray.Dataset
        The new dataset.

    See also
    --------
    pysteps.utils.dimension.aggregate_fields_space,
    pysteps.utils.dimension.aggregate_fields
    """

    if time_window_min is None:
        return dataset

    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs

    unit = metadata["units"]

    timestamps = dataset["time"].values

    # assumes that frames are evenly spaced
    delta = (timestamps[1] - timestamps[0]) / np.timedelta64(1, "m")
    if delta == time_window_min:
        return dataset
    if time_window_min % delta:
        raise ValueError("time_window_size does not equally split dataset")

    window_size = int(time_window_min / delta)

    # specify the operator to be used to aggregate
    # the values within the time window
    if unit == "mm/h":
        method = "mean"
    elif unit == "mm":
        method = "sum"
    else:
        raise ValueError(f"can only aggregate units of 'mm/h' or 'mm' not {unit}")

    if ignore_nan:
        method = "".join(("nan", method))

    return aggregate_fields(dataset, window_size, dim="time", method=method)


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
        a time series of (ensemble) input fields.
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

    # assumes that frames are evenly spaced
    ydelta = dataset["y"].values[1] - dataset["y"].values[0]
    xdelta = dataset["x"].values[1] - dataset["x"].values[0]

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

    window_size = (int(space_window[0] / ydelta), int(space_window[1] / xdelta))

    return aggregate_fields(dataset, window_size, ["y", "x"], method)


def aggregate_fields(
    dataset: xr.Dataset, window_size, dim="x", method="mean", trim=False
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
        Dataset containing the input fields.
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
        to aggregate the values within the window.
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
            "Only integers or integer 1D arrays can be used for the " "'axis' argument."
        )

    if np.ndim(dim) == 0:
        dim = [dim]

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

    # FIXME: The aggregation method is applied to all DataArrays in the Dataset
    #        Fix to allow support for an aggregation method per DataArray
    return (
        dataset.rolling(dict(zip(dim, window_size)))
        .reduce(_aggregation_methods[method])
        .isel(
            {
                d: slice(ws - 1, dataset.sizes[d] - dataset.sizes[d] % ws, ws)
                for d, ws in zip(dim, window_size)
            }
        )
    )


def clip_domain(dataset: xr.Dataset, extent=None):
    """
    Clip the field domain by geographical coordinates.

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset containing the input fields.
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
    return dataset.sel(x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3]))


def _pad_domain(
    dataset: xr.Dataset, dim_to_pad: str, idx_buffer: int, zerovalue: float
) -> xr.Dataset:
    # assumes that frames are evenly spaced
    delta = dataset[dim_to_pad].values[1] - dataset[dim_to_pad].values[0]
    end_values = (
        dataset[dim_to_pad].values[0] - delta * idx_buffer,
        dataset[dim_to_pad].values[-1] + delta * idx_buffer,
    )

    dataset_ref = dataset

    # FIXME: The same zerovalue is used for all DataArrays in the Dataset
    #        Fix to allow support for a zerovalue per DataArray
    dataset = dataset_ref.pad({dim_to_pad: idx_buffer}, constant_values=zerovalue)
    dataset[dim_to_pad] = dataset_ref[dim_to_pad].pad(
        {dim_to_pad: idx_buffer},
        mode="linear_ramp",
        end_values={dim_to_pad: end_values},
    )
    dataset.lat.data[:], dataset.lon.data[:] = compute_lat_lon(
        dataset.x.values, dataset.y.values, dataset.attrs["projection"]
    )
    return dataset


def square_domain(dataset: xr.Dataset, method="pad", inverse=False):
    """
    Either pad or crop a field to obtain a square domain.

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset containing the input fields.
    method: {'pad', 'crop'}, optional
        Either pad or crop.
        If pad, an equal number of zeros is added to both ends of its shortest
        side in order to produce a square domain.
        If crop, an equal number of pixels is removed
        to both ends of its longest side in order to produce a square domain.
        Note that the crop method involves an irreversible loss of data.
    inverse: bool, optional
        Perform the inverse method to recover the original domain shape.
        After a crop, the inverse is performed by padding the field with zeros.

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
