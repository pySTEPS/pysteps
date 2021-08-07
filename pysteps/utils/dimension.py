# -*- coding: utf-8 -*-
"""
pysteps.utils.dimension
=======================

Utility functions to manipulate array dimensions.

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

_aggregation_methods = dict(
    sum=np.sum, mean=np.mean, nanmean=np.nanmean, nansum=np.nansum
)


def aggregate_fields_time(data_array, time_window_min, ignore_nan=False):
    """Aggregate fields in time.

    Parameters
    ----------
    data_array: xr.DataArray
        Array of shape (t,m,n) or (l,t,m,n) containing a time series of (ensemble)
        input fields. They must be evenly spaced in time.
    time_window_min: float or None
        The length in minutes of the time window that is used to aggregate the fields.
        The time spanned by the t dimension of data_array must be a multiple of
        time_window_min.
        If set to None, it returns a copy of the original data_array.
    ignore_nan: bool, optional
        If True, ignore nan values.

    Returns
    -------
    data_array: xr.DataArray
        The new array of aggregated fields of shape (k,m,n) or (l,k,m,n), where
        k = t*delta/time_window_min and delta is the time interval between two
        successive timestamps.

    See also
    --------
    pysteps.utils.dimension.aggregate_fields_space,
    pysteps.utils.dimension.aggregate_fields
    """

    data_array = data_array.copy()

    if time_window_min is None:
        return data_array

    if len(data_array.shape) < 3:
        raise ValueError("The number of dimension must be > 2")
    if len(data_array.shape) == 3:
        axis = 0
    if len(data_array.shape) == 4:
        axis = 1
    if len(data_array.shape) > 4:
        raise ValueError("The number of dimension must be <= 4")

    # assumes that frames are evenly spaced
    delta = (data_array.t[1] - data_array.t[0]).seconds / 60
    if delta == time_window_min:
        return data_array
    if (data_array.shape[axis] * delta) % time_window_min:
        raise ValueError("time_window_size does not equally split data_array")

    nframes = int(time_window_min / delta)

    # specify the operator to be used to aggregate
    # the values within the time window
    if unit == "mm/h":
        method = "mean"
    elif unit == "mm":
        method = "sum"
    else:
        raise ValueError(
            "can only aggregate units of 'mm/h' or 'mm'" + " not %s" % unit
        )

    if ignore_nan:
        method = "".join(("nan", method))

    data_array = aggregate_fields(data_array, nframes, axis=axis, method=method)

    metadata["accutime"] = time_window_min
    metadata["timestamps"] = timestamps[nframes - 1 :: nframes]
    if "leadtimes" in metadata:
        metadata["leadtimes"] = leadtimes[nframes - 1 :: nframes]

    return data_array


def aggregate_fields_space(data_array, space_window, ignore_nan=False):
    """Upscale fields in space.

    Parameters
    ----------
    data_array: xr.DataArray
        Array of shape (m,n), (t,m,n) or (l,t,m,n) containing a single field or
        a time series of (ensemble) input fields.
    space_window: float or None
        The length of the space window that is used to upscale the fields.
        The space_window unit is the same used in the geographical projection
        of data_array and hence the same as for the xpixelsize and ypixelsize attributes.
        The space spanned by the m and n dimensions of data_array must be a multiple of
        space_window.
        If set to None, it returns a copy of the original data_array and metadata.
    ignore_nan: bool, optional
        If True, ignore nan values.

    Returns
    -------
    data_array: xr.DataArray
        The new array of aggregated fields of shape (k,j), (t,k,j)
        or (l,t,k,j),
        where k = m*ypixelsize/space_window and j = n*xpixelsize/space_window.

    See also
    --------
    pysteps.utils.dimension.aggregate_fields_time,
    pysteps.utils.dimension.aggregate_fields
    """

    data_array = data_array.copy()

    if space_window is None:
        return data_array

    unit = metadata["unit"]
    ypixelsize = metadata["ypixelsize"]
    xpixelsize = metadata["xpixelsize"]

    if len(data_array.shape) < 2:
        raise ValueError("The number of dimensions must be >= 2")
    if len(data_array.shape) == 2:
        axes = [0, 1]
    if len(data_array.shape) == 3:
        axes = [1, 2]
    if len(data_array.shape) == 4:
        axes = [2, 3]
    if len(data_array.shape) > 4:
        raise ValueError("The number of dimensions must be <= 4")

    # assumes that frames are evenly spaced
    if ypixelsize == space_window and xpixelsize == space_window:
        return data_array
    if (data_array.shape[axes[0]] * ypixelsize) % space_window or (
        data_array.shape[axes[1]] * xpixelsize
    ) % space_window:
        raise ValueError("space_window does not equally split data_array")

    nframes = [int(space_window / ypixelsize), int(space_window / xpixelsize)]

    # specify the operator to be used to aggregate the values
    # within the space window
    if unit == "mm/h":
        method = "mean"
    elif unit == "mm":
        method = "sum"
    else:
        raise ValueError(
            "can only aggregate units of 'mm/h' or 'mm' " + "not %s" % unit
        )

    if ignore_nan:
        method = "".join(("nan", method))

    data_array = aggregate_fields(data_array, nframes[0], axis=axes[0], method=method)
    data_array = aggregate_fields(data_array, nframes[1], axis=axes[1], method=method)

    metadata["ypixelsize"] = space_window
    metadata["xpixelsize"] = space_window

    return data_array


def aggregate_fields(data, window_size, axis=0, method="mean", trim=False):
    """Aggregate fields along a given direction.

    It attempts to aggregate the given data_array axis in an integer number of sections
    of length = ``window_size``.
    If such a aggregation is not possible, an error is raised unless ``trim``
    set to True, in which case the axis is trimmed (from the end)
    to make it perfectly divisible".

    Parameters
    ----------
    data_array: xr.DataArray
        Array of any shape containing the input fields.
    window_size: int or tuple of ints
        The length of the window that is used to aggregate the fields.
        If a single integer value is given, the same window is used for
        all the selected axis.

        If ``window_size`` is a 1D array-like,
        each element indicates the length of the window that is used
        to aggregate the fields along each axis. In this case,
        the number of elements of 'window_size' must be the same as the elements
        in the ``axis`` argument.
    axis: int or array-like of ints
        Axis or axes where to perform the aggregation.
        If this is a tuple of ints, the aggregation is performed over multiple
        axes, instead of a single axis
    method: string, optional
        Optional argument that specifies the operation to use
        to aggregate the values within the window.
        Default to mean operator.
    trim: bool
         In case that the ``data`` is not perfectly divisible by
         ``window_size`` along the selected axis:

         - trim=True: the data will be trimmed (from the end) along that
           axis to make it perfectly divisible.
         - trim=False: a ValueError exception is raised.

    Returns
    -------
    data_array: xr.DataArray
        The new aggregated array with shape[axis] = k,
        where k = data_array.shape[axis] / window_size.

    See also
    --------
    pysteps.utils.dimension.aggregate_fields_time,
    pysteps.utils.dimension.aggregate_fields_space
    """

    if np.ndim(axis) > 1:
        raise TypeError(
            "Only integers or integer 1D arrays can be used for the " "'axis' argument."
        )

    if np.ndim(axis) == 1:
        axis = np.asarray(axis)
        if np.ndim(window_size) == 0:
            window_size = (window_size,) * axis.size

        window_size = np.asarray(window_size, dtype="int")

        if window_size.shape != axis.shape:
            raise ValueError(
                "The 'window_size' and 'axis' shapes are incompatible."
                f"window_size.shape: {str(window_size.shape)}, "
                f"axis.shape: {str(axis.shape)}, "
            )

        new_data = data.copy()
        for i in range(axis.size):
            # Recursively call the aggregate_fields function
            new_data = aggregate_fields(
                new_data, window_size[i], axis=axis[i], method=method, trim=trim
            )

        return new_data

    if np.ndim(window_size) != 0:
        raise TypeError(
            "A single axis was selected for the aggregation but several"
            f"of window_sizes were given: {str(window_size)}."
        )

    data = np.asarray(data).copy()
    orig_shape = data.shape

    if method not in _aggregation_methods:
        raise ValueError(
            "Aggregation method not recognized. "
            f"Available methods: {str(list(_aggregation_methods.keys()))}"
        )

    if window_size <= 0:
        raise ValueError("'window_size' must be strictly positive")

    if (orig_shape[axis] % window_size) and (not trim):
        raise ValueError(
            f"Since 'trim' argument was set to False,"
            f"the 'window_size' {window_size} must exactly divide"
            f"the dimension along the selected axis:"
            f"data.shape[axis]={orig_shape[axis]}"
        )

    new_data = data.swapaxes(axis, 0)
    if trim:
        trim_size = data.shape[axis] % window_size
        if trim_size > 0:
            new_data = new_data[:-trim_size]

    new_data_shape = list(new_data.shape)
    new_data_shape[0] //= window_size  # Final shape

    new_data = new_data.reshape(new_data_shape[0], window_size, -1)

    new_data = _aggregation_methods[method](new_data, axis=1)

    new_data = new_data.reshape(new_data_shape).swapaxes(axis, 0)

    return new_data


def clip_domain(data_array, extent=None):
    """Clip the field domain by geographical coordinates.

    Parameters
    ----------
    data_array: xr.DataArray
        Array of shape (m,n) or (t,m,n) containing the input fields.
    extent: scalars (left, right, bottom, top), optional
        The extent of the bounding box in data coordinates to be used to clip
        the data.
        Note that the direction of the vertical axis and thus the default
        values for top and bottom depend on origin. We follow the same
        convention as in the imshow method of matplotlib:
        https://matplotlib.org/tutorials/intermediate/imshow_extent.html

    Returns
    -------
    data_array: xr.DataArray
        the clipped array
    """

    data_array = data_array.copy()
    R_shape = np.array(data_array.shape)
    metadata = metadata.copy()

    if extent is None:
        return data_array

    if len(data_array.shape) < 2:
        raise ValueError("The number of dimension must be > 1")
    if len(data_array.shape) == 2:
        data_array = data_array[None, None, :, :]
    if len(data_array.shape) == 3:
        data_array = data_array[None, :, :, :]
    if len(data_array.shape) > 4:
        raise ValueError("The number of dimension must be <= 4")

    # extract original domain coordinates
    left = metadata["x1"]
    right = metadata["x2"]
    bottom = metadata["y1"]
    top = metadata["y2"]

    # extract bounding box coordinates
    left_ = extent[0]
    right_ = extent[1]
    bottom_ = extent[2]
    top_ = extent[3]

    # compute its extent in pixels
    dim_x_ = int((right_ - left_) / metadata["xpixelsize"])
    dim_y_ = int((top_ - bottom_) / metadata["ypixelsize"])
    R_ = np.ones((data_array.shape[0], data_array.shape[1], dim_y_, dim_x_)) * metadata["zerovalue"]

    # build set of coordinates for the original domain
    y_coord = (
        np.linspace(bottom, top - metadata["ypixelsize"], data_array.shape[2])
        + metadata["ypixelsize"] / 2.0
    )
    x_coord = (
        np.linspace(left, right - metadata["xpixelsize"], data_array.shape[3])
        + metadata["xpixelsize"] / 2.0
    )

    # build set of coordinates for the new domain
    y_coord_ = (
        np.linspace(bottom_, top_ - metadata["ypixelsize"], R_.shape[2])
        + metadata["ypixelsize"] / 2.0
    )
    x_coord_ = (
        np.linspace(left_, right_ - metadata["xpixelsize"], R_.shape[3])
        + metadata["xpixelsize"] / 2.0
    )

    # origin='upper' reverses the vertical axes direction
    if metadata["yorigin"] == "upper":
        y_coord = y_coord[::-1]
        y_coord_ = y_coord_[::-1]

    # extract original domain
    idx_y = np.where(np.logical_and(y_coord < top_, y_coord > bottom_))[0]
    idx_x = np.where(np.logical_and(x_coord < right_, x_coord > left_))[0]

    # extract new domain
    idx_y_ = np.where(np.logical_and(y_coord_ < top, y_coord_ > bottom))[0]
    idx_x_ = np.where(np.logical_and(x_coord_ < right, x_coord_ > left))[0]

    # compose the new array
    R_[:, :, idx_y_[0] : (idx_y_[-1] + 1), idx_x_[0] : (idx_x_[-1] + 1)] = data_array[
        :, :, idx_y[0] : (idx_y[-1] + 1), idx_x[0] : (idx_x[-1] + 1)
    ]

    # update coordinates
    metadata["y1"] = bottom_
    metadata["y2"] = top_
    metadata["x1"] = left_
    metadata["x2"] = right_

    R_shape[-2] = R_.shape[-2]
    R_shape[-1] = R_.shape[-1]

    return R_.reshape(R_shape)


def square_domain(data_array, method="pad", inverse=False):
    """Either pad or crop a field to obtain a square domain.

    Parameters
    ----------
    data_array: xr.DataArray
        Array of shape (m,n) or (t,m,n) containing the input fields.
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
    data_array: xr.DataArray
        the reshape dataset
    """

    data_array = data_array.copy()
    R_shape = np.array(data_array.shape)
    metadata = metadata.copy()

    if not inverse:

        if len(data_array.shape) < 2:
            raise ValueError("The number of dimension must be > 1")
        if len(data_array.shape) == 2:
            data_array = data_array[None, None, :]
        if len(data_array.shape) == 3:
            data_array = data_array[None, :]
        if len(data_array.shape) > 4:
            raise ValueError("The number of dimension must be <= 4")

        if data_array.shape[2] == data_array.shape[3]:
            return data_array.squeeze()

        orig_dim = data_array.shape
        orig_dim_n = orig_dim[0]
        orig_dim_t = orig_dim[1]
        orig_dim_y = orig_dim[2]
        orig_dim_x = orig_dim[3]

        if method == "pad":

            new_dim = np.max(orig_dim[2:])
            R_ = np.ones((orig_dim_n, orig_dim_t, new_dim, new_dim)) * data_array.min()

            if orig_dim_x < new_dim:
                idx_buffer = int((new_dim - orig_dim_x) / 2.0)
                R_[:, :, :, idx_buffer : (idx_buffer + orig_dim_x)] = data_array
                metadata["x1"] -= idx_buffer * metadata["xpixelsize"]
                metadata["x2"] += idx_buffer * metadata["xpixelsize"]

            elif orig_dim_y < new_dim:
                idx_buffer = int((new_dim - orig_dim_y) / 2.0)
                R_[:, :, idx_buffer : (idx_buffer + orig_dim_y), :] = data_array
                metadata["y1"] -= idx_buffer * metadata["ypixelsize"]
                metadata["y2"] += idx_buffer * metadata["ypixelsize"]

        elif method == "crop":

            new_dim = np.min(orig_dim[2:])
            R_ = np.zeros((orig_dim_n, orig_dim_t, new_dim, new_dim))

            if orig_dim_x > new_dim:
                idx_buffer = int((orig_dim_x - new_dim) / 2.0)
                R_ = data_array[:, :, :, idx_buffer : (idx_buffer + new_dim)]
                metadata["x1"] += idx_buffer * metadata["xpixelsize"]
                metadata["x2"] -= idx_buffer * metadata["xpixelsize"]

            elif orig_dim_y > new_dim:
                idx_buffer = int((orig_dim_y - new_dim) / 2.0)
                R_ = data_array[:, :, idx_buffer : (idx_buffer + new_dim), :]
                metadata["y1"] += idx_buffer * metadata["ypixelsize"]
                metadata["y2"] -= idx_buffer * metadata["ypixelsize"]

        else:
            raise ValueError("Unknown type")

        metadata["orig_domain"] = (orig_dim_y, orig_dim_x)
        metadata["square_method"] = method

        R_shape[-2] = R_.shape[-2]
        R_shape[-1] = R_.shape[-1]

        return R_.reshape(R_shape), metadata

    elif inverse:

        if len(data_array.shape) < 2:
            raise ValueError("The number of dimension must be > 2")
        if len(data_array.shape) == 2:
            data_array = data_array[None, None, :]
        if len(data_array.shape) == 3:
            data_array = data_array[None, :]
        if len(data_array.shape) > 4:
            raise ValueError("The number of dimension must be <= 4")

        method = metadata.pop("square_method")
        shape = metadata.pop("orig_domain")

        if data_array.shape[2] == shape[0] and data_array.shape[3] == shape[1]:
            return data_array.squeeze(), metadata

        R_ = np.zeros((data_array.shape[0], data_array.shape[1], shape[0], shape[1]))

        if method == "pad":

            if data_array.shape[2] == shape[0]:
                idx_buffer = int((data_array.shape[3] - shape[1]) / 2.0)
                R_ = data_array[:, :, :, idx_buffer : (idx_buffer + shape[1])]
                metadata["x1"] += idx_buffer * metadata["xpixelsize"]
                metadata["x2"] -= idx_buffer * metadata["xpixelsize"]

            elif data_array.shape[3] == shape[1]:
                idx_buffer = int((data_array.shape[2] - shape[0]) / 2.0)
                R_ = data_array[:, :, idx_buffer : (idx_buffer + shape[0]), :]
                metadata["y1"] += idx_buffer * metadata["ypixelsize"]
                metadata["y2"] -= idx_buffer * metadata["ypixelsize"]

        elif method == "crop":

            if data_array.shape[2] == shape[0]:
                idx_buffer = int((shape[1] - data_array.shape[3]) / 2.0)
                R_[:, :, :, idx_buffer : (idx_buffer + data_array.shape[3])] = data_array
                metadata["x1"] -= idx_buffer * metadata["xpixelsize"]
                metadata["x2"] += idx_buffer * metadata["xpixelsize"]

            elif data_array.shape[3] == shape[1]:
                idx_buffer = int((shape[0] - data_array.shape[2]) / 2.0)
                R_[:, :, idx_buffer : (idx_buffer + data_array.shape[2]), :] = data_array
                metadata["y1"] -= idx_buffer * metadata["ypixelsize"]
                metadata["y2"] += idx_buffer * metadata["ypixelsize"]

        R_shape[-2] = R_.shape[-2]
        R_shape[-1] = R_.shape[-1]

        return R_.reshape(R_shape), metadata
