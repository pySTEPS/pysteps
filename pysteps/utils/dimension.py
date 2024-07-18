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

_aggregation_methods = dict(
    sum=np.sum, mean=np.mean, nanmean=np.nanmean, nansum=np.nansum
)


def aggregate_fields_time(dataset: xr.Dataset, time_window_min, ignore_nan=False):
    """Aggregate fields in time.

    Parameters
    ----------
    dataset: Dataset
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
    dataset: Dataset
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


def aggregate_fields_space(dataset: xr.Dataset, space_window, ignore_nan=False):
    """
    Upscale fields in space.

    Parameters
    ----------
    dataset: Dataset
        Dataset containing a single field or
        a time series of (ensemble) input fields.
    space_window: float, tuple or None
        The length of the space window that is used to upscale the fields.
        If a float is given, the same window size is used for the x- and
        y-directions. Separate window sizes are used for x- and y-directions if
        a two-element tuple is given. The space_window unit is the same used in
        the geographical projection of R and hence the same as for the xpixelsize
        and ypixelsize attributes. The space spanned by the n- and m-dimensions
        of R must be a multiple of space_window. If set to None, the function
        returns a copy of the original R and metadata.
    ignore_nan: bool, optional
        If True, ignore nan values.

    Returns
    -------
    dataset: Dataset
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
):
    """Aggregate fields along a given direction.

    It attempts to aggregate the given R dim in an integer number of sections
    of length = ``window_size``.
    If such a aggregation is not possible, an error is raised unless ``trim``
    set to True, in which case the dim is trimmed (from the end)
    to make it perfectly divisible".

    Parameters
    ----------
    dataset: Dataset
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
    dataset: Dataset
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
    dataset: Dataset
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
    dataset: Dataset
        The clipped dataset
    """
    if extent is None:
        return dataset
    return dataset.sel(x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3]))


def square_domain(R, metadata, method="pad", inverse=False):
    """
    Either pad or crop a field to obtain a square domain.

    Parameters
    ----------
    R: array-like
        Array of shape (m,n) or (t,m,n) containing the input fields.
    metadata: dict
        Metadata dictionary containing the x1, x2, y1, y2,
        xpixelsize, ypixelsize,
        attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.
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
    R: array-like
        the reshape dataset
    metadata: dict
        the metadata with updated attributes.
    """

    R = R.copy()
    R_shape = np.array(R.shape)
    metadata = metadata.copy()

    if not inverse:
        if len(R.shape) < 2:
            raise ValueError("The number of dimension must be > 1")
        if len(R.shape) == 2:
            R = R[None, None, :]
        if len(R.shape) == 3:
            R = R[None, :]
        if len(R.shape) > 4:
            raise ValueError("The number of dimension must be <= 4")

        if R.shape[2] == R.shape[3]:
            return R.squeeze()

        orig_dim = R.shape
        orig_dim_n = orig_dim[0]
        orig_dim_t = orig_dim[1]
        orig_dim_y = orig_dim[2]
        orig_dim_x = orig_dim[3]

        if method == "pad":
            new_dim = np.max(orig_dim[2:])
            R_ = np.ones((orig_dim_n, orig_dim_t, new_dim, new_dim)) * R.min()

            if orig_dim_x < new_dim:
                idx_buffer = int((new_dim - orig_dim_x) / 2.0)
                R_[:, :, :, idx_buffer : (idx_buffer + orig_dim_x)] = R
                metadata["x1"] -= idx_buffer * metadata["xpixelsize"]
                metadata["x2"] += idx_buffer * metadata["xpixelsize"]

            elif orig_dim_y < new_dim:
                idx_buffer = int((new_dim - orig_dim_y) / 2.0)
                R_[:, :, idx_buffer : (idx_buffer + orig_dim_y), :] = R
                metadata["y1"] -= idx_buffer * metadata["ypixelsize"]
                metadata["y2"] += idx_buffer * metadata["ypixelsize"]

        elif method == "crop":
            new_dim = np.min(orig_dim[2:])
            R_ = np.zeros((orig_dim_n, orig_dim_t, new_dim, new_dim))

            if orig_dim_x > new_dim:
                idx_buffer = int((orig_dim_x - new_dim) / 2.0)
                R_ = R[:, :, :, idx_buffer : (idx_buffer + new_dim)]
                metadata["x1"] += idx_buffer * metadata["xpixelsize"]
                metadata["x2"] -= idx_buffer * metadata["xpixelsize"]

            elif orig_dim_y > new_dim:
                idx_buffer = int((orig_dim_y - new_dim) / 2.0)
                R_ = R[:, :, idx_buffer : (idx_buffer + new_dim), :]
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
        if len(R.shape) < 2:
            raise ValueError("The number of dimension must be > 2")
        if len(R.shape) == 2:
            R = R[None, None, :]
        if len(R.shape) == 3:
            R = R[None, :]
        if len(R.shape) > 4:
            raise ValueError("The number of dimension must be <= 4")

        method = metadata.pop("square_method")
        shape = metadata.pop("orig_domain")

        if R.shape[2] == shape[0] and R.shape[3] == shape[1]:
            return R.squeeze(), metadata

        R_ = np.zeros((R.shape[0], R.shape[1], shape[0], shape[1]))

        if method == "pad":
            if R.shape[2] == shape[0]:
                idx_buffer = int((R.shape[3] - shape[1]) / 2.0)
                R_ = R[:, :, :, idx_buffer : (idx_buffer + shape[1])]
                metadata["x1"] += idx_buffer * metadata["xpixelsize"]
                metadata["x2"] -= idx_buffer * metadata["xpixelsize"]

            elif R.shape[3] == shape[1]:
                idx_buffer = int((R.shape[2] - shape[0]) / 2.0)
                R_ = R[:, :, idx_buffer : (idx_buffer + shape[0]), :]
                metadata["y1"] += idx_buffer * metadata["ypixelsize"]
                metadata["y2"] -= idx_buffer * metadata["ypixelsize"]

        elif method == "crop":
            if R.shape[2] == shape[0]:
                idx_buffer = int((shape[1] - R.shape[3]) / 2.0)
                R_[:, :, :, idx_buffer : (idx_buffer + R.shape[3])] = R
                metadata["x1"] -= idx_buffer * metadata["xpixelsize"]
                metadata["x2"] += idx_buffer * metadata["xpixelsize"]

            elif R.shape[3] == shape[1]:
                idx_buffer = int((shape[0] - R.shape[2]) / 2.0)
                R_[:, :, idx_buffer : (idx_buffer + R.shape[2]), :] = R
                metadata["y1"] -= idx_buffer * metadata["ypixelsize"]
                metadata["y2"] += idx_buffer * metadata["ypixelsize"]

        R_shape[-2] = R_.shape[-2]
        R_shape[-1] = R_.shape[-1]

        return R_.reshape(R_shape), metadata
