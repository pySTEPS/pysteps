"""
pysteps.utils.dimension
=======================

Functions to manipulate array dimensions.

.. autosummary::
    :toctree: ../generated/

    aggregate_fields_time
    aggregate_fields_space
    aggregate_fields
    clip_domain
    square_domain
"""

import numpy as np

def aggregate_fields_time(R, metadata, time_window_min, ignore_nan=False):
    """Aggregate fields in time.

    Parameters
    ----------
    R : array-like
        Array of shape (t,m,n) or (l,t,m,n) containing a time series of (ensemble)
        input fields.
        They must be evenly spaced in time.
    metadata : dict
        Metadata dictionary containing the timestamps and unit attributes as
        described in the documentation of :py:mod:`pysteps.io.importers`.
    time_window_min : float or None
        The length in minutes of the time window that is used to aggregate the fields.
        The time spanned by the t dimension of R must be a multiple of time_window_min.
        If set to None, it returns a copy of the original R and metadata.
    ignore_nan : bool, optional
        If True, ignore nan values.

    Returns
    -------
    outputarray : array-like
        The new array of aggregated fields of shape (k,m,n) or (l,k,m,n), where
        k = t*delta/time_window_min and delta is the time interval between two
        successive timestamps.
    metadata : dict
        The metadata with updated attributes.

    See also
    --------
    pysteps.utils.dimension.aggregate_fields_space, 
    pysteps.utils.dimension.aggregate_fields

    """

    R = R.copy()
    metadata = metadata.copy()

    if time_window_min is None:
        return R, metadata

    unit       = metadata["unit"]
    timestamps = metadata["timestamps"]
    if "leadtimes" in metadata:
        leadtimes = metadata["leadtimes"]

    if len(R.shape) < 3:
        raise ValueError("The number of dimension must be > 2")
    if len(R.shape) == 3:
        axis = 0
    if len(R.shape) == 4:
        axis = 1
    if len(R.shape) > 4:
        raise ValueError("The number of dimension must be <= 4")

    if R.shape[axis] != len(timestamps):
        raise ValueError("The list of timestamps has length %i, but R contains %i frames"
                         % (len(timestamps), R.shape[axis]))

    # assumes that frames are evenly spaced
    delta = (timestamps[1] - timestamps[0]).seconds/60
    if delta == time_window_min:
        return R, metadata
    if (R.shape[axis]*delta) % time_window_min:
        raise ValueError('time_window_size does not equally split R')

    nframes = int(time_window_min/delta)

    # specify the operator to be used to aggregate the values within the time window
    if unit == "mm/h":
        method = "mean"
    elif unit == "mm":
        method = "sum"
    else:
        raise ValueError("can only aggregate units of 'mm/h' or 'mm' not %s" % unit)

    if ignore_nan:
        method = "".join(("nan", method))

    R = aggregate_fields(R, nframes, axis=axis, method=method)

    metadata["accutime"] = time_window_min
    metadata["timestamps"] = timestamps[nframes-1::nframes]
    if "leadtimes" in metadata:
        metadata["leadtimes"] = leadtimes[nframes-1::nframes]

    return R, metadata

def aggregate_fields_space(R, metadata, space_window_m, ignore_nan=False):
    """Upscale fields in space.

    Parameters
    ----------
    R : array-like
        Array of shape (m,n), (t,m,n) or (l,t,m,n) containing a single field or
        a time series of (ensemble) input fields.
    metadata : dict
        Metadata dictionary containing the xpixelsize, ypixelsize and unit
        attributes as described in the documentation of :py:mod:`pysteps.io.importers`.
    space_window_m : float or None
        The length in meters of the space window that is used to upscale the fields.
        The space spanned by the m and n dimensions of R must be a multiple of
        space_window_m. If set to None, it returns a copy of the original R and
        metadata.
    ignore_nan : bool, optional
        If True, ignore nan values.

    Returns
    -------
    outputarray : array-like
        The new array of aggregated fields of shape (k,j), (t,k,j) or (l,t,k,j),
        where k = m*delta/space_window_m and j = n*delta/space_window_m; delta is
        the grid size.
    metadata : dict
        The metadata with updated attributes.

    See also
    --------
    pysteps.utils.dimension.aggregate_fields_time,
    pysteps.utils.dimension.aggregate_fields

    """

    R = R.copy()
    metadata = metadata.copy()

    if space_window_m is None:
        return R, metadata

    unit       = metadata["unit"]
    ypixelsize = metadata["ypixelsize"]
    xpixelsize = metadata["xpixelsize"]
    if "leadtimes" in metadata:
        leadtimes = metadata["leadtimes"]

    if len(R.shape) < 2:
        raise ValueError("The number of dimensions must be >= 2")
    if len(R.shape) == 2:
        axes = [0,1]
    if len(R.shape) == 3:
        axes = [1,2]
    if len(R.shape) == 4:
        axes = [2,3]
    if len(R.shape) > 4:
        raise ValueError("The number of dimensions must be <= 4")

    # assumes that frames are evenly spaced
    if ypixelsize == space_window_m and xpixelsize == space_window_m:
        return R, metadata
    if (R.shape[axes[0]]*ypixelsize) % space_window_m or \
       (R.shape[axes[1]]*xpixelsize) % space_window_m:
        raise ValueError('space_window_m does not equally split R')

    nframes = [int(space_window_m/ypixelsize), int(space_window_m/xpixelsize)]

    # specify the operator to be used to aggregate the values within the space window
    if unit == "mm/h":
        method = "mean"
    elif unit == "mm":
        method = "sum"
    else:
        raise ValueError("can only aggregate units of 'mm/h' or 'mm' not %s" % unit)

    if ignore_nan:
        method = "".join(("nan", method))

    R = aggregate_fields(R, nframes[0], axis=axes[0], method=method)
    R = aggregate_fields(R, nframes[1], axis=axes[1], method=method)

    metadata["ypixelsize"] = space_window_m
    metadata["xpixelsize"] = space_window_m

    return R, metadata

def aggregate_fields(R, window_size, axis=0, method="mean"):
    """Aggregate fields.
    It attemps to aggregate the given R axis in an integer number of sections of
    length = window_size.  If such a aggregation is not possible, an error is raised.

    Parameters
    ----------
    R : array-like
        Array of any shape containing the input fields.
    window_size : int
        The length of the window that is used to aggregate the fields.
    axis : int, optional
        The axis where to perform the aggregation.
    method : string, optional
        Optional argument that specifies the operation to use to aggregate the values within the
        window. Default to mean operator.

    Returns
    -------
    outputarray : array-like
        The new aggregated array with shape[axis] = k, where k = R.shape[axis]/window_size

    See also
    --------
    pysteps.utils.dimension.aggregate_fields_time,
    pysteps.utils.dimension.aggregate_fields_space
    """

    N = R.shape[axis]
    if N % window_size:
        raise ValueError('window_size %i does not equally split R.shape[axis] %i' % (window_size, N))

    R = R.copy().swapaxes(axis, 0)
    shape = list(R.shape)
    R_ = R.reshape((N, -1))

    if   method.lower() == "sum":
        R__ = R_.reshape(int(N/window_size), window_size, R_.shape[1]).sum(axis=1)
    elif method.lower() == "mean":
        R__ = R_.reshape(int(N/window_size), window_size, R_.shape[1]).mean(axis=1)
    elif method.lower() == "nansum":
        R__ = np.nansum(R_.reshape(int(N/window_size), window_size, R_.shape[1]), axis=1)
    elif method.lower() == "nanmean":
        R__ = np.nanmean(R_.reshape(int(N/window_size), window_size, R_.shape[1]), axis=1)
    else:
        raise ValueError("unknown method %s" % method)

    shape[0] = int(N/window_size)
    R = R__.reshape(shape).swapaxes(axis, 0)

    return R

def clip_domain(R, metadata, extent=None):
    """Clip the field domain by geographical coordinates.

    Parameters
    ----------
    R : array-like
        Array of shape (m,n) or (t,m,n) containing the input fields.
    metadata : dict
        Metadata dictionary containing the x1, x2, y1, y2, xpixelsize, ypixelsize,
        zerovalue and yorigin attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    extent : scalars (left, right, bottom, top), optional
        The extent of the bounding box in data coordinates to be used to clip
        the data.
        Note that the direction of the vertical axis and thus the default
        values for top and bottom depend on origin. We follow the same
        convention as in the imshow method of matplotlib:
        https://matplotlib.org/tutorials/intermediate/imshow_extent.html

    Returns
    -------
    R : array-like
        the clipped array
    metadata : dict
        the metadata with updated attributes.

    """

    R = R.copy()
    R_shape = np.array(R.shape)
    metadata = metadata.copy()

    if extent is None:
        return R,metadata

    if len(R.shape) < 2:
        raise ValueError("The number of dimension must be > 1")
    if len(R.shape) == 2:
        R = R[None, None, :, :]
    if len(R.shape) == 3:
        R = R[None, :, :, :]
    if len(R.shape) > 4:
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
    dim_x_ = int((right_ - left_)/metadata["xpixelsize"])
    dim_y_ = int((top_ - bottom_)/metadata["ypixelsize"])
    R_ = np.ones((R.shape[0], R.shape[1], dim_y_, dim_x_))*metadata["zerovalue"]

    # build set of coordinates for the original domain
    y_coord = np.linspace(bottom, top - metadata["ypixelsize"], R.shape[2]) \
                        + metadata["ypixelsize"]/2.
    x_coord = np.linspace(left, right - metadata["xpixelsize"], R.shape[3]) \
                        + metadata["xpixelsize"]/2.

    # build set of coordinates for the new domain
    y_coord_ = np.linspace(bottom_, top_ - metadata["ypixelsize"], R_.shape[2]) \
                        + metadata["ypixelsize"]/2.
    x_coord_ = np.linspace(left_, right_ - metadata["xpixelsize"], R_.shape[3]) \
                        + metadata["xpixelsize"]/2.

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
    R_[:, :, idx_y_[0]:(idx_y_[-1] + 1), idx_x_[0]:(idx_x_[-1] + 1)] = \
                    R[:, :, idx_y[0]:(idx_y[-1] + 1), idx_x[0]:(idx_x[-1] + 1)]

    # update coordinates
    metadata["y1"] = bottom_
    metadata["y2"] = top_
    metadata["x1"] = left_
    metadata["x2"] = right_

    R_shape[-2] = R_.shape[-2]
    R_shape[-1] = R_.shape[-1]

    return R_.reshape(R_shape), metadata

def square_domain(R, metadata, method="pad", inverse=False):
    """Either pad or crop a field to obtain a square domain.

    Parameters
    ----------
    R : array-like
        Array of shape (m,n) or (t,m,n) containing the input fields.
    metadata : dict
        Metadata dictionary containing the x1, x2, y1, y2, xpixelsize, ypixelsize,
        attributes as described in the documentation of :py:mod:`pysteps.io.importers`.
    method : {'pad', 'crop'}, optional
        Either pad or crop.
        If pad, an equal number of zeros is added to both ends of its shortest
        side in order to produce a square domain.
        If crop, an equal number of pixels is removed to both ends of its longest
        side in order to produce a square domain.
        Note that the crop method involves an irreversible loss of data.
    inverse : bool, optional
        Perform the inverse method to recover the original domain shape. After a
        crop, the inverse is performed by padding the field with zeros.

    Returns
    -------
    R : array-like
        the reshape dataset
    metadata : dict
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

        orig_dim = (R.shape)
        orig_dim_n = orig_dim[0]
        orig_dim_t = orig_dim[1]
        orig_dim_y = orig_dim[2]
        orig_dim_x = orig_dim[3]

        if method == "pad":

            new_dim = np.max(orig_dim[2:])
            R_ = np.ones((orig_dim_n, orig_dim_t, new_dim, new_dim))*R.min()

            if(orig_dim_x < new_dim):
                idx_buffer = int((new_dim - orig_dim_x)/2.)
                R_[:, :, :, idx_buffer:(idx_buffer + orig_dim_x)] = R
                metadata["x1"] -= idx_buffer*metadata["xpixelsize"]
                metadata["x2"] += idx_buffer*metadata["xpixelsize"]

            elif(orig_dim_y < new_dim):
                idx_buffer = int((new_dim - orig_dim_y)/2.)
                R_[:, :, idx_buffer:(idx_buffer + orig_dim_y), :] = R
                metadata["y1"] -= idx_buffer*metadata["ypixelsize"]
                metadata["y2"] += idx_buffer*metadata["ypixelsize"]

        elif method == "crop":

            new_dim = np.min(orig_dim[2:])
            R_ = np.zeros((orig_dim_n, orig_dim_t, new_dim, new_dim))

            if(orig_dim_x > new_dim):
                idx_buffer = int((orig_dim_x - new_dim)/2.)
                R_ = R[:, :, :, idx_buffer:(idx_buffer + new_dim)]
                metadata["x1"] += idx_buffer*metadata["xpixelsize"]
                metadata["x2"] -= idx_buffer*metadata["xpixelsize"]

            elif(orig_dim_y > new_dim):
                idx_buffer = int((orig_dim_y - new_dim)/2.)
                R_ = R[:, :, idx_buffer:(idx_buffer + new_dim), :]
                metadata["y1"] += idx_buffer*metadata["ypixelsize"]
                metadata["y2"] -= idx_buffer*metadata["ypixelsize"]

        else:
            raise ValueError("Unknown type")

        metadata["orig_domain"] = (orig_dim_y, orig_dim_x)
        metadata["square_method"] = method

        R_shape[-2] = R_.shape[-2]
        R_shape[-1] = R_.shape[-1]

        return R_.reshape(R_shape),metadata

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
            return R.squeeze()

        R_ = np.zeros((R.shape[0], R.shape[1], shape[0], shape[1]))

        if method == "pad":

            if R.shape[2] == shape[0]:
                idx_buffer = int((R.shape[3] - shape[1])/2.)
                R_ = R[:, :, :, idx_buffer:(idx_buffer + shape[1])]
                metadata["x1"] += idx_buffer*metadata["xpixelsize"]
                metadata["x2"] -= idx_buffer*metadata["xpixelsize"]

            elif R.shape[3] == shape[1]:
                idx_buffer = int((R.shape[2] - shape[0])/2.)
                R_ = R[:, :, idx_buffer:(idx_buffer + shape[0]), :]
                metadata["y1"] += idx_buffer*metadata["ypixelsize"]
                metadata["y2"] -= idx_buffer*metadata["ypixelsize"]

        elif method == "crop":

            if R.shape[2] == shape[0]:
                idx_buffer = int((shape[1] - R.shape[3])/2.)
                R_[:, :, :, idx_buffer:(idx_buffer + R.shape[3])] = R
                metadata["x1"] -= idx_buffer*metadata["xpixelsize"]
                metadata["x2"] += idx_buffer*metadata["xpixelsize"]

            elif R.shape[3] == shape[1]:
                idx_buffer = int((shape[0] - R.shape[2])/2.)
                R_[:, :, idx_buffer:(idx_buffer + R.shape[2]), :] = R
                metadata["y1"] -= idx_buffer*metadata["ypixelsize"]
                metadata["y2"] += idx_buffer*metadata["ypixelsize"]

        R_shape[-2] = R_.shape[-2]
        R_shape[-1] = R_.shape[-1]

        return R_.reshape(R_shape),metadata
