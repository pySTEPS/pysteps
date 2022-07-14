# -*- coding: utf-8 -*-
"""
pysteps.utils.reprojection
==========================

Reprojection tools to reproject grids and adjust the grid cell size of an
input field to a destination field.

.. autosummary::
    :toctree: ../generated/

    reproject_grids
"""
from pysteps.exceptions import MissingOptionalDependency

import numpy as np

try:
    from rasterio import Affine as A
    from rasterio.warp import reproject, Resampling

    RASTERIO_IMPORTED = True
except ImportError:
    RASTERIO_IMPORTED = False


def reproject_grids(src_array, dst_array, metadata_src, metadata_dst):
    """
    Reproject precipitation fields to the domain of another precipitation field.

    Parameters
    ----------
    src_array: array-like
        Three-dimensional array of shape (t, x, y) containing a time series of
        precipitation fields. These precipitation fields will be reprojected.
    dst_array: array-like
        Array containing a precipitation field or a time series of precipitation
        fields. The src_array will be reprojected to the domain of
        dst_array.
    metadata_src: dict
        Metadata dictionary containing the projection, x- and ypixelsize, x1 and
        y2 attributes of the src_array as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    metadata_dst: dict
        Metadata dictionary containing the projection, x- and ypixelsize, x1 and
        y2 attributes of the dst_array.

    Returns
    -------
    r_rprj: array-like
        Three-dimensional array of shape (t, x, y) containing the precipitation
        fields of src_array, but reprojected to the domain of dst_array.
    metadata: dict
        Metadata dictionary containing the projection, x- and ypixelsize, x1 and
        y2 attributes of the reprojected src_array.
    """

    if not RASTERIO_IMPORTED:
        raise MissingOptionalDependency(
            "rasterio package is required for the reprojection module, but it is "
            "not installed"
        )

    # Extract the grid info from src_array
    src_crs = metadata_src["projection"]
    x1_src = metadata_src["x1"]
    y2_src = metadata_src["y2"]
    xpixelsize_src = metadata_src["xpixelsize"]
    ypixelsize_src = metadata_src["ypixelsize"]
    src_transform = A.translation(float(x1_src), float(y2_src)) * A.scale(
        float(xpixelsize_src), float(-ypixelsize_src)
    )

    # Extract the grid info from dst_array
    dst_crs = metadata_dst["projection"]
    x1_dst = metadata_dst["x1"]
    y2_dst = metadata_dst["y2"]
    xpixelsize_dst = metadata_dst["xpixelsize"]
    ypixelsize_dst = metadata_dst["ypixelsize"]
    dst_transform = A.translation(float(x1_dst), float(y2_dst)) * A.scale(
        float(xpixelsize_dst), float(-ypixelsize_dst)
    )

    # Initialise the reprojected array
    r_rprj = np.zeros((src_array.shape[0], dst_array.shape[-2], dst_array.shape[-1]))

    # For every timestep, reproject the precipitation field of src_array to
    # the domain of dst_array
    if metadata_src["yorigin"] != metadata_dst["yorigin"]:
        src_array = src_array[:, ::-1, :]

    for i in range(src_array.shape[0]):
        reproject(
            src_array[i, :, :],
            r_rprj[i, :, :],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=np.nan,
        )

    # Update the metadata
    metadata = metadata_src.copy()

    for key in [
        "projection",
        "yorigin",
        "xpixelsize",
        "ypixelsize",
        "x1",
        "x2",
        "y1",
        "y2",
        "cartesian_unit",
    ]:
        metadata[key] = metadata_dst[key]

    return r_rprj, metadata
