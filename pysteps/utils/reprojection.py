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
from scipy.interpolate import griddata

import numpy as np
import xarray as xr

try:
    from rasterio import Affine as A
    from rasterio.warp import reproject, Resampling

    RASTERIO_IMPORTED = True
except ImportError:
    RASTERIO_IMPORTED = False

try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False


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


def unstructured2regular(src_array, metadata_src, metadata_dst):
    """
    Reproject unstructured data onto a regular grid.

    Parameters
    ----------
    src_array: xarray
        xarray of shape (t, n_ens, ngridcells) containing a time
        series of precipitation enesemble forecasts. These precipitation fields
        will be reprojected.
    metadata_src: dict
        Metadata dictionary containing the projection, clon, clat, and ngridcells
        and attributes of the src_array as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    metadata_dst: dict
        Metadata dictionary containing the projection, x- and ypixelsize, x1 and
        y2 attributes of the dst_array.

    Returns
    -------
    r_rprj: xarray
        Three-dimensional array of shape (t, n_ens, x, y) containing the
        precipitation fields of src_array, but reprojected to the domain
        of dst_array.
    metadata: dict
        Metadata dictionary containing the projection, x- and ypixelsize, x1 and
        y2 attributes of the reprojected src_array.
    """

    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required to reproject DWD's NWP data"
            "but it is not installed"
        )

    # Get number of grid cells
    Nc = src_array["longitude"].shape[0]
    ic_in = np.arange(Nc)

    # Get cartesian coordinates of destination grid
    x_dst = np.arange(
        np.float32(metadata_dst["x1"]),
        np.float32(metadata_dst["x2"]),
        metadata_dst["xpixelsize"],
    )

    y_dst = np.arange(
        np.float32(metadata_dst["y1"]),
        np.float32(metadata_dst["y2"]),
        metadata_dst["ypixelsize"],
    )

    if metadata_dst["yorigin"] == "upper":
        y_dst = y_dst[::-1]
    xx_dst, yy_dst = np.meshgrid(x_dst, y_dst)
    s_out = yy_dst.shape
    P_out = np.array((xx_dst.flatten(), yy_dst.flatten())).T

    # Extract the grid info from src_array assuming the same projection of src and dst
    pr = pyproj.Proj(metadata_dst["projection"])
    x_src, y_src = pr(src_array["longitude"].values, src_array["latitude"].values)
    P_in = np.stack((x_src, y_src)).T

    ic_out = (
        griddata(P_in, ic_in.flatten(), P_out, method="nearest")
        .reshape(s_out)
        .astype(int)
    )

    data_rprj = np.array(
        [
            [src_array[i, j].values[ic_out] for j in range(src_array.shape[1])]
            for i in range(src_array.shape[0])
        ]
    )
    dims = ["time", "ens_no", "south_north", "west_east"]
    lon, lat = pr(xx_dst, yy_dst, inverse=True)
    coords = {
        "time": src_array["time"],
        "ens_no": src_array["ens_no"],
        "west_east": np.arange(0, len(x_dst), 1),
        "south_north": np.arange(0, len(y_dst), 1),
        "x": ("west_east", np.arange(0, len(x_dst), 1), {"units": "1"}),
        "y": ("south_north", np.arange(0, len(y_dst), 1), {"units": "1"}),
        "projection_x_coordinate": ("west_east", x_dst, {"units": "m"}),
        "projection_y_coordinate": ("south_north", y_dst, {"units": "m"}),
        "longitude": (
            ["south_north", "west_east"],
            lon,
            {"units": "degrees_north"},
        ),
        "latitude": (["south_north", "west_east"], lat, {"units": "degrees_east"}),
    }
    xr_rprj = xr.DataArray(data=data_rprj, dims=dims, coords=coords)

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

    return xr_rprj, metadata
