# -*- coding: utf-8 -*-
"""
pysteps.utils.reprojection
==================

Reprojection tool to reproject the grid and adjust the grid cell size of an 
input field to a destination field.

.. autosummary::
    :toctree: ../generated/

    reprojection
"""
from pysteps.exceptions import MissingOptionalDependency
import xarray as xr
import numpy as np

try:
    import rasterio
    from rasterio import Affine as A
    from rasterio.warp import reproject, Resampling

    RASTERIO_IMPORTED = True
except ImportError:
    RASTERIO_IMPORTED = False


def reprojection(src_array, dst_array):
    """Reprojects precipitation fields to the domain of another precipitation
    field.

    Parameters
    ----------
    src_array: xr.DataArray
        Three-dimensional xarray DataArray with dimensions (t, x, y) containing
        a time series of precipitation fields. These precipitation fields
        will be reprojected.
    dst_array: xr.DataArray
        Xarray DataArray containing a precipitation field or a time series of
        precipitation fields. The xarray src_array will be reprojected to the
        domain of dst_array.

    Returns
    -------
    r_rprj: xr.DataArray
        Three-dimensional xarray DataArray with dimensions (t, x, y) containing
        the precipitation fields of src_array, but reprojected to the domain of
        dst_array.
    """

    if not RASTERIO_IMPORTED:
        raise MissingOptionalDependency(
            "rasterio package is required for the reprojection tool, but it is"
            "not installed"
        )

    # Extract the grid info from src_array
    src_crs = src_array.attrs["projection"]
    x1_src = src_array.x.attrs["x1"]
    y2_src = src_array.y.attrs["y2"]
    xpixelsize_src = src_array.attrs["xpixelsize"]
    ypixelsize_src = src_array.attrs["ypixelsize"]
    src_transform = A.translation(float(x1_src), float(y2_src)) * A.scale(
        float(xpixelsize_src), float(-ypixelsize_src)
    )

    # Extract the grid info from dst_array
    dst_crs = dst_array.attrs["projection"]
    x1_dst = dst_array.x.attrs["x1"]
    y2_dst = dst_array.y.attrs["y2"]
    xpixelsize_dst = dst_array.attrs["xpixelsize"]
    ypixelsize_dst = dst_array.attrs["ypixelsize"]
    dst_transform = A.translation(float(x1_dst), float(y2_dst)) * A.scale(
        float(xpixelsize_dst), float(-ypixelsize_dst)
    )

    # Initialise the reprojected (x)array
    r_rprj = np.zeros((src_array.shape[0], dst_array.shape[-2], dst_array.shape[-1]))

    # For every timestep, reproject the precipitation field of src_array to
    # the domain of dst_array
    if src_array.attrs["yorigin"] != dst_array.attrs["yorigin"]:
        src_array = src_array[:, ::-1, :]

    for i in range(src_array["t"].shape[0]):
        reproject(
            src_array.isel(t=i).values,
            r_rprj[i, :, :],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=np.nan,
        )

    # Assign the necessary attributes from src_array and dst_array to R_rprj
    r_rprj = xr.DataArray(
        data=r_rprj,
        dims=("t", "y", "x"),
        coords=dict(
            t=("t", src_array.coords["t"].data),
            x=("x", dst_array.coords["x"].data),
            y=("y", dst_array.coords["y"].data),
        ),
    )

    # xr.apply_ufunc(reproject, src_array, input_core_dims=[["y", "x"]], output_core_dims=[["y", "x"]],
    #                 kwargs={'destination': R_rprj, 'src_transform': src_transform, 'src_crs': src_crs, 'dst_transform': dst_transform, 'dst_crs': dst_crs, 'resampling': Resampling.nearest, 'dst_nodata': np.nan},
    #                 dask='allowed',
    #                 exclude_dims=set(("y","x",)),
    #                 vectorize=True,)

    r_rprj.attrs.update(src_array.attrs)
    r_rprj.x.attrs.update(dst_array.x.attrs)
    r_rprj.y.attrs.update(dst_array.y.attrs)
    for key in ["projection", "yorigin", "xpixelsize", "ypixelsize"]:
        r_rprj.attrs[key] = dst_array.attrs[key]

    return r_rprj
