from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import xarray as xr
import numpy as np


def reprojection(R_src, R_dst):
    """Reprojects precipitation fields to the domain of another precipiation
    field.

    Parameters
    ----------
    R_src: xarray
        Three-dimensional xarray with dimensions (t, x, y) containing a
        time series of precipitation fields. These precipitaiton fields
        will be reprojected.
    R_dst: xarray
        Xarray containing a precipitation field or a time series of precipitation
        fields. The xarray R_src will be reprojected to the domain of R_dst.

    Returns
    -------
    R_rprj: xarray
        Three-dimensional xarray with dimensions (t, x, y) containing the
        precipitation fields of R_src, but reprojected to the domain of
        R_dst.
    """

    # Extract the grid info from R_src
    src_crs = R_src.attrs["projection"]
    x_src = R_src.x
    y_src = R_src.y
    x1_src = np.min(x_src)
    y2_src = np.max(y_src)
    xpixelsize_src = R_src.attrs["xpixelsize"]
    ypixelsize_src = R_src.attrs["ypixelsize"]
    src_transform = A.translation(x1_src, y2_src) * A.scale(
        xpixelsize_src, -ypixelsize_src
    )

    # Extract the grid info from R_dst
    dst_crs = R_dst.attrs["projection"]
    x_dst = R_dst.x
    y_dst = R_dst.y
    x1_dst = np.min(x_dst)
    y2_dst = np.max(y_dst)
    xpixelsize_dst = R_dst.attrs["xpixelsize"]
    ypixelsize_dst = R_dst.attrs["ypixelsize"]
    dst_transform = A.translation(x1_dst, y2_dst) * A.scale(
        xpixelsize_dst, -ypixelsize_dst
    )

    # Initialise the reprojected (x)array
    R_rprj = np.zeros((R_src.shape[0], R_dst.shape[-2], R_dst.shape[-1]))

    # For every timestep, reproject the precipitation field of R_src to
    # the domain of R_dst
    for i in range(R_src.shape[0]):
        reproject(
            R_src[i, :, :],
            R_rprj[i, :, :],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=np.nan,
        )

    # Assign the necessary attributes from R_src and R_dst to R_rprj
    R_rprj = xr.DataArray(
        data=R_rprj,
        dims=("t", "y", "x"),
        coords=dict(
            t=("t", R_src.coords["t"].data),
            x=("x", R_dst.coords["x"].data),
            y=("y", R_dst.coords["y"].data),
        ),
    )
    R_rprj.attrs.update(R_src.attrs)
    R_rprj.x.attrs.update(R_dst.x.attrs)
    R_rprj.y.attrs.update(R_dst.y.attrs)
    for key in ["projection", "yorigin", "xpixelsize", "ypixelsize"]:
        R_rprj.attrs[key] = R_dst.attrs[key]

    return R_rprj
