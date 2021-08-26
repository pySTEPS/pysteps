from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import xarray as xr
import numpy as np


def reprojection(R_src, R_dst):
    src_crs = R_src.attrs["projection"]
    x_src = R_src.x
    y_src = R_src.y
    x1_src = x_src.x1
    y2_src = y_src.y2
    xpixelsize_src = R_src.attrs["xpixelsize"]
    ypixelsize_src = R_src.attrs["ypixelsize"]
    src_transform = A.translation(x1_src, y2_src) * A.scale(
        xpixelsize_src, -ypixelsize_src
    )

    dst_crs = R_dst.attrs["projection"]
    x_dst = R_dst.x
    y_dst = R_dst.y
    x1_dst = x_dst.x1
    y2_dst = y_dst.y2
    xpixelsize_dst = R_dst.attrs["xpixelsize"]
    ypixelsize_dst = R_dst.attrs["ypixelsize"]
    dst_transform = A.translation(x1_dst, y2_dst) * A.scale(
        xpixelsize_dst, -ypixelsize_dst
    )

    R_rprj = np.zeros_like(R_dst[:])

    reproject(
        R_src[:],
        R_rprj,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        dst_nodata=np.nan,
    )

    R_rprj = xr.DataArray(
        data=R_rprj,
        dims=("y", "x"),
        coords=dict(
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
