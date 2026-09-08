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

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from pysteps.exceptions import MissingOptionalDependency

try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False

try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False


def reproject_grids(src_dataset, dst_dataset):
    """
    Reproject precipitation fields to the domain of another precipitation field.

    Parameters
    ----------
    src_dataset: xr.Dataset
        xr.Dataset containing a precipitation variable which needs to be reprojected
    dst_dataset: xr.Dataset
        xr.Dataset containing a precipitation variable which is used to project the provided src_dataset

    Returns
    -------
    reprojected_dataset: xr.Dataset
        xr.Dataset containing the reprojected precipitation variable
    """

    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required for the reprojection module, but it is "
            "not installed"
        )

    x_r = dst_dataset.x.values
    y_r = dst_dataset.y.values
    x_2d, y_2d = np.meshgrid(x_r, y_r)
    # Calculate match  between the two projections
    transfomer = pyproj.Transformer.from_proj(
        src_dataset.attrs["projection"], dst_dataset.attrs["projection"]
    )
    dest_src_x, dest_src_y = transfomer.transform(
        x_2d.flatten(), y_2d.flatten(), direction="INVERSE"
    )
    dest_src_x, dest_src_y = (
        dest_src_x.reshape(x_2d.shape),
        dest_src_y.reshape(y_2d.shape),
    )
    dest_src_x_dataarray = xr.DataArray(
        dest_src_x, dims=("y_src", "x_src"), coords={"y_src": y_r, "x_src": x_r}
    )
    dest_src_y_dataarray = xr.DataArray(
        dest_src_y, dims=("y_src", "x_src"), coords={"y_src": y_r, "x_src": x_r}
    )
    # Select the nearest neighbour in the source dataset for each point in the destination dataset
    reproj_dataset = src_dataset.sel(
        x=dest_src_x_dataarray, y=dest_src_y_dataarray, method="nearest"
    )
    # Clean up the dataset
    reproj_dataset = reproj_dataset.drop_vars(["x", "y"])
    reproj_dataset = reproj_dataset.rename({"x_src": "x", "y_src": "y"})
    # Fill attributes from dst_dataset to reproj_dataset
    reproj_dataset.attrs = dst_dataset.attrs
    reproj_dataset[reproj_dataset.attrs["precip_var"]].attrs = dst_dataset[
        dst_dataset.attrs["precip_var"]
    ].attrs

    return reproj_dataset


def unstructured2regular(src_array, metadata_src, metadata_dst):
    """
    Reproject unstructured data onto a regular grid on the assumption that
    both src data and dst grid have the same projection.

    Parameters
    ----------
    src_array: np.ndarray
        Three-dimensional array of shape (t, n_ens, n_gridcells) containing a
        time series of precipitation ensemble forecasts. These precipitation
        fields will be reprojected.
    metadata_src: dict
        Metadata dictionary containing the projection, clon, clat, and ngridcells
        and attributes of the src_array as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    metadata_dst: dict
        Metadata dictionary containing the projection, x- and ypixelsize, x1 and
        y2 attributes of the dst_array.

    Returns
    -------
    tuple
        A tuple containing:
        - r_rprj: np.ndarray
            Four dimensional array of shape (t, n_ens, x, y) containing the
            precipitation fields of src_array, but reprojected to the grid
            of dst_array.
        - metadata: dict
            Dictionary containing geospatial metadat such as:
            - 'projection' : PROJ.4 string defining the stereographic projection.
            - 'xpixelsize', 'ypixelsize': Pixel size in meters.
            - 'x1', 'y1': Carthesian coordinates of the lower-left corner.
            - 'x2', 'y2': Carthesian coordinates of the upper-right corner.
            - 'cartesian_unit': Unit of the coordinate system (meters).
    """

    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required to reproject DWD's NWP data"
            "but it is not installed"
        )

    if not "clon" in metadata_src.keys():
        raise KeyError("Center longitude (clon) is missing in metadata_src")
    if not "clat" in metadata_src.keys():
        raise KeyError("Center latitude (clat) is missing in metadata_src")

    # Get number of grid cells
    Nc = metadata_src["clon"].shape[0]
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

    # Create destination grid
    if metadata_dst["yorigin"] == "upper":
        y_dst = y_dst[::-1]
    xx_dst, yy_dst = np.meshgrid(x_dst, y_dst)
    s_out = yy_dst.shape

    # Extract the grid info of src_array assuming the same projection of src and dst
    pr = pyproj.Proj(metadata_dst["projection"])
    x_src, y_src = pr(metadata_src["clon"], metadata_src["clat"])

    # Create array of x-y pairs for interpolation
    P_in = np.stack((x_src, y_src)).T
    P_out = np.array((xx_dst.flatten(), yy_dst.flatten())).T

    # Nearest neighbor interpolation of x-y pairs
    ic_out = (
        griddata(P_in, ic_in.flatten(), P_out, method="nearest")
        .reshape(s_out)
        .astype(int)
    )

    # Apply interpolation on all time steps and ensemble members
    r_rprj = np.array(
        [
            [src_array[i, j][ic_out] for j in range(src_array.shape[1])]
            for i in range(src_array.shape[0])
        ]
    )

    # Update the src metadata
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
