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
import xarray as xr

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
    dest_src_x, dest_src_y = dest_src_x.reshape(x_2d.shape), dest_src_y.reshape(
        y_2d.shape
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
