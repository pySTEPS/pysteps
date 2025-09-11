# -*- coding: utf-8 -*-

import os
import numpy as np
import pytest
import xarray as xr

from pysteps.utils import reprojection as rpj
from pysteps.tests.helpers import get_precipitation_fields


def build_precip_dataset(
    data: np.ndarray,  # shape (time, y, x)
    *,
    projection: str = "EPSG:3035",  # PROJ4/EPSG string
    cartesian_unit: str = "m",  # 'm' or 'km'
    institution: str = "rmi",
    precip_var_name: str = "precip_intensity",  # or 'precip_accum' / 'reflectivity'
    # grid + time spec (regular spacing)
    nx: int | None = None,
    ny: int | None = None,
    dx: float = 1000.0,  # x stepsize, in cartesian_unit
    dy: float = 1000.0,  # y stepsize, in cartesian_unit
    x0: float | None = None,  # center of pixel (0,0) in cartesian_unit
    y0: float | None = None,  # center of pixel (0,0) in cartesian_unit
    nt: int | None = None,
    t0: int = 0,  # seconds since forecast start
    dt: int = 3600,  # timestep in seconds
    # precip variable attrs
    units: str = "mm/h",
    transform_attr: str | None = None,  # e.g. 'dB', 'Box-Cox', or None
    accutime_min: float = 60.0,
    threshold: float = 0.1,
    zerovalue: float = 0.0,
    zr_a: float = 200.0,
    zr_b: float = 1.6,
) -> xr.Dataset:
    assert data.ndim == 3, "data must be (time, y, x)"
    nt_ = data.shape[0] if nt is None else nt
    ny_ = data.shape[1] if ny is None else ny
    nx_ = data.shape[2] if nx is None else nx

    # Build regular coords (centers). If x0/y0 are not given, start at half a pixel.
    if x0 is None:
        x0 = 0.5 * dx
    if y0 is None:
        y0 = 0.5 * dy

    x = x0 + dx * np.arange(nx_)
    y = y0 + dy * np.arange(ny_)
    time = t0 + dt * np.arange(nt_)

    da = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": time, "y": y, "x": x},
        name=precip_var_name,
        attrs={
            "units": units,
            "accutime": float(accutime_min),
            "threshold": float(threshold),
            "zerovalue": float(zerovalue),
            "zr_a": float(zr_a),
            "zr_b": float(zr_b),
            **({"transform": transform_attr} if transform_attr is not None else {}),
        },
    )

    # stepsize attrs on coords (required by your spec)
    da.coords["time"].attrs["stepsize"] = int(dt)  # seconds
    da.coords["time"].attrs["standard_name"] = "time"
    da.coords["x"].attrs["stepsize"] = float(dx)  # in cartesian_unit
    da.coords["x"].attrs["units"] = cartesian_unit
    da.coords["y"].attrs["stepsize"] = float(dy)  # in cartesian_unit
    da.coords["y"].attrs["units"] = cartesian_unit

    ds = xr.Dataset({precip_var_name: da})
    ds.attrs.update(
        {
            "projection": projection,  # PROJ string or EPSG code
            "institution": institution,
            "precip_var": precip_var_name,
            "cartesian_unit": cartesian_unit,
        }
    )

    return ds


precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    source="rmi",
    return_raw=True,
    metadata=True,
    log_transform=False,
)

# Initialise dummy NWP data
nwp_array = np.zeros((24, 564, 564))

for t in range(nwp_array.shape[0]):
    nwp_array[t, 30 + t : 185 + t, 30 + 2 * t] = 0.1
    nwp_array[t, 30 + t : 185 + t, 31 + 2 * t] = 0.1
    nwp_array[t, 30 + t : 185 + t, 32 + 2 * t] = 1.0
    nwp_array[t, 30 + t : 185 + t, 33 + 2 * t] = 5.0
    nwp_array[t, 30 + t : 185 + t, 34 + 2 * t] = 5.0
    nwp_array[t, 30 + t : 185 + t, 35 + 2 * t] = 4.5
    nwp_array[t, 30 + t : 185 + t, 36 + 2 * t] = 4.5
    nwp_array[t, 30 + t : 185 + t, 37 + 2 * t] = 4.0
    nwp_array[t, 30 + t : 185 + t, 38 + 2 * t] = 2.0
    nwp_array[t, 30 + t : 185 + t, 39 + 2 * t] = 1.0
    nwp_array[t, 30 + t : 185 + t, 40 + 2 * t] = 0.5
    nwp_array[t, 30 + t : 185 + t, 41 + 2 * t] = 0.1

nwp_proj = (
    "+proj=lcc +lon_0=4.55 +lat_1=50.8 +lat_2=50.8 "
    "+a=6371229 +es=0 +lat_0=50.8 +x_0=365950 +y_0=-365950.000000001"
)

nwp_dataset = build_precip_dataset(
    nwp_array,
    projection=nwp_proj,  # ETRS89 / LAEA Europe (meters)
    cartesian_unit="m",
    precip_var_name="precip_intensity",
    dx=1300.0,
    dy=1300.0,  # 1 km grid
    dt=300,  # hourly
    accutime_min=5.0,  # accumulation window (min)
    threshold=0.1,  # mm/h rain/no-rain threshold
    zerovalue=0.0,
)

steps_arg_names = (
    "precip_dataset",
    "nwp_dataset",
)

steps_arg_values = [
    (precip_dataset, nwp_dataset),
]


# XR: since reproject_grids is not xarray compatible yet, we cannot use xarray DataArrays in the tests
@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_utils_reproject_grids(precip_dataset, nwp_dataset):
    # Reproject
    nwp_dataset_reproj = rpj.reproject_grids(nwp_dataset, precip_dataset)

    nwp_dataset_reproj_dataarray = nwp_dataset_reproj[
        nwp_dataset_reproj.attrs["precip_var"]
    ]
    nwp_dataarray = nwp_dataset[nwp_dataset.attrs["precip_var"]]
    precip_dataarray = precip_dataset[precip_dataset.attrs["precip_var"]]
    # The tests
    assert (
        nwp_dataset_reproj_dataarray.shape[0] == nwp_dataarray.shape[0]
    ), "Time dimension has not the same length as source"
    assert (
        nwp_dataset_reproj_dataarray.shape[1] == precip_dataarray.shape[1]
    ), "y dimension has not the same length as radar composite"
    assert (
        nwp_dataset_reproj_dataarray.shape[2] == precip_dataarray.shape[2]
    ), "x dimension has not the same length as radar composite"

    assert float(nwp_dataset_reproj_dataarray.x.isel(x=0).values) == float(
        precip_dataarray.x.isel(x=0).values
    ), "x-value lower left corner is not equal to radar composite"
    assert float(nwp_dataset_reproj_dataarray.x.isel(x=-1).values) == float(
        precip_dataarray.x.isel(x=-1).values
    ), "x-value upper right corner is not equal to radar composite"
    assert float(nwp_dataset_reproj_dataarray.y.isel(y=0).values) == float(
        precip_dataarray.y.isel(y=0).values
    ), "y-value lower left corner is not equal to radar composite"
    assert float(nwp_dataset_reproj_dataarray.y.isel(y=-1).values) == float(
        precip_dataarray.y.isel(y=-1).values
    ), "y-value upper right corner is not equal to radar composite"

    assert (
        nwp_dataset_reproj.attrs["projection"] == precip_dataset.attrs["projection"]
    ), "projection is different than destination projection"
