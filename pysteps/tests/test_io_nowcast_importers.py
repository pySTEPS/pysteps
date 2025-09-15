import numpy as np
import pytest

from tempfile import NamedTemporaryFile
from pysteps import io
from pysteps.tests.helpers import get_precipitation_fields
import xarray as xr

precip_dataset = get_precipitation_fields(
    num_prev_files=1,
    num_next_files=0,
    return_raw=False,
    metadata=True,
    upscale=2000,
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]

zeros = np.zeros(precip_dataarray.shape, dtype=np.float32)

zero_dataset = xr.Dataset(
    data_vars={
        "precip_intensity": (
            ("time", "y", "x"),
            zeros,
            {
                "long_name": "Precipitation intensity",
                "units": "mm hr-1",  # keep attrs simple types, no None
                "_FillValue": np.float32(-9999),  # valid NetCDF fill value
                # omit standard_name unless you have a CF-valid value
            },
        )
    },
    coords={
        "time": ("time", precip_dataarray["time"].values),
        "y": ("y", precip_dataarray["y"].values),
        "x": ("x", precip_dataarray["x"].values),
    },
    attrs={"precip_var": "precip_intensity"},  # simple, serializable globals
)


@pytest.mark.parametrize(
    "precip_dataset",
    [(precip_dataset), (zero_dataset)],
)
def test_import_netcdf(precip_dataset):
    # XR: this test might not make that much sense in the future
    with NamedTemporaryFile() as tempfile:
        precip_var = precip_dataset.attrs["precip_var"]
        precip_dataarray = precip_dataset[precip_var]
        field_shape = (precip_dataarray.shape[1], precip_dataarray.shape[2])

        precip_dataset.to_netcdf(tempfile.name)
        precip_netcdf = io.import_netcdf_pysteps(tempfile.name, dtype="float64")

        assert isinstance(precip_netcdf, xr.Dataset)
        assert (
            precip_netcdf[precip_var].ndim == precip_dataarray.ndim
        ), "Wrong number of dimensions"
        assert (
            precip_netcdf[precip_var].shape[0] == precip_dataarray.shape[0]
        ), "Wrong number of lead times"
        assert precip_netcdf[precip_var].shape[1:] == field_shape, "Wrong field shape"
        assert np.allclose(precip_netcdf[precip_var].values, precip_dataarray.values)
