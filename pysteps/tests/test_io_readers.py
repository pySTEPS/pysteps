from datetime import datetime

import numpy as np

import xarray as xr
from pysteps.tests.helpers import get_precipitation_fields


def test_read_timeseries_mch():
    precip_dataset = get_precipitation_fields(
        num_prev_files=1,
        num_next_files=1,
        return_raw=True,
        source="mch",
        log_transform=False,
    )

    precip_var = precip_dataset.attrs["precip_var"]
    precip_dataarray = precip_dataset[precip_var]

    assert isinstance(precip_dataset, xr.Dataset)
    assert precip_dataarray.shape[0] == 3
