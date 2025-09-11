from datetime import datetime

import numpy as np
import pytest

import pysteps

from pysteps.tests.helpers import get_precipitation_fields


def test_read_timeseries_mch():
    precip_dataset = get_precipitation_fields(
        num_prev_files=1,
        num_next_files=1,
        return_raw=True,
        metadata=True,
        source="mch",
        log_transform=False,
    )

    precip_var = precip_dataset.attrs["precip_var"]
    precip_dataarray = precip_dataset[precip_var]

    importer = pysteps.io.get_method(importer_name, "importer")
    precip, _, metadata = pysteps.io.read_timeseries(fns, importer, **importer_kwargs)

    assert isinstance(precip, np.ndarray)
    assert isinstance(metadata, dict)
    assert precip.shape[0] == 3
