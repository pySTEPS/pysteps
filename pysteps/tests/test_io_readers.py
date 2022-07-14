from datetime import datetime

import numpy as np
import pytest

import pysteps


def test_read_timeseries_mch():

    pytest.importorskip("PIL")

    date = datetime.strptime("201505151630", "%Y%m%d%H%M")
    data_source = pysteps.rcparams.data_sources["mch"]
    root_path = data_source["root_path"]
    path_fmt = data_source["path_fmt"]
    fn_pattern = data_source["fn_pattern"]
    fn_ext = data_source["fn_ext"]
    importer_name = data_source["importer"]
    importer_kwargs = data_source["importer_kwargs"]
    timestep = data_source["timestep"]

    fns = pysteps.io.archive.find_by_date(
        date,
        root_path,
        path_fmt,
        fn_pattern,
        fn_ext,
        timestep=timestep,
        num_prev_files=1,
        num_next_files=1,
    )

    importer = pysteps.io.get_method(importer_name, "importer")
    precip, _, metadata = pysteps.io.read_timeseries(fns, importer, **importer_kwargs)

    assert isinstance(precip, np.ndarray)
    assert isinstance(metadata, dict)
    assert precip.shape[0] == 3
