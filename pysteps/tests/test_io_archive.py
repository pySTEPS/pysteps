from datetime import datetime

import pytest

import pysteps


def test_find_by_date_mch():

    pytest.importorskip("PIL")

    date = datetime.strptime("201505151630", "%Y%m%d%H%M")
    data_source = pysteps.rcparams.data_sources["mch"]
    root_path = data_source["root_path"]
    path_fmt = data_source["path_fmt"]
    fn_pattern = data_source["fn_pattern"]
    fn_ext = data_source["fn_ext"]
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

    assert len(fns) == 2
    assert len(fns[0]) == 3
    assert len(fns[1]) == 3
    assert isinstance(fns[0][0], str)
    assert isinstance(fns[1][0], datetime)
