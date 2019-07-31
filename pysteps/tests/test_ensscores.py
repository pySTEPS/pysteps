# -*- coding: utf-8 -*-

import datetime
import numpy as np
from numpy.testing import assert_array_almost_equal
import os
import pytest
from pysteps import io, rcparams
from pysteps.verification import ensscores


def import_mch_gif():
    date = datetime.datetime.strptime("201505151630", "%Y%m%d%H%M")
    data_source = "mch"

    # Load data source config
    root_path = rcparams.data_sources[data_source]["root_path"]
    path_fmt = rcparams.data_sources[data_source]["path_fmt"]
    fn_pattern = rcparams.data_sources[data_source]["fn_pattern"]
    fn_ext = rcparams.data_sources[data_source]["fn_ext"]
    importer_name = rcparams.data_sources[data_source]["importer"]
    importer_kwargs = rcparams.data_sources[data_source]["importer_kwargs"]
    timestep = rcparams.data_sources[data_source]["timestep"]

    # Find the input files from the archive
    fns = io.archive.find_by_date(
        date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_next_files=10
    )

    # Read the radar composites
    importer = io.get_method(importer_name, "importer")
    R, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)

    return R, metadata


R, _ = import_mch_gif()
np.random.seed(42)

# rankhist
test_data = [(R[:10], R[-1], None, True, 11), (R[:10], R[-1], None, False, 11)]


@pytest.mark.parametrize("X_f, X_o, X_min, normalize, expected", test_data)
def test_rankhist_size(X_f, X_o, X_min, normalize, expected):
    """Test the rankhist."""
    assert_array_almost_equal(
        ensscores.rankhist(X_f, X_o, X_min, normalize).size, expected
    )


# ensemble_skill
test_data = [
    (R[:10], R[-1], "RMSE", {"axis": None, "conditioning": "single"}, 0.26054151),
    (R[:10], R[-1], "CSI", {"thr": 1.0, "axis": None}, 0.22017924),
    (R[:10], R[-1], "FSS", {"thr": 1.0, "scale": 10}, 0.63239752),
]


@pytest.mark.parametrize("X_f, X_o, metric, kwargs, expected", test_data)
def test_ensemble_skill(X_f, X_o, metric, kwargs, expected):
    """Test the ensemble_skill."""
    assert_array_almost_equal(
        ensscores.ensemble_skill(X_f, X_o, metric, **kwargs), expected
    )


# ensemble_spread
test_data = [
    (R, "RMSE", {"axis": None, "conditioning": "single"}, 0.22635757),
    (R, "CSI", {"thr": 1.0, "axis": None}, 0.25218158),
    (R, "FSS", {"thr": 1.0, "scale": 10}, 0.70235667),
]


@pytest.mark.parametrize("X_f, metric, kwargs, expected", test_data)
def test_ensemble_spread(X_f, metric, kwargs, expected):
    """Test the ensemble_spread."""
    assert_array_almost_equal(
        ensscores.ensemble_spread(X_f, metric, **kwargs), expected
    )
