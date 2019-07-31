# -*- coding: utf-8 -*-

import datetime
import numpy as np
from numpy.testing import assert_array_almost_equal
import os
import pytest
from pysteps import io, rcparams
from pysteps.verification import probscores
from pysteps.postprocessing.ensemblestats import excprob


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

# CRPS
test_data = [(R[:10], R[-1], 0.01470871)]


@pytest.mark.parametrize("X_f, X_o, expected", test_data)
def test_CRPS(X_f, X_o, expected):
    """Test the CRPS."""
    assert_array_almost_equal(probscores.CRPS(X_f, X_o), expected)


# reldiag
test_data = [(R[:10], R[-1], 1.0, 10, 10, 3.38751492)]


@pytest.mark.parametrize("X_f, X_o, X_min, n_bins, min_count, expected", test_data)
def test_reldiag_sum(X_f, X_o, X_min, n_bins, min_count, expected):
    """Test the reldiag."""
    P_f = excprob(X_f, X_min, ignore_nan=False)
    assert_array_almost_equal(
        np.sum(probscores.reldiag(P_f, X_o, X_min, n_bins, min_count)[1]), expected
    )


# ROC_curve
test_data = [(R[:10], R[-1], 1.0, 10, True, 0.79557329)]


@pytest.mark.parametrize(
    "X_f, X_o, X_min, n_prob_thrs, compute_area, expected", test_data
)
def test_ROC_curve_area(X_f, X_o, X_min, n_prob_thrs, compute_area, expected):
    """Test the ROC_curve."""
    P_f = excprob(X_f, X_min, ignore_nan=False)
    assert_array_almost_equal(
        probscores.ROC_curve(P_f, X_o, X_min, n_prob_thrs, compute_area)[2], expected
    )
