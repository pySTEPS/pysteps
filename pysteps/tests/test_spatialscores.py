# -*- coding: utf-8 -*-

import datetime
import numpy as np
from numpy.testing import assert_array_almost_equal
import os
import pytest
from pysteps import io, rcparams
from pysteps.verification import spatialscores

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
    fns = io.archive.find_by_date(date, root_path, path_fmt, fn_pattern, fn_ext,
                                  timestep=5, num_prev_files=1)
                      
    # Read the radar composites
    importer = io.get_method(importer_name, "importer")
    R, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)
    
    return R, metadata
    
R, _  = import_mch_gif()
test_data = [
    (R[0], R[1], "FSS", [1], [10], None, 0.85161531),
    (R[0], R[1], "BMSE", [1], None, "Haar", 0.99989651),
]

@pytest.mark.parametrize("X_f, X_o, name, thrs, scales, wavelet, expected", test_data)
def test_intensity_scale(X_f, X_o, name, thrs, scales, wavelet, expected):
    """Test the intensity_scale."""
    assert_array_almost_equal(
        spatialscores.intensity_scale(X_f, X_o, name, thrs, scales, wavelet)[0][0], expected
    )