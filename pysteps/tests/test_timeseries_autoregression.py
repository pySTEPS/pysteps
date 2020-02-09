# -*- coding: utf-8 -*-

import os

import pytest

import pysteps
from pysteps.timeseries import autoregression


def test_estimate_ar_params_ols():
    R = _read_data()


def _read_data():
    root_path = pysteps.rcparams.data_sources["fmi"]["root_path"]
    filename = os.path.join(root_path, "20160928",
                            "201609281600_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz")
    R, _, _ = pysteps.io.import_fmi_pgm(filename, gzipped=True)

    return R
