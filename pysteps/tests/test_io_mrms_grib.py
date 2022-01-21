# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pysteps

pytest.importorskip("pygrib")


def test_io_import_mrms_grib():
    """Test the importer for NSSL data."""

    root_path = pysteps.rcparams.data_sources["mrms"]["root_path"]
    filename = os.path.join(
        root_path, "2019/06/10/", "PrecipRate_00.00_20190610-000000.grib2"
    )
    precip, _, metadata = pysteps.io.import_mrms_grib(filename, fillna=0, window_size=1)

    assert precip.shape == (3500, 7000)
    assert precip.dtype == "single"

    expected_metadata = {
        "institution": "NOAA National Severe Storms Laboratory",
        "xpixelsize": 0.01,
        "ypixelsize": 0.01,
        "unit": "mm/h",
        "transform": None,
        "zerovalue": 0,
        "projection": "+proj=longlat  +ellps=IAU76",
        "yorigin": "upper",
        "threshold": 0.1,
        "x1": -129.99999999999997,
        "x2": -60.00000199999991,
        "y1": 20.000001,
        "y2": 55.00000000000001,
        "cartesian_unit": "degrees",
    }

    for key, value in expected_metadata.items():
        if isinstance(value, float):
            assert_array_almost_equal(metadata[key], expected_metadata[key])
        else:
            assert metadata[key] == expected_metadata[key]

    x = np.arange(metadata["x1"], metadata["x2"], metadata["xpixelsize"])
    y = np.arange(metadata["y1"], metadata["y2"], metadata["ypixelsize"])

    assert y.size == precip.shape[0]
    assert x.size == precip.shape[1]

    # The full latitude range is (20.005, 54.995)
    # The full longitude range is (230.005, 299.995)

    # Test that if the bounding box is larger than the domain, all the points are returned.
    precip2, _, _ = pysteps.io.import_mrms_grib(
        filename, fillna=0, extent=(220, 300, 20, 55), window_size=1
    )
    assert precip2.shape == (3500, 7000)

    assert_array_almost_equal(precip, precip2)

    del precip2

    # Test that a portion of the domain is returned correctly
    precip3, _, _ = pysteps.io.import_mrms_grib(
        filename, fillna=0, extent=(250, 260, 30, 35), window_size=1
    )

    assert precip3.shape == (500, 1000)
    assert_array_almost_equal(precip3, precip[2000:2500, 2000:3000])
    del precip3

    precip4, _, _ = pysteps.io.import_mrms_grib(filename, dtype="double", fillna=0)
    assert precip4.dtype == "double"
    del precip4

    precip5, _, _ = pysteps.io.import_mrms_grib(filename, dtype="single", fillna=0)
    assert precip5.dtype == "single"
    del precip5
