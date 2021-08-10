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

    precip_full, _, metadata = pysteps.io.import_mrms_grib(
        filename, fillna=0, window_size=1
    )
    assert precip_full.shape == (3500, 7000)
    assert precip_full.dtype == "single"

    expected_metadata = {
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
    assert y.size == precip_full.shape[0]
    assert x.size == precip_full.shape[1]

    # The full latitude range is (20.005, 54.995)
    # The full longitude range is (230.005, 299.995)

    # Test that if the bounding box is larger than the domain, all the points are returned.
    precip_full2 = pysteps.io.import_mrms_grib(
        filename, fillna=0, extent=(220, 300, 20, 55), window_size=1
    )[0]
    assert precip_full2.shape == (3500, 7000)

    assert_array_almost_equal(precip_full, precip_full2)

    del precip_full2

    # Test that a portion of the domain is returned correctly
    precip_clipped = pysteps.io.import_mrms_grib(
        filename, fillna=0, extent=(250, 260, 30, 35), window_size=1
    )[0]

    assert precip_clipped.shape == (500, 1000)
    assert_array_almost_equal(precip_clipped, precip_full[2000:2500, 2000:3000])
    del precip_clipped

    precip_single = pysteps.io.import_mrms_grib(filename, dtype="double", fillna=0)[0]
    assert precip_single.dtype == "double"
    del precip_single

    precip_single = pysteps.io.import_mrms_grib(filename, dtype="single", fillna=0)[0]
    assert precip_single.dtype == "single"
    del precip_single

    precip_donwscaled = pysteps.io.import_mrms_grib(
        filename, dtype="single", fillna=0, window_size=2
    )[0]
    assert precip_donwscaled.shape == (3500 / 2, 7000 / 2)

    precip_donwscaled, _, metadata = pysteps.io.import_mrms_grib(
        filename, dtype="single", fillna=0, window_size=3
    )
    print(metadata)
    expected_metadata.update(
        xpixelsize=0.03,
        ypixelsize=0.03,
        x1=-130.00000000028575,
        x2=-60.01000199942843,
        y1=20.02000099914261,
        y2=55.000000000285794,
    )

    # Remove the threshold keyword from the test when the window_size>1 is used.
    # The threshold is computed automatically from the minimum valid precipitation values.
    # The minimum value is affected by the the block_averaging (window_size=3 keyword)
    # of the precipitation fields. Hence, the "threshold" value will depend on the
    # precipitation pattern (i.e. the file being read).
    expected_metadata.pop("threshold", None)

    for key in expected_metadata.keys():
        assert metadata[key] == expected_metadata[key]
    assert precip_donwscaled.shape == (3500 // 3, 7000 // 3)
