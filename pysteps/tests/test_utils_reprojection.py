# -*- coding: utf-8 -*-

import os
import numpy as np
import pytest
import pysteps
from pysteps.utils import reprojection as rpj

pytest.importorskip("rasterio")

root_path_radar = pysteps.rcparams.data_sources["rmi"]["root_path"]

rel_path_radar = "20210704"  # Different date, but that does not matter for the tester

filename_radar = os.path.join(
    root_path_radar, rel_path_radar, "20210704180500.rad.best.comp.rate.qpe.hdf"
)

# Open the radar data
radar_array, _, metadata_dst = pysteps.io.importers.import_odim_hdf5(filename_radar)

# Initialise dummy NWP data
nwp_array = np.zeros((24, 564, 564))

for t in range(nwp_array.shape[0]):
    nwp_array[t, 30 + t : 185 + t, 30 + 2 * t] = 0.1
    nwp_array[t, 30 + t : 185 + t, 31 + 2 * t] = 0.1
    nwp_array[t, 30 + t : 185 + t, 32 + 2 * t] = 1.0
    nwp_array[t, 30 + t : 185 + t, 33 + 2 * t] = 5.0
    nwp_array[t, 30 + t : 185 + t, 34 + 2 * t] = 5.0
    nwp_array[t, 30 + t : 185 + t, 35 + 2 * t] = 4.5
    nwp_array[t, 30 + t : 185 + t, 36 + 2 * t] = 4.5
    nwp_array[t, 30 + t : 185 + t, 37 + 2 * t] = 4.0
    nwp_array[t, 30 + t : 185 + t, 38 + 2 * t] = 2.0
    nwp_array[t, 30 + t : 185 + t, 39 + 2 * t] = 1.0
    nwp_array[t, 30 + t : 185 + t, 40 + 2 * t] = 0.5
    nwp_array[t, 30 + t : 185 + t, 41 + 2 * t] = 0.1

nwp_proj = (
    "+proj=lcc +lon_0=4.55 +lat_1=50.8 +lat_2=50.8 "
    "+a=6371229 +es=0 +lat_0=50.8 +x_0=365950 +y_0=-365950.000000001"
)

metadata_src = dict(
    projection=nwp_proj,
    institution="Royal Meteorological Institute of Belgium",
    transform=None,
    zerovalue=0.0,
    threshold=0,
    unit="mm",
    accutime=5,
    xpixelsize=1300.0,
    ypixelsize=1300.0,
    yorigin="upper",
    cartesian_unit="m",
    x1=0.0,
    x2=731900.0,
    y1=-731900.0,
    y2=0.0,
)

steps_arg_names = (
    "radar_array",
    "nwp_array",
    "metadata_src",
    "metadata_dst",
)

steps_arg_values = [
    (radar_array, nwp_array, metadata_src, metadata_dst),
]


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_utils_reproject_grids(
    radar_array,
    nwp_array,
    metadata_src,
    metadata_dst,
):

    # Reproject
    nwp_array_reproj, metadata_reproj = rpj.reproject_grids(
        nwp_array, radar_array, metadata_src, metadata_dst
    )

    # The tests
    assert (
        nwp_array_reproj.shape[0] == nwp_array.shape[0]
    ), "Time dimension has not the same length as source"
    assert (
        nwp_array_reproj.shape[1] == radar_array.shape[0]
    ), "y dimension has not the same length as radar composite"
    assert (
        nwp_array_reproj.shape[2] == radar_array.shape[1]
    ), "x dimension has not the same length as radar composite"

    assert (
        metadata_reproj["x1"] == metadata_dst["x1"]
    ), "x-value lower left corner is not equal to radar composite"
    assert (
        metadata_reproj["x2"] == metadata_dst["x2"]
    ), "x-value upper right corner is not equal to radar composite"
    assert (
        metadata_reproj["y1"] == metadata_dst["y1"]
    ), "y-value lower left corner is not equal to radar composite"
    assert (
        metadata_reproj["y2"] == metadata_dst["y2"]
    ), "y-value upper right corner is not equal to radar composite"

    assert (
        metadata_reproj["projection"] == metadata_dst["projection"]
    ), "projection is different than destionation projection"
