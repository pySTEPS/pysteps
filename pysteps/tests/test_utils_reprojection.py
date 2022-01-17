# -*- coding: utf-8 -*-

import os
import pytest
import pysteps
from pysteps.utils import reprojection

pytest.importorskip("rasterio")

root_path_radar = pysteps.rcparams.data_sources["rmi"]["root_path"]
root_path_nwp = pysteps.rcparams.data_sources["rmi_nwp"]["root_path"]

rel_path_nwp = os.path.join("2021", "07", "04")
rel_path_radar = "20210704"  # Different date, but that does not matter for the tester

filename_nwp = os.path.join(
    root_path_nwp, rel_path_nwp, "ao13_2021070412_native_5min.nc"
)
filename_radar = os.path.join(
    root_path_radar, rel_path_radar, "20210704180500.rad.best.comp.rate.qpe.hdf"
)

# Open the radar and NWP data
radar_array, _, metadata_src = pysteps.io.importers.import_odim_hdf5(filename_radar)
nwp_array, metadata_dst = pysteps.io.import_rmi_nwp_xr(filename_nwp)

steps_arg_names = (
    "radar_array",
    "nwp_array",
    "metadata_src",
)

steps_arg_values = [
    (radar_array, nwp_array, metadata_src, metadata_dst),
]


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_utils_reprojection(
    radar_array,
    nwp_array,
    metadata_src,
    metadata_dst,
):

    # Reproject
    nwp_array_reproj, metadata_reproj = reprojection(
        nwp_array, radar_array, metadata_src, metadata_dst
    )

    # The tests
    assert (
        nwp_array_reproj.shape[0] == nwp_array.shape[0]
    ), "Time dimension has not the same length as source"
    assert (
        nwp_array_reproj.shape[1] == radar_array.shape[1]
    ), "y dimension has not the same length as radar composite"
    assert (
        nwp_array_reproj.shape[2] == radar_array.shape[2]
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
