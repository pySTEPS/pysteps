# -*- coding: utf-8 -*-

import os
import xarray as xr
import pytest
import pysteps
from pysteps.utils import reprojection

root_path_radar = pysteps.rcparams.data_sources["knmi"]["root_path"]
root_path_nwp = pysteps.rcparams.data_sources["knmi_nwp"]["root_path"]

rel_path_nwp = os.path.join("2018", "09", "05")
rel_path_radar = os.path.join(
    "2010", "08"
)  # Different day, but that does not matter for the tester

filename_nwp = os.path.join(
    root_path_nwp, rel_path_nwp, "20180905_0600_Pforecast_Harmonie.nc"
)
filename_radar = os.path.join(
    root_path_radar, rel_path_radar, "RAD_NL25_RAP_5min_201008260230.h5"
)

# Open the radar and NWP data
radar_array_xr = pysteps.io.importers.import_knmi_hdf5(filename_radar)
nwp_array_xr = pysteps.io.import_knmi_nwp_xr(filename_nwp)

steps_arg_names = (
    "radar_array_xr",
    "nwp_array_xr",
    "t",
)

steps_arg_values = [
    (radar_array_xr, nwp_array_xr, 1),
    (radar_array_xr, nwp_array_xr, 4),
    (radar_array_xr, nwp_array_xr, 8),
]


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_utils_reprojection(
    radar_array_xr,
    nwp_array_xr,
    t,
):

    # Reproject
    nwp_array_xr_reproj = reprojection(nwp_array_xr, radar_array_xr)

    # The tests
    assert (
        nwp_array_xr_reproj["t"].shape[0] == nwp_array_xr["t"].shape[0]
    ), "Time dimension has not the same length as source"
    assert (
        nwp_array_xr_reproj["y"].shape[0] == radar_array_xr["y"].shape[0]
    ), "y dimension has not the same length as radar composite"
    assert (
        nwp_array_xr_reproj["x"].shape[0] == radar_array_xr["x"].shape[0]
    ), "x dimension has not the same length as radar composite"

    assert (
        nwp_array_xr_reproj.x.attrs["x1"] == radar_array_xr.x.attrs["x1"]
    ), "x-value lower left corner is not equal to radar composite"
    assert (
        nwp_array_xr_reproj.x.attrs["x2"] == radar_array_xr.x.attrs["x2"]
    ), "x-value upper right corner is not equal to radar composite"
    assert (
        nwp_array_xr_reproj.y.attrs["y1"] == radar_array_xr.y.attrs["y1"]
    ), "y-value lower left corner is not equal to radar composite"
    assert (
        nwp_array_xr_reproj.y.attrs["y2"] == radar_array_xr.y.attrs["y2"]
    ), "y-value upper right corner is not equal to radar composite"

    assert (
        nwp_array_xr_reproj.x.min().values == radar_array_xr.x.min().values
    ), "First x-coordinate does not equal first x-coordinate radar composite"
    assert (
        nwp_array_xr_reproj.y.min().values == radar_array_xr.y.min().values
    ), "First y-coordinate does not equal first y-coordinate radar composite"
    assert (
        nwp_array_xr_reproj.x.max().values == radar_array_xr.x.max().values
    ), "Last x-coordinate does not equal last x-coordinate radar composite"
    assert (
        nwp_array_xr_reproj.y.max().values == radar_array_xr.y.max().values
    ), "Last y-coordinate does not equal last y-coordinate radar composite"

    assert (
        nwp_array_xr_reproj.attrs["projection"] == radar_array_xr.attrs["projection"]
    ), "projection is different than destionation projection"
