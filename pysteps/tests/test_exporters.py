# -*- coding: utf-8 -*-

from datetime import datetime

from pysteps.io.exporters import _get_geotiff_filename
from pysteps.io.exporters import initialize_forecast_exporter_netcdf
from pysteps.io.exporters import export_forecast_dataset
from pysteps.io.exporters import close_forecast_files

import numpy as np
import os
import pysteps
import pytest
import tempfile



def test_get_geotiff_filename():
    """Test the geotif name generator."""

    start_date = datetime.strptime("201909082022", "%Y%m%d%H%M")

    n_timesteps = 50
    timestep = 5

    for timestep_index in range(n_timesteps):
        file_name = _get_geotiff_filename("test/path",
                                          start_date, n_timesteps, timestep,
                                          timestep_index)
        expected = (f"test/path_201909082022_"
                    f"{(timestep_index + 1) * timestep:03d}.tif")
        assert expected == file_name


def test_io_export_netcdf_one_member_one_time_step():
    """Test the export netcdf."""

    pytest.importorskip('netCDF4')
    pytest.importorskip('pyproj')

    # open a netcdf file
    root_path = pysteps.rcparams.data_sources["bom"]["root_path"]
    rel_path = os.path.join("prcp-cscn", "2", "2018", "06", "16")
    filename = os.path.join(root_path, rel_path,
                            "2_20180616_100000.prcp-cscn.nc")
    precip, _, metadata = pysteps.io.import_bom_rf3(filename)

    # save it back to disk
    with tempfile.TemporaryDirectory() as outpath:
        outfnprefix = 'test_netcdf_out'
        file_path = os.path.join(outpath, outfnprefix+'.nc')
        startdate = datetime.strptime("2018-06-16 10:00:00",
                                      "%Y-%m-%d %H:%M:%S")
        timestep = metadata['accutime']
        n_timesteps = 1
        shape = precip.shape
        exporter = initialize_forecast_exporter_netcdf(
            outpath, outfnprefix, startdate,
            timestep, n_timesteps, shape, metadata,
            n_ens_members=1,)
        export_forecast_dataset(precip[np.newaxis, :], exporter)
        close_forecast_files(exporter)
        # assert if netcdf file was saved and file size is not zero
        assert (os.path.exists(file_path) and os.path.getsize(file_path) > 0)
