# -*- coding: utf-8 -*-

from datetime import datetime

from pysteps.io.exporters import _get_geotiff_filename


def test_get_geotiff_filename():
    """Test the geotif name generator."""

    start_date = datetime.strptime("201909082022", "%Y%m%d%H%M")

    n_timesteps = 50
    timestep = 5

    for timestep_index in range(n_timesteps):
        file_name = _get_geotiff_filename("test/path", start_date, n_timesteps, timestep,
                                          timestep_index)

        assert f"test/path_201909082022_{(timestep_index + 1) * timestep:03d}.tif" == file_name
