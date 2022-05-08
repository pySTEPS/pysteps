# -*- coding: utf-8 -*-

import os
import tempfile
from datetime import datetime

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.io import import_netcdf_pysteps
from pysteps.io.exporters import _get_geotiff_filename
from pysteps.io.exporters import close_forecast_files
from pysteps.io.exporters import export_forecast_dataset
from pysteps.io.exporters import initialize_forecast_exporter_netcdf
from pysteps.tests.helpers import get_precipitation_fields, get_invalid_mask

# Test arguments
exporter_arg_names = ("n_ens_members", "incremental")

exporter_arg_values = [
    (1, None),
    (1, "timestep"),
    (2, None),
    (2, "timestep"),
    (2, "member"),
]


def test_get_geotiff_filename():
    """Test the geotif name generator."""

    start_date = datetime.strptime("201909082022", "%Y%m%d%H%M")

    n_timesteps = 50
    timestep = 5

    for timestep_index in range(n_timesteps):
        file_name = _get_geotiff_filename(
            "test/path", start_date, n_timesteps, timestep, timestep_index
        )
        expected = (
            f"test/path_201909082022_" f"{(timestep_index + 1) * timestep:03d}.tif"
        )
        assert expected == file_name


@pytest.mark.parametrize(exporter_arg_names, exporter_arg_values)
def test_io_export_netcdf_one_member_one_time_step(n_ens_members, incremental):
    """
    Test the export netcdf.
    Also, test that the exported file can be read by the importer.
    """

    pytest.importorskip("pyproj")

    precip, metadata = get_precipitation_fields(
        num_prev_files=2, return_raw=True, metadata=True, source="fmi"
    )

    invalid_mask = get_invalid_mask(precip)

    with tempfile.TemporaryDirectory() as outpath:
        # save it back to disk
        outfnprefix = "test_netcdf_out"
        file_path = os.path.join(outpath, outfnprefix + ".nc")
        startdate = metadata["timestamps"][0]
        timestep = metadata["accutime"]
        n_timesteps = 3
        shape = precip.shape[1:]

        exporter = initialize_forecast_exporter_netcdf(
            outpath,
            outfnprefix,
            startdate,
            timestep,
            n_timesteps,
            shape,
            metadata,
            n_ens_members=n_ens_members,
            incremental=incremental,
        )

        if n_ens_members > 1:
            precip = np.repeat(precip[np.newaxis, :, :, :], n_ens_members, axis=0)

        if incremental == None:
            export_forecast_dataset(precip, exporter)
        if incremental == "timestep":
            for t in range(n_timesteps):
                if n_ens_members > 1:
                    export_forecast_dataset(precip[:, t, :, :], exporter)
                else:
                    export_forecast_dataset(precip[t, :, :], exporter)
        if incremental == "member":
            for ens_mem in range(n_ens_members):
                export_forecast_dataset(precip[ens_mem, :, :, :], exporter)

        close_forecast_files(exporter)

        # assert if netcdf file was saved and file size is not zero
        assert os.path.exists(file_path) and os.path.getsize(file_path) > 0

        # Test that the file can be read by the nowcast_importer
        output_file_path = os.path.join(outpath, f"{outfnprefix}.nc")

        precip_new, _ = import_netcdf_pysteps(output_file_path)

        assert_array_almost_equal(precip.squeeze(), precip_new.data)
        assert precip_new.dtype == "single"

        precip_new, _ = import_netcdf_pysteps(output_file_path, dtype="double")
        assert_array_almost_equal(precip.squeeze(), precip_new.data)
        assert precip_new.dtype == "double"

        precip_new, _ = import_netcdf_pysteps(output_file_path, fillna=-1000)
        new_invalid_mask = precip_new == -1000
        assert (new_invalid_mask == invalid_mask).all()
