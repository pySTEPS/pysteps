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
from pysteps.io.exporters import _convert_proj4_to_grid_mapping
from pysteps.tests.helpers import get_precipitation_fields, get_invalid_mask

# Test arguments
exporter_arg_names = (
    "n_ens_members",
    "incremental",
    "datatype",
    "fill_value",
    "scale_factor",
    "offset",
    "n_timesteps",
)

exporter_arg_values = [
    (1, None, np.float32, None, None, None, 3),
    (1, "timestep", np.float32, 65535, None, None, 3),
    (2, None, np.float32, 65535, None, None, 3),
    (2, None, np.float32, 65535, None, None, [1, 2, 4]),
    (2, "timestep", np.float32, None, None, None, 3),
    (2, "timestep", np.float32, None, None, None, [1, 2, 4]),
    (2, "member", np.float64, None, 0.01, 1.0, 3),
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
def test_io_export_netcdf_one_member_one_time_step(
    n_ens_members, incremental, datatype, fill_value, scale_factor, offset, n_timesteps
):
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
            datatype=datatype,
            incremental=incremental,
            fill_value=fill_value,
            scale_factor=scale_factor,
            offset=offset,
        )

        if n_ens_members > 1:
            precip = np.repeat(precip[np.newaxis, :, :, :], n_ens_members, axis=0)

        if incremental == None:
            export_forecast_dataset(precip, exporter)
        if incremental == "timestep":
            if isinstance(n_timesteps, list):
                timesteps = len(n_timesteps)
            else:
                timesteps = n_timesteps
            for t in range(timesteps):
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


@pytest.mark.parametrize(
    ["proj4str", "expected_value"],
    [
        (
            "+proj=lcc +lat_1=49.83333333333334 +lat_2=51.16666666666666 +lat_0=50.797815 +lon_0=4.359215833333333 +x_0=649328 +y_0=665262 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ",
            (
                "lcc",
                "lambert_conformal_conic",
                {
                    "false_easting": 649328.0,
                    "false_northing": 665262.0,
                    "longitude_of_central_meridian": 4.359215833333333,
                    "latitude_of_projection_origin": 50.797815,
                    "standard_parallel": (49.83333333333334, 51.16666666666666),
                    "reference_ellipsoid_name": "GRS80",
                    "towgs84": "0,0,0,0,0,0,0",
                },
            ),
        ),
        (
            "+proj=aea +lat_0=-37.852 +lon_0=144.752 +lat_1=-18.0 +lat_2=-36.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0",
            (
                "proj",
                "albers_conical_equal_area",
                {
                    "false_easting": 0.0,
                    "false_northing": 0.0,
                    "longitude_of_central_meridian": 144.752,
                    "latitude_of_projection_origin": -37.852,
                    "standard_parallel": (-18.0, -36.0),
                },
            ),
        ),
        (
            "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0",
            (
                "polar_stereographic",
                "polar_stereographic",
                {
                    "straight_vertical_longitude_from_pole": 0.0,
                    "latitude_of_projection_origin": 90.0,
                    "standard_parallel": 60.0,
                    "false_easting": 0.0,
                    "false_northing": 0.0,
                },
            ),
        ),
    ],
)
def test_convert_proj4_to_grid_mapping(proj4str, expected_value):
    """
    test the grid mapping in function _convert_proj4_to_grid_mapping()
    """
    output = _convert_proj4_to_grid_mapping(proj4str)

    assert output == expected_value
