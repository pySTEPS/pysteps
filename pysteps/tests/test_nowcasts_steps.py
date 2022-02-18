# -*- coding: utf-8 -*-

import os
from datetime import timedelta

import numpy as np
import pytest

from pysteps import io, motion, nowcasts, verification
from pysteps.tests.helpers import get_precipitation_fields


steps_arg_names = (
    "n_ens_members",
    "n_cascade_levels",
    "ar_order",
    "mask_method",
    "probmatching_method",
    "domain",
    "timesteps",
    "max_crps",
)

steps_arg_values = [
    (5, 6, 2, None, None, "spatial", 3, 1.30),
    (5, 6, 2, None, None, "spatial", [3], 1.30),
    (5, 6, 2, "incremental", None, "spatial", 3, 7.25),
    (5, 6, 2, "sprog", None, "spatial", 3, 8.35),
    (5, 6, 2, "obs", None, "spatial", 3, 8.30),
    (5, 6, 2, None, "cdf", "spatial", 3, 0.60),
    (5, 6, 2, None, "mean", "spatial", 3, 1.35),
    (5, 6, 2, "incremental", "cdf", "spectral", 3, 0.60),
]


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_steps_skill(
    n_ens_members,
    n_cascade_levels,
    ar_order,
    mask_method,
    probmatching_method,
    domain,
    timesteps,
    max_crps,
):
    """Tests STEPS nowcast skill."""
    # inputs
    precip_input, metadata = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        metadata=True,
        upscale=2000,
    )
    precip_input = precip_input.filled()

    precip_obs = get_precipitation_fields(
        num_prev_files=0, num_next_files=3, return_raw=False, upscale=2000
    )[1:, :, :]
    precip_obs = precip_obs.filled()

    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    retrieved_motion = oflow_method(precip_input)

    nowcast_method = nowcasts.get_method("steps")

    precip_forecast = nowcast_method(
        precip_input,
        retrieved_motion,
        timesteps=timesteps,
        R_thr=metadata["threshold"],
        kmperpixel=2.0,
        timestep=metadata["accutime"],
        seed=42,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        ar_order=ar_order,
        mask_method=mask_method,
        probmatching_method=probmatching_method,
        domain=domain,
    )

    assert precip_forecast.ndim == 4
    assert precip_forecast.shape[0] == n_ens_members
    assert precip_forecast.shape[1] == (
        timesteps if isinstance(timesteps, int) else len(timesteps)
    )

    crps = verification.probscores.CRPS(precip_forecast[:, -1], precip_obs[-1])
    assert crps < max_crps, f"CRPS={crps:.2f}, required < {max_crps:.2f}"


def test_steps_callback(tmp_path):
    """Test STEPS callback functionality to export the output as a netcdf."""

    pytest.importorskip("netCDF4")

    n_ens_members = 2
    n_timesteps = 3

    precip_input, metadata = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        metadata=True,
        upscale=2000,
    )
    precip_input = precip_input.filled()
    field_shape = (precip_input.shape[1], precip_input.shape[2])
    startdate = metadata["timestamps"][-1]
    timestep = metadata["accutime"]

    motion_field = np.zeros((2, *field_shape))

    exporter = io.initialize_forecast_exporter_netcdf(
        outpath=tmp_path.as_posix(),
        outfnprefix="test_steps",
        startdate=startdate,
        timestep=timestep,
        n_timesteps=n_timesteps,
        shape=field_shape,
        n_ens_members=n_ens_members,
        metadata=metadata,
        incremental="timestep",
    )

    def callback(array):
        return io.export_forecast_dataset(array, exporter)

    precip_output = nowcasts.get_method("steps")(
        precip_input,
        motion_field,
        timesteps=n_timesteps,
        R_thr=metadata["threshold"],
        kmperpixel=2.0,
        timestep=timestep,
        seed=42,
        n_ens_members=n_ens_members,
        vel_pert_method=None,
        callback=callback,
        return_output=True,
    )
    io.close_forecast_files(exporter)

    # assert that netcdf exists and its size is not zero
    tmp_file = os.path.join(tmp_path, "test_steps.nc")
    assert os.path.exists(tmp_file) and os.path.getsize(tmp_file) > 0

    # assert that the file can be read by the nowcast importer
    precip_netcdf, metadata_netcdf = io.import_netcdf_pysteps(tmp_file, dtype="float64")

    # assert that the dimensionality of the array is as expected
    assert precip_netcdf.ndim == 4, "Wrong number of dimensions"
    assert precip_netcdf.shape[0] == n_ens_members, "Wrong ensemble size"
    assert precip_netcdf.shape[1] == n_timesteps, "Wrong number of lead times"
    assert precip_netcdf.shape[2:] == field_shape, "Wrong field shape"

    # assert that the saved output is the same as the original output
    assert np.allclose(
        precip_netcdf, precip_output, equal_nan=True
    ), "Wrong output values"

    # assert that leadtimes and timestamps are as expected
    td = timedelta(minutes=timestep)
    leadtimes = [(i + 1) * timestep for i in range(n_timesteps)]
    timestamps = [startdate + (i + 1) * td for i in range(n_timesteps)]
    assert (metadata_netcdf["leadtimes"] == leadtimes).all(), "Wrong leadtimes"
    assert (metadata_netcdf["timestamps"] == timestamps).all(), "Wrong timestamps"
