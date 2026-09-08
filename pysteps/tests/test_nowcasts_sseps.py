# -*- coding: utf-8 -*-

import numpy as np
import pytest
import xarray as xr

from pysteps import motion, nowcasts, verification
from pysteps.tests.helpers import get_precipitation_fields

sseps_arg_names = (
    "n_ens_members",
    "n_cascade_levels",
    "ar_order",
    "mask_method",
    "probmatching_method",
    "win_size",
    "timesteps",
    "max_crps",
)

sseps_arg_values = [
    (5, 6, 2, "incremental", "cdf", 200, 3, 0.62),
    (5, 6, 2, "incremental", "cdf", 200, [3], 0.62),
]


def test_default_sseps_norain():
    """Tests SSEPS nowcast with default params and all-zero inputs."""

    # Define dummy nowcast input data
    dataset_input = xr.Dataset(
        data_vars={
            "precip_intensity": (
                ["time", "y", "x"],
                np.zeros((3, 100, 100)),
                {
                    "units": "mm/h",
                    "accutime": 5,
                    "threshold": 0.1,
                    "zerovalue": 0,
                },
            )
        },
        coords={
            "time": (
                ["time"],
                np.arange(3.0) * 5.0,
                {"long_name": "forecast time", "stepsize": 5.0},
                {"units": "seconds since 1970-01-01 00:00:00"},
            ),
            "y": (
                ["y"],
                np.arange(100.0) * 1000.0,
                {
                    "axis": "X",
                    "long_name": "x-coordinate in Cartesian system",
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                    "stepsize": 1000.0,
                },
            ),
            "x": (
                ["x"],
                np.arange(100.0) * 1000.0,
                {
                    "axis": "X",
                    "long_name": "x-coordinate in Cartesian system",
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                    "stepsize": 1000.0,
                },
            ),
        },
        attrs={"precip_var": "precip_intensity"},
    )

    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    retrieved_motion = oflow_method(dataset_input)

    nowcast_method = nowcasts.get_method("sseps")
    precip_forecast = nowcast_method(
        retrieved_motion,
        n_ens_members=3,
        timesteps=3,
    )

    assert precip_forecast.ndim == 4
    assert precip_forecast.shape[0] == 3
    assert precip_forecast.shape[1] == 3
    assert precip_forecast.sum() == 0.0


@pytest.mark.parametrize(sseps_arg_names, sseps_arg_values)
def test_sseps(
    n_ens_members,
    n_cascade_levels,
    ar_order,
    mask_method,
    probmatching_method,
    win_size,
    timesteps,
    max_crps,
):
    """Tests SSEPS nowcast."""
    # inputs
    dataset_input = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        upscale=2000,
    )
    precip_var = dataset_input.attrs["precip_var"]

    dataset_obs = get_precipitation_fields(
        num_prev_files=0, num_next_files=3, return_raw=False, upscale=2000
    ).isel(time=slice(1, None, None))

    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    dataset_w_motion = oflow_method(dataset_input)

    nowcast_method = nowcasts.get_method("sseps")

    dataset_forecast = nowcast_method(
        dataset_w_motion,
        timesteps,
        win_size=win_size,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        ar_order=ar_order,
        seed=42,
        mask_method=mask_method,
        probmatching_method=probmatching_method,
    )
    precip_forecast = dataset_forecast[precip_var].values

    assert precip_forecast.ndim == 4
    assert precip_forecast.shape[0] == n_ens_members
    assert precip_forecast.shape[1] == (
        timesteps if isinstance(timesteps, int) else len(timesteps)
    )

    crps = verification.probscores.CRPS(
        precip_forecast[:, -1], dataset_obs[precip_var].values[-1]
    )
    assert crps < max_crps, f"CRPS={crps:.2f}, required < {max_crps:.2f}"


if __name__ == "__main__":
    for n in range(len(sseps_arg_values)):
        test_args = zip(sseps_arg_names, sseps_arg_values[n])
        test_sseps(**dict((x, y) for x, y in test_args))
