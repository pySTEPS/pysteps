# -*- coding: utf-8 -*-

import pytest

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
    (5, 6, 2, "incremental", "cdf", 200, 3, 0.60),
    (5, 6, 2, "incremental", "cdf", 200, [3], 0.60),
]


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
        metadata=True,
        upscale=2000,
    )
    precip_var = dataset_input.attrs["precip_var"]
    metadata = dataset_input[precip_var].attrs

    dataset_obs = get_precipitation_fields(
        num_prev_files=0, num_next_files=3, return_raw=False, upscale=2000
    ).isel(time=slice(1, None, None))

    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    retrieved_motion = oflow_method(dataset_input)

    nowcast_method = nowcasts.get_method("sseps")

    precip_forecast = nowcast_method(
        retrieved_motion,
        timesteps,
        {
            "xpixelsize": dataset_input["x"].values[1] - dataset_input["x"].values[0],
            **metadata,
        },
        win_size=win_size,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        ar_order=ar_order,
        seed=42,
        mask_method=mask_method,
        probmatching_method=probmatching_method,
    )
    precip_forecast_data = precip_forecast[precip_var].values

    assert precip_forecast_data.ndim == 4
    assert precip_forecast_data.shape[0] == n_ens_members
    assert precip_forecast_data.shape[1] == (
        timesteps if isinstance(timesteps, int) else len(timesteps)
    )

    crps = verification.probscores.CRPS(
        precip_forecast_data[:, -1], dataset_obs[precip_var].values[-1]
    )
    assert crps < max_crps, f"CRPS={crps:.2f}, required < {max_crps:.2f}"


if __name__ == "__main__":
    for n in range(len(sseps_arg_values)):
        test_args = zip(sseps_arg_names, sseps_arg_values[n])
        test_sseps(**dict((x, y) for x, y in test_args))
