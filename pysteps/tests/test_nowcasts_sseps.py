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
    (5, 6, 2, "incremental", "cdf", 200, 3, 0.62),
    (5, 6, 2, "incremental", "cdf", 200, [3], 0.62),
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
