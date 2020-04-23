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
    "max_crps",
)

sseps_arg_values = [
    (5, 6, 2, "incremental", "cdf", 200, 0.8),
]


@pytest.mark.parametrize(sseps_arg_names, sseps_arg_values)
def test_sseps(
    n_ens_members,
    n_cascade_levels,
    ar_order,
    mask_method,
    probmatching_method,
    win_size,
    max_crps,
):
    """Tests SSEPS nowcast."""
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

    # Retrieve motion field
    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    retrieved_motion = oflow_method(precip_input)

    # Run nowcast
    nowcast_method = nowcasts.get_method("sseps")

    precip_forecast = nowcast_method(
        precip_input,
        metadata,
        retrieved_motion,
        win_size=win_size,
        n_timesteps=3,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        ar_order=ar_order,
        seed=42,
        mask_method=mask_method,
        probmatching_method=probmatching_method,
    )

    # result
    crps = verification.probscores.CRPS(precip_forecast[-1], precip_obs[-1])
    print(f"got CRPS={crps:.1f}, required < {max_crps:.1f}")
    assert crps < max_crps


if __name__ == "__main__":
    for n in range(len(sseps_arg_values)):
        test_args = zip(sseps_arg_names, sseps_arg_values[n])
        test_sseps(**dict((x, y) for x, y in test_args))
