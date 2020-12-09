# -*- coding: utf-8 -*-

import pytest

from pysteps import motion, nowcasts, verification
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
    (5, 6, 2, None, "mean", "spatial", 3, 1.30),
    (5, 6, 2, "incremental", "cdf", "spectral", 3, 0.60),
]


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_steps(
    n_ens_members,
    n_cascade_levels,
    ar_order,
    mask_method,
    probmatching_method,
    domain,
    timesteps,
    max_crps,
):
    """Tests STEPS nowcast."""
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


if __name__ == "__main__":
    for n in range(len(steps_arg_values)):
        test_args = zip(steps_arg_names, steps_arg_values[n])
        test_steps(**dict((x, y) for x, y in test_args))
