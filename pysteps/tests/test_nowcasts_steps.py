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
    "max_crps",
)

steps_arg_values = [
    (5, 6, 2, None, None, 1.55),
    (5, 6, 2, "incremental", None, 6.65),
    (5, 6, 2, "sprog", None, 7.65),
    (5, 6, 2, "obs", None, 7.65),
    (5, 6, 2, None, "cdf", 0.70),
    (5, 6, 2, None, "mean", 1.55),
]


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_steps(
        n_ens_members,
        n_cascade_levels,
        ar_order,
        mask_method,
        probmatching_method,
        max_crps):
    """Tests STEPS nowcast."""
    # inputs
    precip_input, metadata = get_precipitation_fields(num_prev_files=2,
                                                      num_next_files=0,
                                                      return_raw=False,
                                                      metadata=True,
                                                      upscale=2000)

    precip_obs = get_precipitation_fields(num_prev_files=0,
                                          num_next_files=3,
                                          return_raw=False,
                                          upscale=2000)[1:, :, :]

    # Retrieve motion field
    oflow_method = motion.get_method("LK")
    retrieved_motion = oflow_method(precip_input)

    # Run nowcast
    nowcast_method = nowcasts.get_method("steps")

    precip_forecast = nowcast_method(
        precip_input,
        retrieved_motion,
        n_timesteps=3,
        R_thr=metadata["threshold"],
        kmperpixel=2.0,
        timestep=metadata["accutime"],
        seed=42,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        ar_order=ar_order,
        mask_method=mask_method,
        probmatching_method=probmatching_method,
    )

    # result
    result = verification.probscores.CRPS(precip_forecast[-1], precip_obs[-1])
    assert result < max_crps
