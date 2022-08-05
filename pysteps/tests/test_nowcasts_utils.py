import numpy as np
import pytest

from pysteps import motion
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.tests.helpers import get_precipitation_fields

main_loop_arg_names = (
    "timesteps",
    "ensemble",
    "num_ensemble_members",
    "velocity_perturbations",
)

# TODO: add tests for callback and other untested options
main_loop_arg_values = [
    (6, False, 0, False),
    ([0.5, 1.5], False, 0, False),
    (6, True, 2, False),
    (6, True, 2, True),
]


@pytest.mark.parametrize(main_loop_arg_names, main_loop_arg_values)
def test_nowcast_main_loop(
    timesteps, ensemble, num_ensemble_members, velocity_perturbations
):
    """Test the nowcast_main_loop function."""
    precip = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        metadata=False,
        upscale=2000,
    )
    precip = precip.filled()

    oflow_method = motion.get_method("LK")
    velocity = oflow_method(precip)

    precip = precip[-1]

    state = {"input": precip}
    extrap_method = "semilagrangian"

    def func(state, params):
        if not ensemble:
            precip_out = state["input"]
        else:
            precip_out = state["input"][np.newaxis, :]

        return precip_out, state

    nowcast_utils.nowcast_main_loop(
        precip,
        velocity,
        state,
        timesteps,
        extrap_method,
        func,
        ensemble=ensemble,
        num_ensemble_members=num_ensemble_members,
    )
