# -*- coding: utf-8 -*-

import pytest

from pysteps import motion, nowcasts, verification
from pysteps.tests.helpers import get_precipitation_fields

sprog_arg_names = (
    "n_cascade_levels",
    "ar_order",
    "probmatching_method",
    "domain",
    "min_csi",
)

sprog_arg_values = [
    (6, 1, None, "spatial", 0.5),
    (6, 2, None, "spatial", 0.5),
    (6, 2, "cdf", "spatial", 0.5),
    (6, 2, "mean", "spatial", 0.5),
    (6, 2, "cdf", "spectral", 0.5),
]


@pytest.mark.parametrize(sprog_arg_names, sprog_arg_values)
def test_sprog(n_cascade_levels, ar_order, probmatching_method, domain, min_csi):
    """Tests SPROG nowcast."""
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
    nowcast_method = nowcasts.get_method("sprog")

    precip_forecast = nowcast_method(
        precip_input,
        retrieved_motion,
        n_timesteps=3,
        R_thr=metadata["threshold"],
        n_cascade_levels=n_cascade_levels,
        ar_order=ar_order,
        probmatching_method=probmatching_method,
        domain=domain,
    )

    # result
    result = verification.det_cat_fct(
        precip_forecast[-1], precip_obs[-1], thr=0.1, scores="CSI"
    )["CSI"]
    print(f"got CSI={result:.1f}, required > {min_csi:.1f}")
    assert result > min_csi


if __name__ == "__main__":
    for n in range(len(sprog_arg_values)):
        test_args = zip(sprog_arg_names, sprog_arg_values[n])
        test_sprog(**dict((x, y) for x, y in test_args))
