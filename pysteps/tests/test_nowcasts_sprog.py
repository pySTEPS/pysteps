# -*- coding: utf-8 -*-

import numpy as np
import pytest
import xarray as xr

from pysteps import motion, nowcasts, verification
from pysteps.tests.helpers import get_precipitation_fields

sprog_arg_names = (
    "n_cascade_levels",
    "ar_order",
    "probmatching_method",
    "domain",
    "timesteps",
    "min_csi",
)

sprog_arg_values = [
    (6, 1, None, "spatial", 3, 0.5),
    (6, 1, None, "spatial", [3], 0.5),
    (6, 2, None, "spatial", 3, 0.5),
    (6, 2, "cdf", "spatial", 3, 0.5),
    (6, 2, "mean", "spatial", 3, 0.5),
    (6, 2, "cdf", "spectral", 3, 0.5),
]


def test_default_sprog_norain():
    """Tests SPROG nowcast with default params and all-zero inputs."""

    # Define dummy nowcast input data
    dataset_input = xr.Dataset(
        data_vars={"precip_intensity": (["time", "y", "x"], np.zeros((3, 100, 100)))},
        attrs={"precip_var": "precip_intensity"},
    )

    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    retrieved_motion = oflow_method(dataset_input)

    nowcast_method = nowcasts.get_method("sprog")
    precip_forecast = nowcast_method(
        retrieved_motion,
        timesteps=3,
        precip_thr=0.1,
    )

    assert precip_forecast.ndim == 3
    assert precip_forecast.shape[0] == 3
    assert precip_forecast.sum() == 0.0


@pytest.mark.parametrize(sprog_arg_names, sprog_arg_values)
def test_sprog(
    n_cascade_levels, ar_order, probmatching_method, domain, timesteps, min_csi
):
    """Tests SPROG nowcast."""
    # inputs
    dataset_input = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        upscale=2000,
    )

    dataset_obs = get_precipitation_fields(
        num_prev_files=0, num_next_files=3, return_raw=False, upscale=2000
    ).isel(time=slice(1, None, None))
    precip_var = dataset_input.attrs["precip_var"]
    metadata = dataset_input[precip_var].attrs

    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    dataset_w_motion = oflow_method(dataset_input)

    nowcast_method = nowcasts.get_method("sprog")

    dataset_forecast = nowcast_method(
        dataset_w_motion,
        timesteps=timesteps,
        precip_thr=metadata["threshold"],
        n_cascade_levels=n_cascade_levels,
        ar_order=ar_order,
        probmatching_method=probmatching_method,
        domain=domain,
    )
    precip_forecast = dataset_forecast[precip_var].values

    assert precip_forecast.ndim == 3
    assert precip_forecast.shape[0] == (
        timesteps if isinstance(timesteps, int) else len(timesteps)
    )

    result = verification.det_cat_fct(
        precip_forecast[-1], dataset_obs[precip_var].values[-1], thr=0.1, scores="CSI"
    )["CSI"]
    assert result > min_csi, f"CSI={result:.1f}, required > {min_csi:.1f}"


if __name__ == "__main__":
    for n in range(len(sprog_arg_values)):
        test_args = zip(sprog_arg_names, sprog_arg_values[n])
        test_sprog(**dict((x, y) for x, y in test_args))
