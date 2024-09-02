import pytest

from pysteps import motion, nowcasts, verification
from pysteps.tests.helpers import get_precipitation_fields

anvil_arg_names = (
    "n_cascade_levels",
    "ar_order",
    "ar_window_radius",
    "timesteps",
    "min_csi",
    "apply_rainrate_mask",
    "measure_time",
)

anvil_arg_values = [
    (8, 1, 50, 3, 0.6, True, False),
    (8, 1, 50, [3], 0.6, False, True),
]


@pytest.mark.parametrize(anvil_arg_names, anvil_arg_values)
def test_anvil_rainrate(
    n_cascade_levels,
    ar_order,
    ar_window_radius,
    timesteps,
    min_csi,
    apply_rainrate_mask,
    measure_time,
):
    """Tests ANVIL nowcast using rain rate precipitation fields."""
    # inputs
    dataset_input = get_precipitation_fields(
        num_prev_files=4,
        num_next_files=0,
        return_raw=False,
        metadata=False,
        upscale=2000,
    )

    dataset_obs = get_precipitation_fields(
        num_prev_files=0, num_next_files=3, return_raw=False, upscale=2000
    ).isel(time=slice(1, None, None))
    precip_var = dataset_input.attrs["precip_var"]

    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    dataset_w_motion = oflow_method(dataset_input)

    nowcast_method = nowcasts.get_method("anvil")

    output = nowcast_method(
        dataset_w_motion.isel(time=slice(-(ar_order + 2), None, None)),
        timesteps=timesteps,
        n_cascade_levels=n_cascade_levels,
        ar_order=ar_order,
        ar_window_radius=ar_window_radius,
        apply_rainrate_mask=apply_rainrate_mask,
        measure_time=measure_time,
    )
    if measure_time:
        dataset_forecast, __, __ = output
    else:
        dataset_forecast = output
    precip_forecast = dataset_forecast[precip_var].values

    assert precip_forecast.ndim == 3
    assert precip_forecast.shape[0] == (
        timesteps if isinstance(timesteps, int) else len(timesteps)
    )

    result = verification.det_cat_fct(
        precip_forecast[-1], dataset_obs[precip_var].values[-1], thr=0.1, scores="CSI"
    )["CSI"]
    assert result > min_csi, f"CSI={result:.2f}, required > {min_csi:.2f}"


if __name__ == "__main__":
    for n in range(len(anvil_arg_values)):
        test_args = zip(anvil_arg_names, anvil_arg_values[n])
        test_anvil_rainrate(**dict((x, y) for x, y in test_args))
