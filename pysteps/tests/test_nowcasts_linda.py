import pytest

from pysteps import motion, nowcasts, verification
from pysteps.tests.helpers import get_precipitation_fields
from pysteps.utils import aggregate_fields_space

linda_arg_names = (
    "add_perturbations",
    "min_csi",
    "max_crps",
)

linda_arg_values = [
    (False, 0.5, None),
    (True, None, 0.1),
]


@pytest.mark.parametrize(linda_arg_names, linda_arg_values)
def test_linda(add_perturbations, min_csi, max_crps):
    """Tests LINDA nowcast."""
    # inputs
    precip_input, metadata_raw = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=True,
        metadata=True,
    )
    precip_input, metadata = aggregate_fields_space(precip_input, metadata_raw, 2000)

    precip_obs = get_precipitation_fields(
        num_prev_files=0,
        num_next_files=3,
        return_raw=True,
    )[1:, :, :]
    precip_obs, _ = aggregate_fields_space(precip_obs, metadata_raw, 2000)

    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    retrieved_motion = oflow_method(precip_input)

    nowcast_method = nowcasts.get_method("linda")

    precip_forecast = nowcast_method(
        precip_input,
        retrieved_motion,
        3,
        feature_kwargs={"threshold": 1.5, "min_sigma": 2, "max_sigma": 10},
        add_perturbations=add_perturbations,
        kmperpixel=2.0,
        timestep=metadata["accutime"],
    )

    if not add_perturbations:
        assert precip_forecast.ndim == 3

        csi = verification.det_cat_fct(
            precip_forecast[-1], precip_obs[-1], thr=1.0, scores="CSI"
        )["CSI"]
        assert csi > min_csi, f"CSI={result:.1f}, required > {min_csi:.1f}"
    else:
        assert precip_forecast.ndim == 4

        crps = verification.probscores.CRPS(precip_forecast[:, -1], precip_obs[-1])
        assert crps < max_crps, f"CRPS={crps:.2f}, required < {max_crps:.2f}"


if __name__ == "__main__":
    for n in range(len(linda_arg_values)):
        test_args = zip(linda_arg_names, linda_arg_values[n])
        test_linda(**dict((x, y) for x, y in test_args))
