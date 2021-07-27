import numpy as np
import pytest

from pysteps import motion, verification
from pysteps.nowcasts.linda import forecast
from pysteps.tests.helpers import get_precipitation_fields
from pysteps.utils import aggregate_fields_space

linda_arg_names = (
    "add_perturbations",
    "kernel_type",
    "measure_time",
    "min_csi",
    "max_crps",
)

linda_arg_values = [
    (False, "anisotropic", False, 0.5, None),
    (True, "isotropic", True, None, 0.1),
]


@pytest.mark.parametrize(linda_arg_names, linda_arg_values)
def test_linda(add_perturbations, kernel_type, measure_time, min_csi, max_crps):
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

    precip_forecast = forecast(
        precip_input,
        retrieved_motion,
        3,
        kernel_type=kernel_type,
        feature_kwargs={"threshold": 1.5, "min_sigma": 2, "max_sigma": 10},
        add_perturbations=add_perturbations,
        kmperpixel=2.0,
        timestep=metadata["accutime"],
        measure_time=measure_time,
    )

    if not add_perturbations:
        assert precip_forecast.ndim == 3
        assert precip_forecast.shape[0] == 3
        assert precip_forecast.shape[1:] == precip_input.shape[1:]

        csi = verification.det_cat_fct(
            precip_forecast[-1], precip_obs[-1], thr=1.0, scores="CSI"
        )["CSI"]
        assert csi > min_csi, f"CSI={csi:.1f}, required > {min_csi:.1f}"
    else:
        assert precip_forecast.ndim == 4
        assert precip_forecast.shape[0] == 40
        assert precip_forecast.shape[1] == 3
        assert precip_forecast.shape[2:] == precip_input.shape[1:]

        crps = verification.probscores.CRPS(precip_forecast[:, -1], precip_obs[-1])
        assert crps < max_crps, f"CRPS={crps:.2f}, required < {max_crps:.2f}"


def test_linda_wrong_inputs():

    # dummy inputs
    precip = np.zeros((3, 3, 3))
    velocity = np.zeros((2, 3, 3))

    # vel_pert_method is set but kmperpixel is None
    with pytest.raises(ValueError):
        forecast(precip, velocity, 1.0, vel_pert_method="bps", kmperpixel=None)

    # vel_pert_method is set but timestep is None
    with pytest.raises(ValueError):
        forecast(
            precip, velocity, 1.0, vel_pert_method="bps", kmperpixel=1, timestep=None
        )

    # fractional time steps not yet implemented
    with pytest.raises(NotImplementedError):
        forecast(precip, velocity, [1.0, 2.0], vel_pert_method=None)

    # ari_order 1 or 2 required
    with pytest.raises(ValueError):
        forecast(precip, velocity, 1, vel_pert_method=None, ari_order=3)

    # precip_fields must be a three-dimensional array
    with pytest.raises(ValueError):
        forecast(np.zeros((3, 3, 3, 3)), velocity, 1, vel_pert_method=None)

    # precip_fields.shape[0] < ari_order+2
    with pytest.raises(ValueError):
        forecast(np.zeros((4, 3, 3)), velocity, 1, vel_pert_method=None, ari_order=1)

    # advection_field must be a three-dimensional array
    with pytest.raises(ValueError):
        forecast(precip, velocity[0], 1, vel_pert_method=None)

    # dimension mismatch between precip_fields and advection_field
    with pytest.raises(ValueError):
        forecast(np.zeros((3, 2, 3)), velocity, 1, vel_pert_method=None)


if __name__ == "__main__":
    for n in range(len(linda_arg_values)):
        test_args = zip(linda_arg_names, linda_arg_values[n])
        test_linda(**dict((x, y) for x, y in test_args))
