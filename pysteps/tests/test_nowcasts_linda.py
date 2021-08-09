import numpy as np
import pytest

from pysteps import motion, verification
from pysteps.nowcasts.linda import forecast
from pysteps.tests.helpers import get_precipitation_fields


linda_arg_names = (
    "add_perturbations",
    "kernel_type",
    "vel_pert_method",
    "num_workers",
    "measure_time",
    "min_csi",
    "max_crps",
)

linda_arg_values = [
    (False, "anisotropic", None, 1, False, 0.5, None),
    (False, "isotropic", None, 5, True, 0.5, None),
    (True, "anisotropic", None, 1, True, None, 0.3),
    (True, "isotropic", "bps", 5, True, None, 0.3),
]


@pytest.mark.parametrize(linda_arg_names, linda_arg_values)
def test_linda(
    add_perturbations,
    kernel_type,
    vel_pert_method,
    num_workers,
    measure_time,
    min_csi,
    max_crps,
):
    """Tests LINDA nowcast."""

    pytest.importorskip("cv2")
    pytest.importorskip("skimage")

    # inputs
    precip_input, metadata = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        metadata=True,
        clip=(354000, 866000, -96000, 416000),
        upscale=4000,
        log_transform=False,
    )

    precip_obs = get_precipitation_fields(
        num_prev_files=0,
        num_next_files=3,
        clip=(354000, 866000, -96000, 416000),
        upscale=4000,
        log_transform=False,
    )[1:, :, :]

    oflow_method = motion.get_method("LK")
    retrieved_motion = oflow_method(precip_input)

    precip_forecast = forecast(
        precip_input,
        retrieved_motion,
        3,
        kernel_type=kernel_type,
        vel_pert_method=vel_pert_method,
        feature_kwargs={"threshold": 1.5, "min_sigma": 2, "max_sigma": 10},
        add_perturbations=add_perturbations,
        kmperpixel=4.0,
        timestep=metadata["accutime"],
        measure_time=measure_time,
        num_ens_members=5,
        num_workers=num_workers,
        seed=42,
    )
    if measure_time:
        assert len(precip_forecast) == 3
        assert isinstance(precip_forecast[1], float)
        precip_forecast = precip_forecast[0]

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
        assert precip_forecast.shape[0] == 5
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
        forecast(precip, velocity, 1, vel_pert_method="bps", kmperpixel=None)

    # vel_pert_method is set but timestep is None
    with pytest.raises(ValueError):
        forecast(
            precip, velocity, 1, vel_pert_method="bps", kmperpixel=1, timestep=None
        )

    # fractional time steps not yet implemented
    # timesteps is not an integer
    with pytest.raises(ValueError):
        forecast(precip, velocity, [1.0, 2.0])

    # ari_order 1 or 2 required
    with pytest.raises(ValueError):
        forecast(precip, velocity, 1, ari_order=3)

    # precip_fields must be a three-dimensional array
    with pytest.raises(ValueError):
        forecast(np.zeros((3, 3, 3, 3)), velocity, 1)

    # precip_fields.shape[0] < ari_order+2
    with pytest.raises(ValueError):
        forecast(np.zeros((2, 3, 3)), velocity, 1, ari_order=1)

    # advection_field must be a three-dimensional array
    with pytest.raises(ValueError):
        forecast(precip, velocity[0], 1)

    # dimension mismatch between precip_fields and advection_field
    with pytest.raises(ValueError):
        forecast(np.zeros((3, 2, 3)), velocity, 1)
