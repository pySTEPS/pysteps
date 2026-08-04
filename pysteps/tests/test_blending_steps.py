# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pytest

import pysteps
from pysteps import blending, cascade

# fmt:off
steps_arg_values = [
    (1, 3, 4, 8, 'steps', None, None, False, "spn", True, 4, False, False, 0, False, None, None, None),
    (1, 3, 4, 8,'steps', "obs", None, False, "spn", True, 4, False, False, 0, False, None, None, None),
    (1, 3, 4, 8,'steps', "incremental", None, False, "spn", True, 4, False, False, 0, False, None, None, None),
    (1, 3, 4, 8,'steps', None, "mean", False, "spn", True, 4, False, False, 0, False, None, None, None),
    (1, 3, 4, 8,'steps', None, "mean", False, "spn", True, 4, False, False, 0, True, None, None, None),
    (1, 3, 4, 8,'steps', None, "cdf", False, "spn", True, 4, False, False, 0, False, None, None, None),
    (1, [1, 2, 3], 4, 8,'steps', None, "cdf", False, "spn", True, 4, False, False, 0, False, None, None, None),
    (1, 3, 4, 8,'steps', "incremental", "cdf", False, "spn", True, 4, False, False, 0, False, None, None, None),
    (1, 3, 4, 6,'steps', "incremental", "cdf", False, "bps", True, 4, False, False, 0, False, None, None, None),
    (1, 3, 4, 6,'steps', "incremental", "cdf", False, "bps", False, 4, False, False, 0, False, None, None, None),
    (1, 3, 4, 6,'steps', "incremental", "cdf", False, "bps", False, 4, False, False, 0, True, None, None, None),
    (1, 3, 4, 9,'steps', "incremental", "cdf", False, "spn", True, 4, False, False, 0, False, None, None, None),
    (2, 3, 10, 8,'steps', "incremental", "cdf", False, "spn", True, 10, False, False, 0, False, None, None, None),
    (5, 3, 5, 8,'steps', "incremental", "cdf", False, "spn", True, 5, False, False, 0, False, None, None, None),
    (1, 10, 1, 8,'steps', "incremental", "cdf", False, "spn", True, 1, False, False, 0, False, None, None, None),
    (2, 3, 2, 8,'steps', "incremental", "cdf", True, "spn", True, 2, False, False, 0, False, None, None, None),
    (1, 3, 6, 8,'steps', None, None, False, "spn", True, 6, False, False, 0, False, None, None, None),
    (1, 3, 6, 8,'steps', None, None, False, "spn", True, 6, False, False, 0, False, "bps", None, None),
    # Test the case where the radar image contains no rain.
    (1, 3, 6, 8,'steps', None, None, False, "spn", True, 6, True, False, 0, False, None, None, None),
    (5, 3, 5, 6,'steps', "incremental", "cdf", False, "spn", False, 5, True, False, 0, False, None, None, None),
    (5, 3, 5, 6,'steps', "incremental", "cdf", False, "spn", False, 5, True, False, 0, True, None, None, None),
    # Test the case where the NWP fields contain no rain.
    (1, 3, 6, 8,'steps', None, None, False, "spn", True, 6, False, True, 0, False, None, None, None),
    (5, 3, 5, 6,'steps', "incremental", "cdf", False, "spn", False, 5, False, True, 0, True, None, None, None),
    # Test the case where both the radar image and the NWP fields contain no rain.
    (1, 3, 6, 8,'steps', None, None, False, "spn", True, 6, True, True, 0, False, None, None, None),
    (5, 3, 5, 6,'steps', "incremental", "cdf", False, "spn", False, 5, True, True, 0, False, None, None, None),
    (5, 3, 5, 6,'steps', "obs", "mean", True, "spn", True, 5, True, True, 0, False, None, None, None),
    # Test cases where we apply timestep_start_full_nwp_weight
    (1, 10, 2, 6,'steps', "incremental", "cdf", False, "bps", False, 2, False, False, 0, True, None, None, 5),
    (1, 10, 2, 6,'steps', "incremental", "cdf", False, "spn", False, 2, False, False, 0, False, None, None, 5),
    # Test for smooth radar mask
    (1, 3, 6, 8,'steps', None, None, False, "spn", True, 6, False, False, 80, False, None, None, None),
    (5, 3, 5, 6,'steps', "incremental", "cdf", False, "spn", False, 5, False, False, 80, False, None, None, None),
    (5, 3, 5, 6,'steps', "obs", "mean", False, "spn", False, 5, False, False, 80, False, None, None, None),
    (1, 3, 6, 8,'steps', None, None, False, "spn", True, 6, False, True, 80, False, None, None, None),
    (5, 3, 5, 6,'steps', "incremental", "cdf", False, "spn", False, 5, True, False, 80, True, None, None, None),
    (5, 3, 5, 6,'steps', "obs", "mean", False, "spn", False, 5, True, True, 80, False, None, None, None),
    (5, [1, 2, 3], 5, 6,'steps', "obs", "mean", False, "spn", False, 5, True, True, 80, False, None, None, None),
    (5, [1, 3], 5, 6,'steps', "obs", "mean", False, "spn", False, 5, True, True, 80, False, None, None, None),
    # Test the usage of a max_mask_rim in the mask_kwargs
    (1, 3, 6, 8,'steps', None, None, False, "bps", True, 6, False, False, 80, False, None, 40, None),
    (5, 3, 5, 6,'steps', "obs", "mean", False, "bps", False, 5, False, False, 80, False, None, 40, None),
    (5, 3, 5, 6,'steps', "incremental", "cdf", False, "bps", False, 5, False, False, 80, False, None, 25, None),
    (5, 3, 5, 6,'steps', "incremental", "cdf", False, "bps", False, 5, False, False, 80, False, None, 40, None),
    (5, 3, 5, 6,'steps', "incremental", "cdf", False, "bps", False, 5, False, False, 80, False, None, 60, None),
    #Test the externally provided nowcast
    (1, 10, 1, 8,'external_nowcast_det', None, None, False, "spn", True, 1, False, False, 0, False, None, None, None),
    (1, 10, 1, 8,'external_nowcast_det', "incremental", None, False, "bps", True, 1, False, False, 0, False, None, None, None),
    (1, 10, 1, 8,'external_nowcast_det', "incremental", None, False, "spn", True, 1, False, False, 80, False, None, None, None),
    (1, 10, 1, 8,'external_nowcast_det', "incremental", None, False, "bps", True, 1, True, False, 0, False, None, None, None),
    (1, 10, 1, 8,'external_nowcast_det', "incremental", None, False, "spn", True, 1, False, True, 0, False, None, None, None),
    (1, 10, 1, 8,'external_nowcast_det', "incremental", None, False, "bps", True, 1, True, True, 0, False, None, None, None),
    (1, 10, 1, 8,'external_nowcast_det', "incremental", "cdf", False, "spn", True, 1, False, False, 0, True, None, None, None),
    (1, 10, 1, 8,'external_nowcast_det', "incremental", "obs", False, "bps", True, 1, False, False, 0, False, None, None, None),
    (1, 10, 1, 8,'external_nowcast_det', "incremental", None, False, "bps", True, 1, False, False, 0, False, None, None, 5),
    (5, 10, 5, 8,'external_nowcast_ens', "incremental", None, False, "spn", True, 5, False, False, 0, False, None, None, None),
    (5, 10, 5, 8,'external_nowcast_ens', "incremental", None, False, "spn", True, 5, False, False, 0, False, None, None, None),
    (1, 10, 5, 8,'external_nowcast_ens', "incremental", None, False, "spn", True, 5, False, False, 0, False, None, None, None),
    (1, 10, 1, 8,'external_nowcast_ens', "incremental", "cdf", False, "bps", True, 5, False, False, 0, False, None, None, None),
    (5, 10, 1, 8,'external_nowcast_ens', "incremental", "obs", False, "spn", True, 5, False, False, 0, False, None, None, None),
    (1, 10, 5, 8,'external_nowcast_ens', "incremental", "cdf", False, "bps", True, 5, False, False, 0, False, None, None, 5)
]

# fmt:on


def run_and_assert_forecast(
    precip, forecast_kwargs, expected_n_ens_members, n_timesteps, converter, metadata
):
    """Run a blended nowcast and assert the output has the expected shape."""
    precip_forecast = blending.steps.forecast(precip=precip, **forecast_kwargs)

    assert precip_forecast.ndim == 4, "Wrong amount of dimensions in forecast output"
    assert (
        precip_forecast.shape[0] == expected_n_ens_members
    ), "Wrong amount of output ensemble members in forecast output"
    assert (
        precip_forecast.shape[1] == n_timesteps
    ), "Wrong amount of output time steps in forecast output"

    # Transform the data back into mm/h
    precip_forecast, _ = converter(precip_forecast, metadata)

    assert (
        precip_forecast.ndim == 4
    ), "Wrong amount of dimensions in converted forecast output"
    assert (
        precip_forecast.shape[0] == expected_n_ens_members
    ), "Wrong amount of output ensemble members in converted forecast output"
    assert (
        precip_forecast.shape[1] == n_timesteps
    ), "Wrong amount of output time steps in converted forecast output"


steps_arg_names = (
    "n_models",
    "timesteps",
    "n_ens_members",
    "n_cascade_levels",
    "nowcasting_method",
    "mask_method",
    "probmatching_method",
    "blend_nwp_members",
    "weights_method",
    "decomposed_nwp",
    "expected_n_ens_members",
    "zero_radar",
    "zero_nwp",
    "smooth_radar_mask_range",
    "resample_distribution",
    "vel_pert_method",
    "max_mask_rim",
    "timestep_start_full_nwp_weight",
)


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_steps_blending(
    n_models,
    timesteps,
    n_ens_members,
    n_cascade_levels,
    nowcasting_method,
    mask_method,
    probmatching_method,
    blend_nwp_members,
    weights_method,
    decomposed_nwp,
    expected_n_ens_members,
    zero_radar,
    zero_nwp,
    smooth_radar_mask_range,
    resample_distribution,
    vel_pert_method,
    max_mask_rim,
    timestep_start_full_nwp_weight,
):
    pytest.importorskip("cv2")

    ###
    # The input data
    ###
    # Initialise dummy NWP data
    if not isinstance(timesteps, int):
        n_timesteps = len(timesteps)
        last_timestep = timesteps[-1]
    else:
        n_timesteps = timesteps
        last_timestep = timesteps

    nwp_precip = np.zeros((n_models, last_timestep + 1, 200, 200))

    if not zero_nwp:
        for n_model in range(n_models):
            for i in range(nwp_precip.shape[1]):
                nwp_precip[n_model, i, 30:185, 30 + 1 * (i + 1) * n_model] = 0.1
                nwp_precip[n_model, i, 30:185, 31 + 1 * (i + 1) * n_model] = 0.1
                nwp_precip[n_model, i, 30:185, 32 + 1 * (i + 1) * n_model] = 1.0
                nwp_precip[n_model, i, 30:185, 33 + 1 * (i + 1) * n_model] = 5.0
                nwp_precip[n_model, i, 30:185, 34 + 1 * (i + 1) * n_model] = 5.0
                nwp_precip[n_model, i, 30:185, 35 + 1 * (i + 1) * n_model] = 4.5
                nwp_precip[n_model, i, 30:185, 36 + 1 * (i + 1) * n_model] = 4.5
                nwp_precip[n_model, i, 30:185, 37 + 1 * (i + 1) * n_model] = 4.0
                nwp_precip[n_model, i, 30:185, 38 + 1 * (i + 1) * n_model] = 2.0
                nwp_precip[n_model, i, 30:185, 39 + 1 * (i + 1) * n_model] = 1.0
                nwp_precip[n_model, i, 30:185, 40 + 1 * (i + 1) * n_model] = 0.5
                nwp_precip[n_model, i, 30:185, 41 + 1 * (i + 1) * n_model] = 0.1

    # Define dummy nowcast input data
    radar_precip = np.zeros((3, 200, 200))

    if not zero_radar:
        for i in range(2):
            radar_precip[i, 5:150, 30 + 1 * i] = 0.1
            radar_precip[i, 5:150, 31 + 1 * i] = 0.5
            radar_precip[i, 5:150, 32 + 1 * i] = 0.5
            radar_precip[i, 5:150, 33 + 1 * i] = 5.0
            radar_precip[i, 5:150, 34 + 1 * i] = 5.0
            radar_precip[i, 5:150, 35 + 1 * i] = 4.5
            radar_precip[i, 5:150, 36 + 1 * i] = 4.5
            radar_precip[i, 5:150, 37 + 1 * i] = 4.0
            radar_precip[i, 5:150, 38 + 1 * i] = 1.0
            radar_precip[i, 5:150, 39 + 1 * i] = 0.5
            radar_precip[i, 5:150, 40 + 1 * i] = 0.5
            radar_precip[i, 5:150, 41 + 1 * i] = 0.1
        radar_precip[2, 30:155, 30 + 1 * 2] = 0.1
        radar_precip[2, 30:155, 31 + 1 * 2] = 0.1
        radar_precip[2, 30:155, 32 + 1 * 2] = 1.0
        radar_precip[2, 30:155, 33 + 1 * 2] = 5.0
        radar_precip[2, 30:155, 34 + 1 * 2] = 5.0
        radar_precip[2, 30:155, 35 + 1 * 2] = 4.5
        radar_precip[2, 30:155, 36 + 1 * 2] = 4.5
        radar_precip[2, 30:155, 37 + 1 * 2] = 4.0
        radar_precip[2, 30:155, 38 + 1 * 2] = 2.0
        radar_precip[2, 30:155, 39 + 1 * 2] = 1.0
        radar_precip[2, 30:155, 40 + 1 * 3] = 0.5
        radar_precip[2, 30:155, 41 + 1 * 3] = 0.1

    precip_nowcast = np.zeros((n_ens_members, last_timestep + 1, 200, 200))

    if nowcasting_method == "external_nowcast_ens":
        nowcasting_method = "external_nowcast"
        for n_ens_member in range(n_ens_members):
            for i in range(precip_nowcast.shape[1]):
                precip_nowcast[
                    n_ens_member, i, 30:165, 30 + 1 * (i + 1) * n_ens_member
                ] = 0.1
                precip_nowcast[
                    n_ens_member, i, 30:165, 31 + 1 * (i + 1) * n_ens_member
                ] = 0.5
                precip_nowcast[
                    n_ens_member, i, 30:165, 32 + 1 * (i + 1) * n_ens_member
                ] = 0.5
                precip_nowcast[
                    n_ens_member, i, 30:165, 33 + 1 * (i + 1) * n_ens_member
                ] = 5.0
                precip_nowcast[
                    n_ens_member, i, 30:165, 34 + 1 * (i + 1) * n_ens_member
                ] = 5.0
                precip_nowcast[
                    n_ens_member, i, 30:165, 35 + 1 * (i + 1) * n_ens_member
                ] = 4.5
                precip_nowcast[
                    n_ens_member, i, 30:165, 36 + 1 * (i + 1) * n_ens_member
                ] = 4.5
                precip_nowcast[
                    n_ens_member, i, 30:165, 37 + 1 * (i + 1) * n_ens_member
                ] = 4.0
                precip_nowcast[
                    n_ens_member, i, 30:165, 38 + 1 * (i + 1) * n_ens_member
                ] = 1.0
                precip_nowcast[
                    n_ens_member, i, 30:165, 39 + 1 * (i + 1) * n_ens_member
                ] = 0.5
                precip_nowcast[
                    n_ens_member, i, 30:165, 40 + 1 * (i + 1) * n_ens_member
                ] = 0.5
                precip_nowcast[
                    n_ens_member, i, 30:165, 41 + 1 * (i + 1) * n_ens_member
                ] = 0.1
        if n_ens_members < expected_n_ens_members:
            n_ens_members = expected_n_ens_members

    elif nowcasting_method == "external_nowcast_det":
        nowcasting_method = "external_nowcast"
        for i in range(precip_nowcast.shape[1]):
            precip_nowcast[0, i, 30:165, 30 + 1 * i] = 0.1
            precip_nowcast[0, i, 30:165, 31 + 1 * i] = 0.5
            precip_nowcast[0, i, 30:165, 32 + 1 * i] = 0.5
            precip_nowcast[0, i, 30:165, 33 + 1 * i] = 5.0
            precip_nowcast[0, i, 30:165, 34 + 1 * i] = 5.0
            precip_nowcast[0, i, 30:165, 35 + 1 * i] = 4.5
            precip_nowcast[0, i, 30:165, 36 + 1 * i] = 4.5
            precip_nowcast[0, i, 30:165, 37 + 1 * i] = 4.0
            precip_nowcast[0, i, 30:165, 38 + 1 * i] = 1.0
            precip_nowcast[0, i, 30:165, 39 + 1 * i] = 0.5
            precip_nowcast[0, i, 30:165, 40 + 1 * i] = 0.5
            precip_nowcast[0, i, 30:165, 41 + 1 * i] = 0.1

    metadata = dict()
    metadata["unit"] = "mm"
    metadata["transformation"] = "dB"
    metadata["accutime"] = 5.0
    metadata["transform"] = "dB"
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = 0.01
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6

    # Also set the outdir_path, clim_kwargs and mask_kwargs
    outdir_path_skill = "./tmp/"
    if n_models == 1:
        clim_kwargs = None
    else:
        clim_kwargs = dict({"n_models": n_models, "window_length": 30})

    if max_mask_rim is not None:
        mask_kwargs = dict({"mask_rim": 10, "max_mask_rim": max_mask_rim})
    else:
        mask_kwargs = None

    ###
    # First threshold the data and convert it to dBR
    ###
    # threshold the data
    radar_precip[radar_precip < metadata["threshold"]] = 0.0
    nwp_precip[nwp_precip < metadata["threshold"]] = 0.0

    # convert the data
    converter = pysteps.utils.get_method("mm/h")
    radar_precip, _ = converter(radar_precip, metadata)
    nwp_precip, metadata = converter(nwp_precip, metadata)

    # transform the data
    transformer = pysteps.utils.get_method(metadata["transformation"])
    radar_precip, _ = transformer(radar_precip, metadata)
    nwp_precip, metadata = transformer(nwp_precip, metadata)

    # set NaN equal to zero
    radar_precip[~np.isfinite(radar_precip)] = metadata["zerovalue"]
    nwp_precip[~np.isfinite(nwp_precip)] = metadata["zerovalue"]

    assert (
        np.any(~np.isfinite(radar_precip)) == False
    ), "There are still infinite values in the input radar data"
    assert (
        np.any(~np.isfinite(nwp_precip)) == False
    ), "There are still infinite values in the NWP data"

    ###
    # Decompose the R_NWP data
    ###

    # Initial decomposition settings
    decomp_method, _ = cascade.get_method("fft")
    bandpass_filter_method = "gaussian"
    precip_shape = radar_precip.shape[1:]
    filter_method = cascade.get_method(bandpass_filter_method)
    bp_filter = filter_method(precip_shape, n_cascade_levels)

    # If we only use one model:
    if nwp_precip.ndim == 3:
        nwp_precip = nwp_precip[None, :]

    if decomposed_nwp:
        nwp_precip_decomp = []
        # Loop through the n_models
        for i in range(nwp_precip.shape[0]):
            R_d_models_ = []
            # Loop through the time steps
            for j in range(nwp_precip.shape[1]):
                R_ = decomp_method(
                    field=nwp_precip[i, j, :, :],
                    bp_filter=bp_filter,
                    normalize=True,
                    compute_stats=True,
                    compact_output=True,
                )
                R_d_models_.append(R_)
            nwp_precip_decomp.append(R_d_models_)

        nwp_precip_decomp = np.array(nwp_precip_decomp)

        assert nwp_precip_decomp.ndim == 2, "Wrong number of dimensions in R_d_models"

    else:
        nwp_precip_decomp = nwp_precip.copy()

        assert nwp_precip_decomp.ndim == 4, "Wrong number of dimensions in R_d_models"

    ###
    # Determine the velocity fields
    ###
    oflow_method = pysteps.motion.get_method("lucaskanade")
    radar_velocity = oflow_method(radar_precip)
    nwp_velocity = []
    # Loop through the models
    for n_model in range(nwp_precip.shape[0]):
        # Loop through the timesteps. We need two images to construct a motion
        # field, so we can start from timestep 1. Timestep 0 will be the same
        # as timestep 0.
        _V_NWP_ = []
        for t in range(1, nwp_precip.shape[1]):
            V_NWP_ = oflow_method(nwp_precip[n_model, t - 1 : t + 1, :])
            _V_NWP_.append(V_NWP_)
            V_NWP_ = None
        _V_NWP_ = np.insert(_V_NWP_, 0, _V_NWP_[0], axis=0)
        nwp_velocity.append(_V_NWP_)

    nwp_velocity = np.stack(nwp_velocity)

    assert nwp_velocity.ndim == 5, "nwp_velocity must be a five-dimensional array"

    ###
    # Shared forecast kwargs
    ###
    forecast_kwargs = dict(
        precip_models=nwp_precip_decomp,
        velocity=radar_velocity,
        velocity_models=nwp_velocity,
        timesteps=timesteps,
        timestep=5.0,
        issuetime=datetime.datetime.strptime("202112012355", "%Y%m%d%H%M"),
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        blend_nwp_members=blend_nwp_members,
        precip_thr=metadata["threshold"],
        kmperpixel=1.0,
        extrap_method="semilagrangian",
        decomp_method="fft",
        bandpass_filter_method="gaussian",
        noise_method="nonparametric",
        noise_stddev_adj="auto",
        ar_order=2,
        vel_pert_method=vel_pert_method,
        weights_method=weights_method,
        timestep_start_full_nwp_weight=timestep_start_full_nwp_weight,
        conditional=False,
        probmatching_method=probmatching_method,
        mask_method=mask_method,
        resample_distribution=resample_distribution,
        smooth_radar_mask_range=smooth_radar_mask_range,
        callback=None,
        return_output=True,
        seed=None,
        num_workers=1,
        fft_method="numpy",
        domain="spatial",
        outdir_path_skill=outdir_path_skill,
        extrap_kwargs=None,
        filter_kwargs=None,
        noise_kwargs=None,
        vel_pert_kwargs=None,
        clim_kwargs=clim_kwargs,
        mask_kwargs=mask_kwargs,
        measure_time=False,
    )

    ###
    # The blending
    ###
    # Test with full radar data
    run_and_assert_forecast(
        radar_precip,
        forecast_kwargs,
        expected_n_ens_members,
        n_timesteps,
        converter,
        metadata,
    )


def _make_external_nowcast_weight_inputs():
    """Build small dummy radar, NWP and external-nowcast fields plus the shared
    forecast kwargs used by the external-nowcast weight regression tests.

    Returns ``(radar_precip, precip_nowcast, common_kwargs)`` where
    ``common_kwargs`` omits ``precip``, ``precip_nowcast``, ``nowcasting_method``
    and ``noise_method`` so each test can set those per run.
    """
    n_cascade_levels = 6
    timesteps = 8

    nwp_precip = np.zeros((1, timesteps + 1, 200, 200))
    for i in range(nwp_precip.shape[1]):
        nwp_precip[0, i, 30:185, 30 + i] = 0.1
        nwp_precip[0, i, 30:185, 31 + i] = 0.1
        nwp_precip[0, i, 30:185, 32 + i] = 1.0
        nwp_precip[0, i, 30:185, 33 + i] = 5.0
        nwp_precip[0, i, 30:185, 34 + i] = 5.0
        nwp_precip[0, i, 30:185, 35 + i] = 4.5
        nwp_precip[0, i, 30:185, 36 + i] = 4.5
        nwp_precip[0, i, 30:185, 37 + i] = 4.0
        nwp_precip[0, i, 30:185, 38 + i] = 2.0
        nwp_precip[0, i, 30:185, 39 + i] = 1.0

    radar_precip = np.zeros((3, 200, 200))
    for i in range(3):
        radar_precip[i, 5:150, 30 + i] = 0.1
        radar_precip[i, 5:150, 31 + i] = 0.5
        radar_precip[i, 5:150, 32 + i] = 0.5
        radar_precip[i, 5:150, 33 + i] = 5.0
        radar_precip[i, 5:150, 34 + i] = 5.0
        radar_precip[i, 5:150, 35 + i] = 4.5
        radar_precip[i, 5:150, 36 + i] = 4.5
        radar_precip[i, 5:150, 37 + i] = 4.0
        radar_precip[i, 5:150, 38 + i] = 1.0
        radar_precip[i, 5:150, 39 + i] = 0.5

    precip_nowcast = np.zeros((1, timesteps + 1, 200, 200))
    for i in range(precip_nowcast.shape[1]):
        precip_nowcast[0, i, 30:165, 30 + i] = 0.1
        precip_nowcast[0, i, 30:165, 31 + i] = 0.5
        precip_nowcast[0, i, 30:165, 32 + i] = 0.5
        precip_nowcast[0, i, 30:165, 33 + i] = 5.0
        precip_nowcast[0, i, 30:165, 34 + i] = 5.0
        precip_nowcast[0, i, 30:165, 35 + i] = 4.5
        precip_nowcast[0, i, 30:165, 36 + i] = 4.5
        precip_nowcast[0, i, 30:165, 37 + i] = 4.0
        precip_nowcast[0, i, 30:165, 38 + i] = 1.0

    metadata = dict(
        unit="mm",
        transformation="dB",
        accutime=5.0,
        transform="dB",
        zerovalue=0.0,
        threshold=0.01,
        zr_a=200.0,
        zr_b=1.6,
    )

    radar_precip[radar_precip < metadata["threshold"]] = 0.0
    nwp_precip[nwp_precip < metadata["threshold"]] = 0.0
    precip_nowcast[precip_nowcast < metadata["threshold"]] = 0.0

    converter = pysteps.utils.get_method("mm/h")
    radar_precip, _ = converter(radar_precip, metadata)
    nwp_precip, _ = converter(nwp_precip, metadata)
    precip_nowcast, metadata = converter(precip_nowcast, metadata)

    transformer = pysteps.utils.get_method(metadata["transformation"])
    radar_precip, _ = transformer(radar_precip, metadata)
    nwp_precip, _ = transformer(nwp_precip, metadata)
    precip_nowcast, metadata = transformer(precip_nowcast, metadata)

    radar_precip[~np.isfinite(radar_precip)] = metadata["zerovalue"]
    nwp_precip[~np.isfinite(nwp_precip)] = metadata["zerovalue"]
    precip_nowcast[~np.isfinite(precip_nowcast)] = metadata["zerovalue"]

    oflow_method = pysteps.motion.get_method("lucaskanade")
    radar_velocity = oflow_method(radar_precip)
    _v_nwp = [
        oflow_method(nwp_precip[0, t - 1 : t + 1, :])
        for t in range(1, nwp_precip.shape[1])
    ]
    nwp_velocity = np.stack([np.insert(_v_nwp, 0, _v_nwp[0], axis=0)])

    common_kwargs = dict(
        precip_models=nwp_precip,
        velocity=radar_velocity,
        velocity_models=nwp_velocity,
        timesteps=timesteps,
        timestep=5.0,
        issuetime=datetime.datetime.strptime("202112012355", "%Y%m%d%H%M"),
        n_ens_members=1,
        n_cascade_levels=n_cascade_levels,
        blend_nwp_members=False,
        precip_thr=metadata["threshold"],
        kmperpixel=1.0,
        extrap_method="semilagrangian",
        decomp_method="fft",
        bandpass_filter_method="gaussian",
        noise_stddev_adj="auto",
        ar_order=2,
        vel_pert_method=None,
        weights_method="bps",
        conditional=False,
        probmatching_method=None,
        mask_method=None,
        resample_distribution=False,
        smooth_radar_mask_range=0,
        callback=None,
        return_output=True,
        seed=42,
        num_workers=1,
        fft_method="numpy",
        domain="spatial",
        outdir_path_skill="./tmp/",
        clim_kwargs=None,
        measure_time=False,
    )

    return radar_precip, precip_nowcast, common_kwargs


def _capture_full_blend_weights(forecast_kwargs):
    """Run a blended forecast and return the weights passed to the full-blend
    ``blend_cascades`` calls.

    ``__blend_cascades`` calls ``blend_cascades`` twice per member/subtimestep:
    once with all components (the full blend) and once with the first
    (nowcast/extrapolation) component dropped (the model-only blend, one fewer
    row). The full-blend calls are the ones with the maximum number of rows.
    """
    original = blending.utils.blend_cascades
    captured = []

    def wrapper(cascades_norm, weights):
        captured.append(np.asarray(weights).copy())
        return original(cascades_norm=cascades_norm, weights=weights)

    blending.utils.blend_cascades = wrapper
    try:
        blending.steps.forecast(**forecast_kwargs)
    finally:
        blending.utils.blend_cascades = original

    n_rows = max(w.shape[0] for w in captured)
    return [w for w in captured if w.shape[0] == n_rows]


def test_steps_blending_external_nowcast_weight_distribution():
    """Regression test for the external-nowcast blending weights (no noise).

    Blending a precomputed (external) nowcast with noise disabled must give the
    NWP component the same variance share as the standard STEPS path, where the
    radar side is split over a separate extrapolation and noise cascade. Both
    paths compute the same extrapolation and NWP skill, and BPS weights are
    L2-normalized (sum of squares == 1), so the radar-side and NWP variance
    shares fed to ``blend_cascades`` must match.

    The external path folds the noise weight into the nowcast component in
    quadrature (w_nowcast**2 = w_extrap**2 + w_noise**2). Simply dropping the
    noise weight and L1-renormalizing (the previous behaviour) inflated the NWP
    share and collapsed the forecast onto the NWP far too quickly.
    """
    pytest.importorskip("cv2")

    radar_precip, precip_nowcast, common_kwargs = _make_external_nowcast_weight_inputs()

    def variance_shares(weights_list, has_noise):
        radar_side, nwp_side = [], []
        for w in weights_list:
            if has_noise:
                # [w_extrap, w_nwp..., w_noise]
                radar_side.append(w[0] ** 2 + w[-1] ** 2)
                nwp_side.append(np.sum(w[1:-1] ** 2, axis=0))
            else:
                # [w_nowcast, w_nwp...] with the noise weight folded in
                radar_side.append(w[0] ** 2)
                nwp_side.append(np.sum(w[1:] ** 2, axis=0))
        return np.array(radar_side), np.array(nwp_side)

    # Standard STEPS path: separate extrapolation + noise cascades.
    normal_weights = _capture_full_blend_weights(
        dict(
            precip=radar_precip.copy(),
            nowcasting_method="steps",
            noise_method="nonparametric",
            **common_kwargs,
        )
    )
    normal_radar, normal_nwp = variance_shares(normal_weights, has_noise=True)

    # External-nowcast path with noise disabled.
    external_weights = _capture_full_blend_weights(
        dict(
            precip=radar_precip.copy(),
            precip_nowcast=precip_nowcast,
            nowcasting_method="external_nowcast",
            noise_method=None,
            **common_kwargs,
        )
    )
    external_radar, external_nwp = variance_shares(external_weights, has_noise=False)

    assert normal_nwp.size > 0, "No full-blend weights were captured"
    assert (
        normal_nwp.shape == external_nwp.shape
    ), "The two paths produced a different number/shape of blend weights"

    # The NWP (and thus radar-side) variance share must be identical between
    # the standard and external-nowcast paths at every blend call and scale.
    np.testing.assert_allclose(external_nwp, normal_nwp, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(external_radar, normal_radar, rtol=1e-6, atol=1e-8)


def test_steps_blending_external_nowcast_with_noise_weight_distribution():
    """Regression test for a deterministic external nowcast blended *with* noise.

    When a deterministic (e.g. advection-only) external nowcast is provided and
    ``noise_method`` is left enabled, the separate stochastic noise cascade is
    still generated - exactly as in the standard STEPS path - and the external
    nowcast merely substitutes the extrapolation cascade values. The AR fit
    (``PHI``), extrapolation skill and NWP skill are computed from the radar
    regardless of the external nowcast, so the full three-component weights
    ``[w_extrap/nowcast, w_nwp, w_noise]`` must be identical to the standard
    path, component-for-component.
    """
    pytest.importorskip("cv2")

    radar_precip, precip_nowcast, common_kwargs = _make_external_nowcast_weight_inputs()

    # Standard STEPS path.
    normal_weights = _capture_full_blend_weights(
        dict(
            precip=radar_precip.copy(),
            nowcasting_method="steps",
            noise_method="nonparametric",
            **common_kwargs,
        )
    )

    # External deterministic nowcast, but with the noise cascade left enabled.
    external_weights = _capture_full_blend_weights(
        dict(
            precip=radar_precip.copy(),
            precip_nowcast=precip_nowcast,
            nowcasting_method="external_nowcast",
            noise_method="nonparametric",
            **common_kwargs,
        )
    )

    assert len(normal_weights) > 0, "No full-blend weights were captured"
    assert len(normal_weights) == len(
        external_weights
    ), "The two paths produced a different number of blend weights"

    # Both paths use the full three-component weights, so they must match
    # exactly (extrapolation, NWP and noise weights alike).
    for w_normal, w_external in zip(normal_weights, external_weights):
        assert w_normal.shape[0] == 3
        assert w_external.shape[0] == 3
        np.testing.assert_allclose(w_external, w_normal, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("ar_order", [1, 2])
def test_steps_blending_partial_zero_radar(ar_order):
    """Test that a forecast succeeds when only the 2 latest radar frames have
    precipitation (initiating cell corner case that produces NaN autocorrelations
    for the earlier, all-zero cascade levels)."""
    pytest.importorskip("cv2")

    n_timesteps = 3
    metadata = dict(
        unit="mm",
        transformation="dB",
        accutime=5.0,
        transform="dB",
        zerovalue=0.0,
        threshold=0.01,
        zr_a=200.0,
        zr_b=1.6,
    )

    # Build minimal NWP data (1 model, 4 time steps)
    nwp_precip = np.zeros((1, n_timesteps + 1, 200, 200))
    for i in range(nwp_precip.shape[1]):
        nwp_precip[0, i, 30:185, 32 + i] = 1.0
        nwp_precip[0, i, 30:185, 33 + i] = 5.0
        nwp_precip[0, i, 30:185, 34 + i] = 5.0
        nwp_precip[0, i, 30:185, 35 + i] = 4.5

    # Build radar data: only the latest (most recent) frame has precipitation
    radar_precip = np.zeros((3, 200, 200))
    radar_precip[-2, 40:125, 30] = 0.5
    radar_precip[-2, 40:125, 31] = 4.5
    radar_precip[-2, 40:125, 32] = 4.0
    radar_precip[-2, 40:125, 33] = 2.0
    radar_precip[-1, 30:155, 32] = 1.0
    radar_precip[-1, 30:155, 33] = 5.0
    radar_precip[-1, 30:155, 34] = 5.0
    radar_precip[-1, 30:155, 35] = 4.5

    # Threshold, convert and transform
    radar_precip[radar_precip < metadata["threshold"]] = 0.0
    nwp_precip[nwp_precip < metadata["threshold"]] = 0.0
    converter = pysteps.utils.get_method("mm/h")
    radar_precip, _ = converter(radar_precip, metadata)
    nwp_precip, metadata = converter(nwp_precip, metadata)
    transformer = pysteps.utils.get_method(metadata["transformation"])
    radar_precip, _ = transformer(radar_precip, metadata)
    nwp_precip, metadata = transformer(nwp_precip, metadata)
    radar_precip[~np.isfinite(radar_precip)] = metadata["zerovalue"]
    nwp_precip[~np.isfinite(nwp_precip)] = metadata["zerovalue"]

    # Decompose NWP
    n_cascade_levels = 6
    nwp_precip_decomp = nwp_precip.copy()

    # Velocity fields
    oflow_method = pysteps.motion.get_method("lucaskanade")
    radar_velocity = oflow_method(radar_precip)
    _V_NWP = [
        oflow_method(nwp_precip[0, t - 1 : t + 1, :])
        for t in range(1, nwp_precip.shape[1])
    ]
    nwp_velocity = np.stack([np.insert(_V_NWP, 0, _V_NWP[0], axis=0)])

    run_and_assert_forecast(
        radar_precip,
        dict(
            precip_models=nwp_precip_decomp,
            velocity=radar_velocity,
            velocity_models=nwp_velocity,
            timesteps=n_timesteps,
            timestep=5.0,
            issuetime=datetime.datetime.strptime("202112012355", "%Y%m%d%H%M"),
            n_ens_members=1,
            n_cascade_levels=n_cascade_levels,
            blend_nwp_members=False,
            precip_thr=metadata["threshold"],
            kmperpixel=1.0,
            extrap_method="semilagrangian",
            decomp_method="fft",
            bandpass_filter_method="gaussian",
            noise_method="nonparametric",
            noise_stddev_adj="auto",
            ar_order=ar_order,
            vel_pert_method=None,
            weights_method="bps",
            conditional=False,
            probmatching_method=None,
            mask_method="incremental",
            resample_distribution=False,
            smooth_radar_mask_range=0,
            callback=None,
            return_output=True,
            seed=42,
            num_workers=1,
            fft_method="numpy",
            domain="spatial",
            outdir_path_skill="./tmp/",
            extrap_kwargs=None,
            filter_kwargs=None,
            noise_kwargs=None,
            vel_pert_kwargs=None,
            clim_kwargs=None,
            mask_kwargs=None,
            measure_time=False,
        ),
        expected_n_ens_members=1,
        n_timesteps=n_timesteps,
        converter=converter,
        metadata=metadata,
    )
