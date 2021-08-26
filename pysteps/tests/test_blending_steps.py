# -*- coding: utf-8 -*-

import os
import numpy as np
import pytest
import pysteps
from pysteps import cascade, blending


# TODO: Fix tests for xarray fields?

steps_arg_names = (
    "n_models",
    "n_timesteps",
    "n_ens_members",
    "n_cascade_levels",
    "mask_method",
    "probmatching_method",
    "blend_nwp_members",
    "expected_n_ens_members",
)

steps_arg_values = [
    (1, 3, 4, 8, None, None, False, 4),
    (1, 3, 4, 8, "obs", None, False, 4),
    (1, 3, 4, 8, "incremental", None, False, 4),
    (1, 3, 4, 8, None, "mean", False, 4),
    (1, 3, 4, 8, None, "cdf", False, 4),
    (1, 3, 4, 8, "incremental", "cdf", False, 4),
    (1, 3, 4, 6, "incremental", "cdf", False, 4),
    (1, 3, 4, 9, "incremental", "cdf", False, 4),
    (2, 3, 10, 8, "incremental", "cdf", False, 10),
    (5, 3, 4, 8, "incremental", "cdf", False, 5),
    (1, 10, 1, 8, "incremental", "cdf", False, 1),
    (5, 3, 2, 8, "incremental", "cdf", True, 2),
]


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_steps_blending(
    n_models,
    n_timesteps,
    n_ens_members,
    n_cascade_levels,
    mask_method,
    probmatching_method,
    blend_nwp_members,
    expected_n_ens_members,
):
    ###
    # The input data
    ###
    # Initialise dummy NWP data
    R_NWP = np.zeros((n_models, n_timesteps + 1, 200, 200))

    for i in range(R_NWP.shape[1]):
        R_NWP[:, i, 5:175, 23 + 2 * i] = 0.5
        R_NWP[:, i, 5:175, 24 + 2 * i] = 0.5
        R_NWP[:, i, 5:175, 25 + 2 * i] = 0.5
        R_NWP[:, i, 5:175, 26 + 2 * i] = 0.5
        R_NWP[:, i, 5:175, 27 + 2 * i] = 1.0
        R_NWP[:, i, 5:175, 28 + 2 * i] = 1.0
        R_NWP[:, i, 5:175, 29 + 2 * i] = 5.0
        R_NWP[:, i, 5:175, 30 + 2 * i] = 5.0
        R_NWP[:, i, 5:175, 31 + 2 * i] = 5.0
        R_NWP[:, i, 5:175, 32 + 2 * i] = 5.0
        R_NWP[:, i, 5:175, 33 + 2 * i] = 1.0
        R_NWP[:, i, 5:175, 34 + 2 * i] = 1.0
        R_NWP[:, i, 5:175, 35 + 2 * i] = 1.0
        R_NWP[:, i, 5:175, 36 + 2 * i] = 0.5
        R_NWP[:, i, 5:175, 37 + 2 * i] = 0.5
        R_NWP[:, i, 5:175, 38 + 2 * i] = 0.5

    # Define dummy nowcast input data
    R_input = np.zeros((3, 200, 200))

    for i in range(3):
        R_input[i, 5:150, 31 + 1 * i] = 0.5
        R_input[i, 5:150, 32 + 1 * i] = 1.0
        R_input[i, 5:150, 33 + 1 * i] = 5.0
        R_input[i, 5:150, 34 + 1 * i] = 5.0
        R_input[i, 5:150, 35 + 1 * i] = 4.5
        R_input[i, 5:150, 37 + 1 * i] = 4.5
        R_input[i, 5:150, 38 + 1 * i] = 4.0
        R_input[i, 5:150, 39 + 1 * i] = 2.0
        R_input[i, 5:150, 40 + 1 * i] = 1.0
        R_input[i, 5:150, 41 + 1 * i] = 0.5

    metadata = dict()
    metadata["unit"] = "mm/h"
    metadata["transformation"] = "dB"
    metadata["accutime"] = 5.0
    metadata["unit"] = "mm"
    metadata["transform"] = "dB"
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = 0.01
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6

    ###
    # First threshold the data and convert it to dBR
    ###
    # threshold the data
    R_input[R_input < metadata["threshold"]] = 0.0
    R_NWP[R_NWP < metadata["threshold"]] = 0.0

    # convert the data
    converter = pysteps.utils.get_method(metadata["unit"])
    R_input, metadata = converter(R_input, metadata)
    R_NWP, _ = converter(R_NWP, metadata)

    # transform the data
    transformer = pysteps.utils.get_method(metadata["transformation"])
    R_input, metadata = transformer(R_input, metadata)
    R_NWP, _ = transformer(R_NWP, metadata)

    # set NaN equal to zero
    R_input[~np.isfinite(R_input)] = metadata["zerovalue"]
    R_NWP[~np.isfinite(R_NWP)] = metadata["zerovalue"]

    assert (
        np.any(~np.isfinite(R_input)) == False
    ), "There are still infinite values in the input radar data"
    assert (
        np.any(~np.isfinite(R_NWP)) == False
    ), "There are still infinite values in the NWP data"

    ###
    # Decompose the R_NWP data
    ###

    # Initial decomposition settings
    decomp_method, recomp_method = cascade.get_method("fft")
    bandpass_filter_method = "gaussian"
    M, N = R_input.shape[1:]
    filter_method = cascade.get_method(bandpass_filter_method)
    filter = filter_method((M, N), n_cascade_levels)

    # If we only use one model:
    if R_NWP.ndim == 3:
        R_NWP = R_NWP[None, :]

    R_d_models = []
    # Loop through the n_models
    for i in range(R_NWP.shape[0]):
        R_d_models_ = []
        # Loop through the time steps
        for j in range(R_NWP.shape[1]):
            R_ = decomp_method(
                field=R_NWP[i, j, :, :],
                bp_filter=filter,
                normalize=True,
                compute_stats=True,
                compact_output=True,
            )
            R_d_models_.append(R_)
        R_d_models.append(R_d_models_)

    R_d_models = np.array(R_d_models)

    assert R_d_models.ndim == 2, "Wrong number of dimensions in R_d_models"

    ###
    # Determine the velocity fields
    ###
    oflow_method = pysteps.motion.get_method("lucaskanade")
    V_radar = oflow_method(R_input)
    V_NWP = []
    # Loop through the models
    for n_model in range(R_NWP.shape[0]):
        # Loop through the timesteps. We need two images to construct a motion
        # field, so we can start from timestep 1. Timestep 0 will be the same
        # as timestep 0.
        _V_NWP_ = []
        for t in range(1, R_NWP.shape[1]):
            V_NWP_ = oflow_method(R_NWP[n_model, t - 1 : t + 1, :])
            _V_NWP_.append(V_NWP_)
            V_NWP_ = None
        _V_NWP_ = np.insert(_V_NWP_, 0, _V_NWP_[0], axis=0)
        V_NWP.append(_V_NWP_)

    V_NWP = np.stack(V_NWP)

    assert V_NWP.ndim == 5, "V_NWP must be a five-dimensional array"

    ###
    # The nowcasting
    ###
    precip_forecast = blending.steps.forecast(
        R=R_input,
        R_d_models=R_d_models,
        V=V_radar,
        V_models=V_NWP,
        timesteps=n_timesteps,
        timestep=5.0,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        blend_nwp_members=blend_nwp_members,
        R_thr=metadata["threshold"],
        kmperpixel=1.0,
        extrap_method="semilagrangian",
        decomp_method="fft",
        bandpass_filter_method="gaussian",
        noise_method="nonparametric",
        noise_stddev_adj="auto",
        ar_order=2,
        vel_pert_method="bps",
        conditional=False,
        probmatching_method=probmatching_method,
        mask_method=mask_method,
        callback=None,
        return_output=True,
        seed=None,
        num_workers=1,
        fft_method="numpy",
        domain="spatial",
        extrap_kwargs=None,
        filter_kwargs=None,
        noise_kwargs=None,
        vel_pert_kwargs=None,
        clim_kwargs=None,
        mask_kwargs=None,
        measure_time=False,
    )

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
