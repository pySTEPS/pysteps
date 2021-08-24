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
    "expected_n_ens_members",
)

steps_arg_values = [
    (1, 3, 4, 8, None, None, 4),
    (1, 3, 4, 8, "obs", None, 4),
    (1, 3, 4, 8, "sprog", None, 4),
    (1, 3, 4, 8, "incremental", None, 4),
    (1, 3, 4, 8, None, "mean", 4),
    (1, 3, 4, 8, None, "cdf", 4),
    (1, 3, 4, 8, "incremental", "cdf", 4),
    (1, 3, 4, 6, "incremental", "cdf", 4),
    (1, 3, 4, 9, "incremental", "cdf", 4),
    (2, 3, 10, 8, "incremental", "cdf", 10),
    (5, 3, 4, 8, "incremental", "cdf", 5),
    (1, 10, 1, 8, "incremental", "cdf", 1),
]


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_steps_blending(
    n_models,
    n_timesteps,
    n_ens_members,
    n_cascade_levels,
    mask_method,
    probmatching_method,
    expected_n_ens_members,
):
    ###
    # The input data
    ###
    # Initialise dummy NWP data
    R_NWP = np.zeros((n_models, n_timesteps + 1, 200, 200))
    print(R_NWP.shape)

    for i in range(R_NWP.shape[1]):
        R_NWP[:, i, :, 26 + 2 * i] = 0.5
        R_NWP[:, i, :, 27 + 2 * i] = 0.5
        R_NWP[:, i, :, 28 + 2 * i] = 0.5
        R_NWP[:, i, :, 29 + 2 * i] = 0.5
        R_NWP[:, i, :, 30 + 2 * i] = 1.0
        R_NWP[:, i, :, 31 + 2 * i] = 1.0
        R_NWP[:, i, :, 32 + 2 * i] = 5.0
        R_NWP[:, i, :, 35 + 2 * i] = 5.0
        R_NWP[:, i, :, 36 + 2 * i] = 5.0
        R_NWP[:, i, :, 37 + 2 * i] = 5.0
        R_NWP[:, i, :, 38 + 2 * i] = 1.0
        R_NWP[:, i, :, 39 + 2 * i] = 1.0
        R_NWP[:, i, :, 40 + 2 * i] = 1.0
        R_NWP[:, i, :, 41 + 2 * i] = 0.5
        R_NWP[:, i, :, 41 + 2 * i] = 0.5
        R_NWP[:, i, :, 42 + 2 * i] = 0.5

    # Define dummy nowcast input data
    R_input = np.zeros((3, 200, 200))

    for i in range(3):
        R_input[i, :, 35 + 1 * i] = 0.5
        R_input[i, :, 36 + 1 * i] = 1.0
        R_input[i, :, 37 + 1 * i] = 5.0
        R_input[i, :, 38 + 1 * i] = 1.0
        R_input[i, :, 39 + 1 * i] = 0.5

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

    # Also obtain the fields from the second cascade for the velocity field
    R_NWP_lev2 = []
    for n_model in range(R_d_models.shape[0]):
        R_NWP_lev2_ = []
        for t in range(R_d_models.shape[1]):
            R_NWP_lev2_.append(R_d_models[n_model, t]["cascade_levels"][1])
        R_NWP_lev2.append(R_NWP_lev2_)
    R_NWP_lev2 = np.array(R_NWP_lev2)

    ###
    # Determine the velocity fields
    ###
    oflow_method = pysteps.motion.get_method("lucaskanade")
    V_radar = oflow_method(R_input)
    V_NWP = []
    for n_model in range(R_NWP_lev2.shape[0]):
        V_NWP_ = oflow_method(R_NWP_lev2[n_model, :])
        V_NWP.append(V_NWP_)

    V_NWP = np.stack(V_NWP)

    assert V_NWP.ndim == 4, "V_NWP must be a four-dimensional array"

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
        blend_nwp_members=False,
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
