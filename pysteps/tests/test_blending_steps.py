# -*- coding: utf-8 -*-

import os
import numpy as np
import datetime
import pytest
import pysteps
from pysteps import cascade, blending


steps_arg_names = (
    "n_models",
    "n_timesteps",
    "n_ens_members",
    "n_cascade_levels",
    "mask_method",
    "probmatching_method",
    "blend_nwp_members",
    "weights_method",
    "decomposed_nwp",
    "expected_n_ens_members",
)

steps_arg_values = [
    (1, 3, 4, 8, None, None, False, "spn", True, 4),
    (1, 3, 4, 8, "obs", None, False, "spn", True, 4),
    (1, 3, 4, 8, "incremental", None, False, "spn", True, 4),
    (1, 3, 4, 8, None, "mean", False, "spn", True, 4),
    (1, 3, 4, 8, None, "cdf", False, "spn", True, 4),
    (1, 3, 4, 8, "incremental", "cdf", False, "spn", True, 4),
    (1, 3, 4, 6, "incremental", "cdf", False, "bps", True, 4),
    (1, 3, 4, 6, "incremental", "cdf", False, "bps", False, 4),
    (1, 3, 4, 9, "incremental", "cdf", False, "spn", True, 4),
    (2, 3, 10, 8, "incremental", "cdf", False, "spn", True, 10),
    (5, 3, 4, 8, "incremental", "cdf", False, "spn", True, 5),
    (1, 10, 1, 8, "incremental", "cdf", False, "spn", True, 1),
    (5, 3, 2, 8, "incremental", "cdf", True, "spn", True, 2),
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
    weights_method,
    decomposed_nwp,
    expected_n_ens_members,
):
    ###
    # The input data
    ###
    # Initialise dummy NWP data
    R_NWP = np.zeros((n_models, n_timesteps + 1, 200, 200))

    for n_model in range(n_models):
        for i in range(R_NWP.shape[1]):
            R_NWP[n_model, i, 30:185, 30 + 1 * (i + 1) * n_model] = 0.1
            R_NWP[n_model, i, 30:185, 31 + 1 * (i + 1) * n_model] = 0.1
            R_NWP[n_model, i, 30:185, 32 + 1 * (i + 1) * n_model] = 1.0
            R_NWP[n_model, i, 30:185, 33 + 1 * (i + 1) * n_model] = 5.0
            R_NWP[n_model, i, 30:185, 34 + 1 * (i + 1) * n_model] = 5.0
            R_NWP[n_model, i, 30:185, 35 + 1 * (i + 1) * n_model] = 4.5
            R_NWP[n_model, i, 30:185, 36 + 1 * (i + 1) * n_model] = 4.5
            R_NWP[n_model, i, 30:185, 37 + 1 * (i + 1) * n_model] = 4.0
            R_NWP[n_model, i, 30:185, 38 + 1 * (i + 1) * n_model] = 2.0
            R_NWP[n_model, i, 30:185, 39 + 1 * (i + 1) * n_model] = 1.0
            R_NWP[n_model, i, 30:185, 40 + 1 * (i + 1) * n_model] = 0.5
            R_NWP[n_model, i, 30:185, 41 + 1 * (i + 1) * n_model] = 0.1

    # Define dummy nowcast input data
    R_input = np.zeros((3, 200, 200))

    for i in range(2):
        R_input[i, 5:150, 30 + 1 * i] = 0.1
        R_input[i, 5:150, 31 + 1 * i] = 0.5
        R_input[i, 5:150, 32 + 1 * i] = 0.5
        R_input[i, 5:150, 33 + 1 * i] = 5.0
        R_input[i, 5:150, 34 + 1 * i] = 5.0
        R_input[i, 5:150, 35 + 1 * i] = 4.5
        R_input[i, 5:150, 36 + 1 * i] = 4.5
        R_input[i, 5:150, 37 + 1 * i] = 4.0
        R_input[i, 5:150, 38 + 1 * i] = 1.0
        R_input[i, 5:150, 39 + 1 * i] = 0.5
        R_input[i, 5:150, 40 + 1 * i] = 0.5
        R_input[i, 5:150, 41 + 1 * i] = 0.1
    R_input[2, 30:155, 30 + 1 * 2] = 0.1
    R_input[2, 30:155, 31 + 1 * 2] = 0.1
    R_input[2, 30:155, 32 + 1 * 2] = 1.0
    R_input[2, 30:155, 33 + 1 * 2] = 5.0
    R_input[2, 30:155, 34 + 1 * 2] = 5.0
    R_input[2, 30:155, 35 + 1 * 2] = 4.5
    R_input[2, 30:155, 36 + 1 * 2] = 4.5
    R_input[2, 30:155, 37 + 1 * 2] = 4.0
    R_input[2, 30:155, 38 + 1 * 2] = 2.0
    R_input[2, 30:155, 39 + 1 * 2] = 1.0
    R_input[2, 30:155, 40 + 1 * 3] = 0.5
    R_input[2, 30:155, 41 + 1 * 3] = 0.1

    metadata = dict()
    metadata["unit"] = "mm"
    metadata["transformation"] = "dB"
    metadata["accutime"] = 5.0
    metadata["transform"] = "dB"
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = 0.01
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6

    # Also set the outdir_path and clim_kwargs
    outdir_path_skill = "./tmp/"
    clim_kwargs = dict({"n_models": n_models, "window_length": 30})

    ###
    # First threshold the data and convert it to dBR
    ###
    # threshold the data
    R_input[R_input < metadata["threshold"]] = 0.0
    R_NWP[R_NWP < metadata["threshold"]] = 0.0

    # convert the data
    converter = pysteps.utils.get_method("mm/h")
    R_input, _ = converter(R_input, metadata)
    R_NWP, metadata = converter(R_NWP, metadata)

    # transform the data
    transformer = pysteps.utils.get_method(metadata["transformation"])
    R_input, _ = transformer(R_input, metadata)
    R_NWP, metadata = transformer(R_NWP, metadata)

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

    if decomposed_nwp:
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

    else:
        R_d_models = R_NWP.copy()

        assert R_d_models.ndim == 4, "Wrong number of dimensions in R_d_models"

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
        issuetime=datetime.datetime.strptime("202112012355", "%Y%m%d%H%M"),
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
        vel_pert_method=None,
        weights_method=weights_method,
        conditional=False,
        probmatching_method=probmatching_method,
        mask_method=mask_method,
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