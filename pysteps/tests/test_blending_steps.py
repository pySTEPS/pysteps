# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np
import pytest

import pysteps
from pysteps import blending, cascade
from pysteps.blending.utils import preprocess_nwp_data
from pysteps.xarray_helpers import convert_input_to_xarray_dataset

# fmt:off
steps_arg_values = [
    (1, 3, 4, 8, None, None, False, "spn", True, 4, False, False, 0, False, None, None),
    (1, 3, 4, 8, "obs", None, False, "spn", True, 4, False, False, 0, False, None, None),
    (1, 3, 4, 8, "incremental", None, False, "spn", True, 4, False, False, 0, False, None, None),
    (1, 3, 4, 8, None, "mean", False, "spn", True, 4, False, False, 0, False, None, None),
    (1, 3, 4, 8, None, "mean", False, "spn", True, 4, False, False, 0, True, None, None),
    (1, 3, 4, 8, None, "cdf", False, "spn", True, 4, False, False, 0, False, None, None),
    (1, [1, 2, 3], 4, 8, None, "cdf", False, "spn", True, 4, False, False, 0, False, None, None),
    (1, 3, 4, 8, "incremental", "cdf", False, "spn", True, 4, False, False, 0, False, None, None),
    (1, 3, 4, 6, "incremental", "cdf", False, "bps", True, 4, False, False, 0, False, None, None),
    (1, 3, 4, 6, "incremental", "cdf", False, "bps", False, 4, False, False, 0, False, None, None),
    (1, 3, 4, 6, "incremental", "cdf", False, "bps", False, 4, False, False, 0, True, None, None),
    (1, 3, 4, 9, "incremental", "cdf", False, "spn", True, 4, False, False, 0, False, None, None),
    (2, 3, 10, 8, "incremental", "cdf", False, "spn", True, 10, False, False, 0, False, None, None),
    (5, 3, 5, 8, "incremental", "cdf", False, "spn", True, 5, False, False, 0, False, None, None),
    (1, 10, 1, 8, "incremental", "cdf", False, "spn", True, 1, False, False, 0, False, None, None),
    (2, 3, 2, 8, "incremental", "cdf", True, "spn", True, 2, False, False, 0, False, None, None),
    (1, 3, 6, 8, None, None, False, "spn", True, 6, False, False, 0, False, None, None),
    (1, 3, 6, 8, None, None, False, "spn", True, 6, False, False, 0, False, "bps", None),
    #    Test the case where the radar image contains no rain.
    (1, 3, 6, 8, None, None, False, "spn", True, 6, True, False, 0, False, None, None),
    (5, 3, 5, 6, "incremental", "cdf", False, "spn", False, 5, True, False, 0, False, None, None),
    (5, 3, 5, 6, "incremental", "cdf", False, "spn", False, 5, True, False, 0, True, None, None),
    #   Test the case where the NWP fields contain no rain.
    (1, 3, 6, 8, None, None, False, "spn", True, 6, False, True, 0, False, None, None),
    (5, 3, 5, 6, "incremental", "cdf", False, "spn", False, 5, False, True, 0, True, None, None),
    # Test the case where both the radar image and the NWP fields contain no rain.
    (1, 3, 6, 8, None, None, False, "spn", True, 6, True, True, 0, False, None, None),
    (5, 3, 5, 6, "incremental", "cdf", False, "spn", False, 5, True, True, 0, False, None, None),
    (5, 3, 5, 6, "obs", "mean", True, "spn", True, 5, True, True, 0, False, None, None),
    # Test for smooth radar mask
    (1, 3, 6, 8, None, None, False, "spn", True, 6, False, False, 80, False, None, None),
    (5, 3, 5, 6, "incremental", "cdf", False, "spn", False, 5, False, False, 80, False, None, None),
    (5, 3, 5, 6, "obs", "mean", False, "spn", False, 5, False, False, 80, False, None, None),
    (1, 3, 6, 8, None, None, False, "spn", True, 6, False, True, 80, False, None, None),
    (5, 3, 5, 6, "incremental", "cdf", False, "spn", False, 5, True, False, 80, True, None, None),
    (5, 3, 5, 6, "obs", "mean", False, "spn", False, 5, True, True, 80, False, None, None),
    (5, [1, 2, 3], 5, 6, "obs", "mean", False, "spn", False, 5, True, True, 80, False, None, None),
    (5, [1, 3], 5, 6, "obs", "mean", False, "spn", False, 5, True, True, 80, False, None, None),
    # Test the usage of a max_mask_rim in the mask_kwargs
    (1, 3, 6, 8, None, None, False, "bps", True, 6, False, False, 80, False, None, 40),
    (5, 3, 5, 6, "obs", "mean", False, "bps", False, 5, False, False, 80, False, None, 40),
    (5, 3, 5, 6, "incremental", "cdf", False, "bps", False, 5, False, False, 80, False, None, 25),
    (5, 3, 5, 6, "incremental", "cdf", False, "bps", False, 5, False, False, 80, False, None, 40),
    (5, 3, 5, 6, "incremental", "cdf", False, "bps", False, 5, False, False, 80, False, None, 60),
]
# fmt:on

steps_arg_names = (
    "n_models",
    "timesteps",
    "n_ens_members",
    "n_cascade_levels",
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
)


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_steps_blending(
    n_models,
    timesteps,
    n_ens_members,
    n_cascade_levels,
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

    metadata = dict()
    metadata["unit"] = "mm"
    metadata["cartesian_unit"] = "km"
    metadata["accutime"] = 5.0
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = 0.01
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6
    metadata["x1"] = 0.0
    metadata["x2"] = 200.0
    metadata["y1"] = 0.0
    metadata["y2"] = 200.0
    metadata["yorigin"] = "lower"
    metadata["institution"] = "test"
    metadata["projection"] = (
        "+proj=lcc +lon_0=4.55 +lat_1=50.8 +lat_2=50.8 +a=6371229 +es=0 +lat_0=50.8 +x_0=365950 +y_0=-365950.000000001"
    )

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

    radar_dataset = convert_input_to_xarray_dataset(
        radar_precip,
        None,
        metadata,
        datetime.fromisoformat("2021-07-04T11:50:00.000000000"),
        300,
    )
    model_dataset = convert_input_to_xarray_dataset(
        nwp_precip,
        None,
        metadata,
        datetime.fromisoformat("2021-07-04T12:00:00.000000000"),
        300,
    )
    # convert the data
    converter = pysteps.utils.get_method("mm/h")
    radar_dataset = converter(radar_dataset)
    model_dataset = converter(model_dataset)

    # transform the data
    transformer = pysteps.utils.get_method("dB")
    radar_dataset = transformer(radar_dataset)
    model_dataset = transformer(model_dataset)

    radar_precip_var = radar_dataset.attrs["precip_var"]
    model_precip_var = model_dataset.attrs["precip_var"]

    # set NaN equal to zero
    radar_dataset[radar_precip_var].data[
        ~np.isfinite(radar_dataset[radar_precip_var].values)
    ] = radar_dataset[radar_precip_var].attrs["zerovalue"]
    model_dataset[model_precip_var].data[
        ~np.isfinite(model_dataset[model_precip_var].values)
    ] = model_dataset[model_precip_var].attrs["zerovalue"]

    assert (
        np.any(~np.isfinite(radar_dataset[radar_precip_var].values)) == False
    ), "There are still infinite values in the input radar data"
    assert (
        np.any(~np.isfinite(model_dataset[radar_precip_var].values)) == False
    ), "There are still infinite values in the NWP data"

    ###
    # Decompose the R_NWP data
    ###

    radar_precip = radar_dataset[radar_precip_var].values

    oflow_method = pysteps.motion.get_method("lucaskanade")
    nwp_preproc_dataset = preprocess_nwp_data(
        model_dataset,
        oflow_method,
        "test",
        None,
        decomposed_nwp,
        {"num_cascade_levels": n_cascade_levels},
    )

    ###
    # Determine the velocity fields
    ###
    radar_dataset_w_velocity = oflow_method(radar_dataset)

    ###
    # The nowcasting
    ###
    precip_forecast_dataset = blending.steps.forecast(
        radar_dataset=radar_dataset_w_velocity,
        model_dataset=nwp_preproc_dataset,
        timesteps=timesteps,
        timestep=5.0,
        issuetime=datetime.fromisoformat("2021-07-04T12:00:00.000000000"),
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
    precip_var_forecast = precip_forecast_dataset.attrs["precip_var"]
    precip_forecast = precip_forecast_dataset[precip_var_forecast].values

    assert precip_forecast.ndim == 4, "Wrong amount of dimensions in forecast output"
    assert (
        precip_forecast.shape[0] == expected_n_ens_members
    ), "Wrong amount of output ensemble members in forecast output"
    assert (
        precip_forecast.shape[1] == n_timesteps
    ), "Wrong amount of output time steps in forecast output"

    # Transform the data back into mm/h
    precip_forecast_dataset = converter(precip_forecast_dataset)
    precip_var_forecast = precip_forecast_dataset.attrs["precip_var"]
    precip_forecast = precip_forecast_dataset[precip_var_forecast].values

    assert (
        precip_forecast.ndim == 4
    ), "Wrong amount of dimensions in converted forecast output"
    assert (
        precip_forecast.shape[0] == expected_n_ens_members
    ), "Wrong amount of output ensemble members in converted forecast output"
    assert (
        precip_forecast.shape[1] == n_timesteps
    ), "Wrong amount of output time steps in converted forecast output"
