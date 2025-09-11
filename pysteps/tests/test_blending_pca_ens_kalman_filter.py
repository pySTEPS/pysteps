# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pytest

from pysteps import blending, motion, utils

# fmt: off
pca_enkf_arg_values = [
    # Standard setting
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False),
    # Coarser NWP temporal resolution
    (20,30,0,-60,False,False,5,15,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False),
    # Coarser Obs temporal resolution
    (20,30,0,-60,False,False,10,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False),
    # Larger shift of the NWP init
    (20,30,0,-30,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False),
    # Zero rain case in observation
    (20,30,0,-60,True,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False),
    # Zero rain case in NWP
    (20,30,0,-60,False,True,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False),
    # Zero rain in both
    (20,30,0,-60,True,True,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False),
    # Accumulated sampling probability
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",True,False),
    # Use full NWP weight
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,True),
    # Both
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",True,True),
    # Explained variance as sampling probability source
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"explained_var",False,False),
    # No combination
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",False,None,1.0,"ensemble",False,False),
    # Standard deviation adjustment
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,"auto",1.0,"ensemble",False,False),
    # Other number of ensemble members
    (10,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False),
    # Other forecast length
    (20,35,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False),
    # Other noise method
    (20,30,0,-60,False,False,5,5,0.05,0.01,"nonparametric","masked_enkf",True,None,1.0,"ensemble",False,False),]
# fmt: on

pca_enkf_arg_names = (
    "n_ens_members",
    "forecast_length",
    "forecast_shift_radar",
    "forecast_shift_nwp",
    "zero_radar",
    "zero_nwp",
    "temporal_res_radar",
    "temporal_res_nwp",
    "thr_prec",
    "norain_thr",
    "noise_method",
    "enkf_method",
    "enable_combination",
    "noise_stddev_adj",
    "inflation_factor_bg",
    "sampling_prob_source",
    "use_accum_sampling_prob",
    "ensure_full_nwp_weight",
)


@pytest.mark.parametrize(pca_enkf_arg_names, pca_enkf_arg_values)
def test_pca_enkf_combination(
    n_ens_members,
    forecast_length,
    forecast_shift_radar,
    forecast_shift_nwp,
    zero_radar,
    zero_nwp,
    temporal_res_radar,
    temporal_res_nwp,
    thr_prec,
    norain_thr,
    noise_method,
    enkf_method,
    enable_combination,
    noise_stddev_adj,
    inflation_factor_bg,
    sampling_prob_source,
    use_accum_sampling_prob,
    ensure_full_nwp_weight,
):
    pytest.importorskip("sklearn")

    # Set forecast init
    forecast_init = datetime.datetime(2025, 6, 4, 17, 0)

    # Initialize dummy radar data
    radar_precip = np.zeros((2, 200, 200))
    if not zero_radar:
        for i in range(radar_precip.shape[0]):
            a = 5 * i
            radar_precip[i, 5 + a : 100 - a, 30 + a : 180 - a] = 0.1
            radar_precip[i, 10 + a : 105 - a, 35 + a : 178 - a] = 0.5
            radar_precip[i, 15 + a : 110 - a, 40 + a : 176 - a] = 0.5
            radar_precip[i, 20 + a : 115 - a, 45 + a : 174 - a] = 5.0
            radar_precip[i, 25 + a : 120 - a, 50 + a : 172 - a] = 5.0
            radar_precip[i, 30 + a : 125 - a, 55 + a : 170 - a] = 4.5
            radar_precip[i, 35 + a : 130 - a, 60 + a : 168 - a] = 4.5
            radar_precip[i, 40 + a : 135 - a, 65 + a : 166 - a] = 4.0
            radar_precip[i, 45 + a : 140 - a, 70 + a : 164 - a] = 1.0
            radar_precip[i, 50 + a : 145 - a, 75 + a : 162 - a] = 0.5
            radar_precip[i, 55 + a : 150 - a, 80 + a : 160 - a] = 0.5
            radar_precip[i, 60 + a : 155 - a, 85 + a : 158 - a] = 0.1

    radar_precip_timestamps = np.array(
        sorted(
            [
                forecast_init
                + datetime.timedelta(minutes=forecast_shift_radar)
                - datetime.timedelta(minutes=i * temporal_res_radar)
                for i in range(radar_precip.shape[0])
            ]
        )
    )

    # Initialize dummy NWP data
    nwp_precip = np.zeros((n_ens_members, 20, 200, 200))
    if not zero_nwp:
        for n_model in range(n_ens_members):
            for i in range(nwp_precip.shape[1]):
                a = 2 * n_model
                b = 2 * i
                nwp_precip[n_model, i, 20 + b : 160 - b, 30 + a : 180 - a] = 0.1
                nwp_precip[n_model, i, 22 + b : 162 - b, 35 + a : 178 - a] = 0.1
                nwp_precip[n_model, i, 24 + b : 164 - b, 40 + a : 176 - a] = 1.0
                nwp_precip[n_model, i, 26 + b : 166 - b, 45 + a : 174 - a] = 5.0
                nwp_precip[n_model, i, 28 + b : 168 - b, 50 + a : 172 - a] = 5.0
                nwp_precip[n_model, i, 30 + b : 170 - b, 35 + a : 170 - a] = 4.5
                nwp_precip[n_model, i, 32 + b : 172 - b, 40 + a : 168 - a] = 4.5
                nwp_precip[n_model, i, 34 + b : 174 - b, 45 + a : 166 - a] = 4.0
                nwp_precip[n_model, i, 36 + b : 176 - b, 50 + a : 164 - a] = 2.0
                nwp_precip[n_model, i, 38 + b : 178 - b, 55 + a : 162 - a] = 1.0
                nwp_precip[n_model, i, 40 + b : 180 - b, 60 + a : 160 - a] = 0.5
                nwp_precip[n_model, i, 42 + b : 182 - b, 65 + a : 158 - a] = 0.1

    nwp_precip_timestamps = np.array(
        sorted(
            [
                forecast_init
                + datetime.timedelta(minutes=forecast_shift_nwp)
                + datetime.timedelta(minutes=i * temporal_res_nwp)
                for i in range(nwp_precip.shape[1])
            ]
        )
    )

    # Metadata of dummy data is necessary for data conversion
    metadata = dict()
    metadata["unit"] = "mm"
    metadata["transformation"] = "dB"
    metadata["accutime"] = 5.0
    metadata["transform"] = None
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = thr_prec
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6

    # Converting the input data
    # Thresholding
    radar_precip[radar_precip < metadata["threshold"]] = 0.0
    nwp_precip[nwp_precip < metadata["threshold"]] = 0.0

    # Convert the data
    converter = utils.get_method("mm/h")
    radar_precip, _ = converter(radar_precip, metadata)
    nwp_precip, metadata = converter(nwp_precip, metadata)

    # Transform the data
    transformer = utils.get_method(metadata["transformation"])
    radar_precip, _ = transformer(radar_precip, metadata)
    nwp_precip, metadata = transformer(nwp_precip, metadata)

    # Set NaN equal to zero
    radar_precip[~np.isfinite(radar_precip)] = metadata["zerovalue"]
    nwp_precip[~np.isfinite(nwp_precip)] = metadata["zerovalue"]

    assert (
        np.any(~np.isfinite(radar_precip)) == False
    ), "There are still infinite values in the input radar data"
    assert (
        np.any(~np.isfinite(nwp_precip)) == False
    ), "There are still infinite values in the NWP data"

    # Initialize radar velocity
    oflow_method = motion.get_method("LK")
    radar_velocity = oflow_method(radar_precip)

    # Set the combination kwargs
    combination_kwargs = dict(
        n_tapering=0,
        non_precip_mask=True,
        n_ens_prec=1,
        lien_criterion=True,
        n_lien=10,
        prob_matching="iterative",
        inflation_factor_bg=inflation_factor_bg,
        inflation_factor_obs=1.0,
        offset_bg=0.0,
        offset_obs=0.0,
        nwp_hres_eff=14.0,
        sampling_prob_source=sampling_prob_source,
        use_accum_sampling_prob=use_accum_sampling_prob,
        ensure_full_nwp_weight=ensure_full_nwp_weight,
    )

    # Call the reduced-spaced ensemble Kalman filter approach.
    combined_forecast = blending.pca_ens_kalman_filter.forecast(
        obs_precip=radar_precip,
        obs_timestamps=radar_precip_timestamps,
        nwp_precip=nwp_precip,
        nwp_timestamps=nwp_precip_timestamps,
        velocity=radar_velocity,
        forecast_horizon=forecast_length,
        issuetime=forecast_init,
        n_ens_members=n_ens_members,
        precip_mask_dilation=1,
        n_cascade_levels=6,
        precip_thr=metadata["threshold"],
        norain_thr=norain_thr,
        extrap_method="semilagrangian",
        decomp_method="fft",
        bandpass_filter_method="gaussian",
        noise_method=noise_method,
        enkf_method=enkf_method,
        enable_combination=enable_combination,
        noise_stddev_adj=noise_stddev_adj,
        ar_order=1,
        callback=None,
        return_output=True,
        seed=None,
        num_workers=1,
        fft_method="numpy",
        domain="spatial",
        extrap_kwargs=None,
        filter_kwargs=None,
        noise_kwargs=None,
        combination_kwargs=combination_kwargs,
        measure_time=False,
    )

    assert combined_forecast.ndim == 4, "Wrong amount of dimensions in forecast output"
    assert (
        combined_forecast.shape[0] == n_ens_members
    ), "Wrong amount of output ensemble members in forecast output"
    assert (
        combined_forecast.shape[1] == forecast_length // temporal_res_radar + 1
    ), "Wrong amount of output time steps in forecast output"

    # Transform the data back into mm/h
    combined_forecast, _ = converter(combined_forecast, metadata)

    assert (
        combined_forecast.ndim == 4
    ), "Wrong amount of dimensions in converted forecast output"
    assert (
        combined_forecast.shape[0] == n_ens_members
    ), "Wrong amount of output ensemble members in converted forecast output"
    assert (
        combined_forecast.shape[1] == forecast_length // temporal_res_radar + 1
    ), "Wrong amount of output time steps in converted forecast output"

    return
