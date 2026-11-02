# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pytest

from pysteps import blending, motion, utils
from pysteps.xarray_helpers import convert_input_to_xarray_dataset

# fmt: off
pca_enkf_arg_values = [
    # Standard setting
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False,0,False),
    # Smooth radar mask
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False,20,False),
    # Coarser NWP temporal resolution
    (20,30,0,-60,False,False,5,15,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False,0,False),
    # Coarser Obs temporal resolution
    (20,30,0,-60,False,False,10,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False,0,False),
    # Larger shift of the NWP init
    (20,30,0,-30,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False,0,False),
    # Zero rain case in observation
    (20,30,0,-60,True,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False,0,False),
    # Zero rain case in NWP
    (20,30,0,-60,False,True,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False,0,False),
    # Zero rain in both
    (20,30,0,-60,True,True,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False,0,False),
    # Accumulated sampling probability
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",True,False,0,False),
    # Use full NWP weight
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,True,0,False),
    # Both
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",True,True,0,False),
    # Explained variance as sampling probability source
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"explained_var",False,False,0,False),
    # No combination
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",False,None,1.0,"ensemble",False,False,0,False),
    # Standard deviation adjustment
    (20,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,"auto",1.0,"ensemble",False,False,0,False),
    # Other number of ensemble members
    (10,30,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False,0,False),
    # Other forecast length
    (20,35,0,-60,False,False,5,5,0.05,0.01,"ssft","masked_enkf",True,None,1.0,"ensemble",False,False,0,False),
    # Other noise method
    (20,30,0,-60,False,False,5,5,0.05,0.01,"nonparametric","masked_enkf",True,None,1.0,"ensemble",False,False,0,False),
    # Verbose output
    (20,30,0,-60,False,False,5,5,0.05,0.01,"nonparametric","masked_enkf",True,None,1.0,"ensemble",False,False,0,True),]
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
    "smooth_radar_mask_range",
    "verbose_output",
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
    smooth_radar_mask_range,
    verbose_output,
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
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = thr_prec
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6
    metadata["x1"] = 0.0
    metadata["x2"] = 200.0
    metadata["y1"] = 0.0
    metadata["y2"] = 200.0
    metadata["yorigin"] = "lower"
    metadata["institution"] = "test"
    metadata["cartesian_unit"] = "km"
    metadata["projection"] = (
        "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0"
    )

    # Converting the input data
    # Thresholding
    radar_precip[radar_precip < metadata["threshold"]] = 0.0
    nwp_precip[nwp_precip < metadata["threshold"]] = 0.0

    # Build the radar and NWP datasets, using the actual timestamps of the
    # synthetic data so that the datasets' time coordinates match
    # `*_precip_timestamps`.
    radar_dataset = convert_input_to_xarray_dataset(
        radar_precip,
        None,
        metadata,
        radar_precip_timestamps[0],
        temporal_res_radar * 60,
    )
    model_dataset = convert_input_to_xarray_dataset(
        nwp_precip,
        None,
        metadata,
        nwp_precip_timestamps[0],
        temporal_res_nwp * 60,
    )

    # Convert the data
    converter_name = "mm/h"
    converter = utils.get_method(converter_name)
    radar_dataset = converter(radar_dataset)
    model_dataset = converter(model_dataset)

    # Transform the data
    transformer_name = "dB"
    transformer = utils.get_method(transformer_name)
    radar_dataset = transformer(radar_dataset)
    model_dataset = transformer(model_dataset)

    radar_precip_var = radar_dataset.attrs["precip_var"]
    model_precip_var = model_dataset.attrs["precip_var"]

    # Set NaN equal to zero
    radar_dataset[radar_precip_var].values[
        ~np.isfinite(radar_dataset[radar_precip_var].values)
    ] = radar_dataset[radar_precip_var].attrs["zerovalue"]
    model_dataset[model_precip_var].values[
        ~np.isfinite(model_dataset[model_precip_var].values)
    ] = model_dataset[model_precip_var].attrs["zerovalue"]

    assert (
        np.any(~np.isfinite(radar_dataset[radar_precip_var].values)) == False
    ), "There are still infinite values in the input radar data"
    assert (
        np.any(~np.isfinite(model_dataset[model_precip_var].values)) == False
    ), "There are still infinite values in the NWP data"

    # Initialize radar velocity
    oflow_method = motion.get_method("LK")
    radar_dataset = oflow_method(radar_dataset)

    # The precipitation threshold has been transformed into dB units above by
    # the transformer; use the transformed value for the forecast call.
    precip_thr = radar_dataset[radar_precip_var].attrs["threshold"]

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
    combined_forecast_dataset = blending.pca_ens_kalman_filter.forecast(
        radar_dataset=radar_dataset,
        model_dataset=model_dataset,
        forecast_horizon=forecast_length,
        issuetime=forecast_init,
        n_ens_members=n_ens_members,
        precip_mask_dilation=1,
        smooth_radar_mask_range=smooth_radar_mask_range,
        n_cascade_levels=6,
        precip_thr=precip_thr,
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
        verbose_output=verbose_output,
    )

    if verbose_output:
        assert len(combined_forecast_dataset) == 2, "Wrong amount of output data"
        combined_forecast_dataset = combined_forecast_dataset[0]

    combined_forecast = combined_forecast_dataset[radar_precip_var].values

    assert combined_forecast.ndim == 4, "Wrong amount of dimensions in forecast output"
    assert (
        combined_forecast.shape[0] == n_ens_members
    ), "Wrong amount of output ensemble members in forecast output"
    assert (
        combined_forecast.shape[1] == forecast_length // temporal_res_radar + 1
    ), "Wrong amount of output time steps in forecast output"

    # Transform the data back into mm/h
    combined_forecast_mmh_dataset = converter(combined_forecast_dataset)
    combined_forecast = combined_forecast_mmh_dataset[radar_precip_var].values

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
