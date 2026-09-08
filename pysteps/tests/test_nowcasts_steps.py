import os
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from pysteps import io, motion, nowcasts, utils, verification
from pysteps.tests.helpers import get_precipitation_fields
from pysteps.xarray_helpers import convert_input_to_xarray_dataset

steps_arg_names = (
    "n_ens_members",
    "n_cascade_levels",
    "ar_order",
    "mask_method",
    "probmatching_method",
    "domain",
    "timesteps",
    "max_crps",
)

steps_arg_values = [
    (5, 6, 2, None, None, "spatial", 3, 1.30),
    (5, 6, 2, None, None, "spatial", [3], 1.30),
    (5, 6, 2, "incremental", None, "spatial", 3, 7.32),
    (5, 6, 2, "sprog", None, "spatial", 3, 8.4),
    (5, 6, 2, "obs", None, "spatial", 3, 8.37),
    (5, 6, 2, None, "cdf", "spatial", 3, 0.60),
    (5, 6, 2, None, "mean", "spatial", 3, 1.35),
    (5, 6, 2, "incremental", "cdf", "spectral", 3, 0.60),
]


def test_default_steps_norain():
    """Tests STEPS nowcast with default params and all-zero inputs."""

    # Define dummy nowcast input data
    dataset_input = xr.Dataset(
        data_vars={"precip_intensity": (["time", "y", "x"], np.zeros((3, 100, 100)))},
        attrs={"precip_var": "precip_intensity"},
    )

    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    retrieved_motion = oflow_method(dataset_input)

    nowcast_method = nowcasts.get_method("steps")
    precip_forecast = nowcast_method(
        retrieved_motion,
        n_ens_members=3,
        timesteps=3,
        precip_thr=0.1,
        kmperpixel=1,
        timestep=5,
    )

    assert precip_forecast.ndim == 4
    assert precip_forecast.shape[0] == 3
    assert precip_forecast.shape[1] == 3
    assert precip_forecast.sum() == 0.0


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_steps_skill(
    n_ens_members,
    n_cascade_levels,
    ar_order,
    mask_method,
    probmatching_method,
    domain,
    timesteps,
    max_crps,
):
    """Tests STEPS nowcast skill."""
    # inputs
    dataset_input = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        upscale=2000,
    )

    dataset_obs = get_precipitation_fields(
        num_prev_files=0, num_next_files=3, return_raw=False, upscale=2000
    ).isel(time=slice(1, None, None))
    precip_var = dataset_input.attrs["precip_var"]
    metadata = dataset_input[precip_var].attrs

    pytest.importorskip("cv2")
    oflow_method = motion.get_method("LK")
    dataset_w_motion = oflow_method(dataset_input)

    nowcast_method = nowcasts.get_method("steps")

    dataset_forecast = nowcast_method(
        dataset_w_motion,
        timesteps=timesteps,
        precip_thr=metadata["threshold"],
        kmperpixel=2.0,
        timestep=metadata["accutime"],
        seed=42,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        ar_order=ar_order,
        mask_method=mask_method,
        probmatching_method=probmatching_method,
        domain=domain,
    )
    precip_forecast = dataset_forecast[precip_var].values

    assert precip_forecast.ndim == 4
    assert precip_forecast.shape[0] == n_ens_members
    assert precip_forecast.shape[1] == (
        timesteps if isinstance(timesteps, int) else len(timesteps)
    )

    crps = verification.probscores.CRPS(
        precip_forecast[:, -1], dataset_obs[precip_var].values[-1]
    )
    assert crps < max_crps, f"CRPS={crps:.2f}, required < {max_crps:.2f}"


def test_steps_callback(tmp_path):
    """Test STEPS callback functionality to export the output as a netcdf."""

    pytest.importorskip("netCDF4")
    pytest.importorskip("cv2")

    n_ens_members = 2
    n_timesteps = 3

    dataset_input = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        upscale=2000,
    )
    precip_var = dataset_input.attrs["precip_var"]
    precip_attrs = dataset_input[precip_var].attrs
    field_shape = dataset_input[precip_var].shape[1:]
    startdate = dataset_input.time.values[-1].astype("datetime64[us]").astype(datetime)
    timestep = precip_attrs["accutime"]

    metadata = {
        "projection": dataset_input.attrs["projection"],
        "x1": dataset_input.x.isel(x=0).values,
        "y1": dataset_input.y.isel(y=0).values,
        "x2": dataset_input.x.isel(x=-1).values,
        "y2": dataset_input.y.isel(y=-1).values,
        "unit": precip_attrs["units"],
        "yorigin": "upper" if dataset_input.y.attrs["stepsize"] < 0 else "lower",
        "cartesian_unit": dataset_input.x.attrs["units"],
    }

    oflow_method = motion.get_method("LK")
    dataset_w_motion = oflow_method(dataset_input)

    exporter = io.initialize_forecast_exporter_netcdf(
        outpath=tmp_path.as_posix(),
        outfnprefix="test_steps",
        startdate=startdate,
        timestep=timestep,
        n_timesteps=n_timesteps,
        shape=field_shape,
        n_ens_members=n_ens_members,
        metadata=metadata,
        incremental="timestep",
    )

    def callback(array):
        return io.export_forecast_dataset(array, exporter)

    dataset_output = nowcasts.get_method("steps")(
        dataset_w_motion,
        timesteps=n_timesteps,
        precip_thr=precip_attrs["threshold"],
        kmperpixel=2.0,
        timestep=timestep,
        seed=42,
        n_ens_members=n_ens_members,
        vel_pert_method=None,
        callback=callback,
        return_output=True,
    )
    io.close_forecast_files(exporter)
    precip_output = dataset_output[precip_var].values

    # assert that netcdf exists and its size is not zero
    tmp_file = os.path.join(tmp_path, "test_steps.nc")
    assert os.path.exists(tmp_file) and os.path.getsize(tmp_file) > 0

    # assert that the file can be read by the nowcast importer
    dataset_netcdf = io.import_netcdf_pysteps(tmp_file, dtype="float64")
    precip_netcdf = dataset_netcdf[precip_var].values

    # assert that the dimensionality of the array is as expected
    assert precip_netcdf.ndim == 4, "Wrong number of dimensions"
    assert precip_netcdf.shape[0] == n_ens_members, "Wrong ensemble size"
    assert precip_netcdf.shape[1] == n_timesteps, "Wrong number of lead times"
    assert precip_netcdf.shape[2:] == field_shape, "Wrong field shape"

    # assert that the saved output is the same as the original output
    assert np.allclose(
        precip_netcdf, precip_output, equal_nan=True
    ), "Wrong output values"

    # assert that leadtimes and timestamps are as expected
    td = timedelta(minutes=timestep)
    leadtimes = [(i + 1) * timestep for i in range(n_timesteps)]
    timestamps = [startdate + (i + 1) * td for i in range(n_timesteps)]
    timestamps_netcdf = [
        ts.astype("datetime64[us]").astype(datetime)
        for ts in dataset_netcdf.time.values
    ]
    leadtimes_netcdf = [
        (ts - startdate).total_seconds() / 60.0 for ts in timestamps_netcdf
    ]
    assert leadtimes_netcdf == leadtimes, "Wrong leadtimes"
    assert timestamps_netcdf == timestamps, "Wrong timestamps"


def run_and_assert_forecast(
    dataset, forecast_kwargs, expected_n_ens_members, n_timesteps, converter
):
    """Run a pysteps nowcast and assert the output has the expected shape."""
    precip_var = dataset.attrs["precip_var"]
    dataset_forecast = nowcasts.steps.forecast(dataset, **forecast_kwargs)
    precip_forecast = dataset_forecast[precip_var].values

    assert precip_forecast.ndim == 4, "Wrong amount of dimensions in forecast output"
    assert (
        precip_forecast.shape[0] == expected_n_ens_members
    ), "Wrong amount of output ensemble members in forecast output"
    assert (
        precip_forecast.shape[1] == n_timesteps
    ), "Wrong amount of output time steps in forecast output"

    # Transform the data back into mm/h
    dataset_forecast = converter(dataset_forecast)
    precip_forecast = dataset_forecast[precip_var].values

    assert (
        precip_forecast.ndim == 4
    ), "Wrong amount of dimensions in converted forecast output"
    assert (
        precip_forecast.shape[0] == expected_n_ens_members
    ), "Wrong amount of output ensemble members in converted forecast output"
    assert (
        precip_forecast.shape[1] == n_timesteps
    ), "Wrong amount of output time steps in converted forecast output"


@pytest.mark.parametrize(steps_arg_names, steps_arg_values)
def test_steps_nowcast(
    n_ens_members,
    n_cascade_levels,
    ar_order,
    mask_method,
    probmatching_method,
    domain,
    timesteps,
    max_crps,
):
    pytest.importorskip("cv2")

    ###
    # The input data
    ###
    # Initialise dummy NWP data
    if not isinstance(timesteps, int):
        n_timesteps = len(timesteps)
    else:
        n_timesteps = timesteps

    vel_pert_method = None

    # Define dummy nowcast input data
    radar_precip = np.zeros((3, 200, 200))

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
    metadata["transformation"] = "dB"
    metadata["accutime"] = 5.0
    metadata["transform"] = "dB"
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = 0.01
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6
    metadata["x1"] = 0.0
    metadata["x2"] = 200.0
    metadata["y1"] = 0.0
    metadata["y2"] = 200.0
    metadata["xpixelsize"] = 1.0
    metadata["ypixelsize"] = 1.0
    metadata["yorigin"] = "lower"
    metadata["cartesian_unit"] = "km"
    metadata["institution"] = "test"
    metadata["projection"] = (
        "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 "
        "+a=6378.137 +b=6356.752 +x_0=0 +y_0=0"
    )

    mask_kwargs = None

    ###
    # First threshold the data and convert it to dBR
    ###
    # threshold the data
    radar_precip[radar_precip < metadata["threshold"]] = 0.0

    dataset_input = convert_input_to_xarray_dataset(
        radar_precip, None, metadata, datetime(2021, 7, 4, 12, 0, 0), 300
    )

    # convert the data
    converter = utils.get_method("mm/h")
    dataset_input = converter(dataset_input)

    # transform the data
    transformer = utils.get_method(metadata["transformation"])
    dataset_input = transformer(dataset_input)

    # set NaN equal to zero
    precip_var = dataset_input.attrs["precip_var"]
    dataset_input[precip_var].values[~np.isfinite(dataset_input[precip_var].values)] = (
        metadata["zerovalue"]
    )

    assert (
        np.any(~np.isfinite(dataset_input[precip_var].values)) == False
    ), "There are still infinite values in the input radar data"

    ###
    # Determine the velocity fields
    ###
    oflow_method = motion.get_method("lucaskanade")
    dataset_w_motion = oflow_method(dataset_input)

    ###
    # Shared forecast kwargs
    ###
    forecast_kwargs = dict(
        timesteps=timesteps,
        timestep=5.0,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        precip_thr=metadata["threshold"],
        kmperpixel=1.0,
        extrap_method="semilagrangian",
        decomp_method="fft",
        bandpass_filter_method="gaussian",
        noise_method="nonparametric",
        noise_stddev_adj="auto",
        ar_order=ar_order,
        vel_pert_method=vel_pert_method,
        conditional=False,
        probmatching_method=probmatching_method,
        mask_method=mask_method,
        callback=None,
        return_output=True,
        seed=None,
        num_workers=1,
        fft_method="numpy",
        domain=domain,
        extrap_kwargs=None,
        filter_kwargs=None,
        noise_kwargs=None,
        vel_pert_kwargs=None,
        mask_kwargs=mask_kwargs,
        measure_time=False,
    )

    ###
    # The Nowcast
    ###
    # Test with full radar data
    run_and_assert_forecast(
        dataset_w_motion,
        forecast_kwargs,
        n_ens_members,
        n_timesteps,
        converter,
    )


@pytest.mark.parametrize("ar_order", [1, 2])
def test_steps_nowcast_partial_zero_radar(ar_order):
    """Test that a forecast succeeds when only the 2 latest radar frames have precipitation (initiating cell corner case that produces NaN autocorrelations for the earlier, all-zero cascade levels)."""
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
        x1=0.0,
        x2=200.0,
        y1=0.0,
        y2=200.0,
        xpixelsize=1.0,
        ypixelsize=1.0,
        yorigin="lower",
        cartesian_unit="km",
        institution="test",
        projection=(
            "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 "
            "+a=6378.137 +b=6356.752 +x_0=0 +y_0=0"
        ),
    )

    # Build radar data: only the 2 latest (most recent) frames have precipitation
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
    dataset_input = convert_input_to_xarray_dataset(
        radar_precip, None, metadata, datetime(2021, 7, 4, 12, 0, 0), 300
    )
    converter = utils.get_method("mm/h")
    dataset_input = converter(dataset_input)
    transformer = utils.get_method(metadata["transformation"])
    dataset_input = transformer(dataset_input)
    precip_var = dataset_input.attrs["precip_var"]
    dataset_input[precip_var].values[~np.isfinite(dataset_input[precip_var].values)] = (
        metadata["zerovalue"]
    )

    # Velocity fields
    oflow_method = motion.get_method("lucaskanade")
    dataset_w_motion = oflow_method(dataset_input)

    run_and_assert_forecast(
        dataset_w_motion,
        dict(
            timesteps=n_timesteps,
            timestep=5.0,
            n_ens_members=2,
            precip_thr=dataset_w_motion[precip_var].attrs["threshold"],
            kmperpixel=1.0,
            extrap_method="semilagrangian",
            decomp_method="fft",
            bandpass_filter_method="gaussian",
            noise_method="nonparametric",
            noise_stddev_adj="auto",
            ar_order=ar_order,
            vel_pert_method=None,
            conditional=False,
            probmatching_method=None,
            mask_method="incremental",
            callback=None,
            return_output=True,
            seed=42,
            num_workers=1,
            fft_method="numpy",
            domain="spatial",
            extrap_kwargs=None,
            filter_kwargs=None,
            noise_kwargs=None,
            vel_pert_kwargs=None,
            mask_kwargs=None,
            measure_time=False,
        ),
        expected_n_ens_members=2,
        n_timesteps=n_timesteps,
        converter=converter,
    )
