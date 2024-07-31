from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import xarray as xr


def xarray_motion(motion_method):
    @wraps(motion_method)
    def _motion_method_wrapper(dataset: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        precip_var = dataset.attrs["precip_var"]
        input_images = dataset[precip_var].values
        output = motion_method(input_images, *args, **kwargs)
        dataset["velocity_x"] = (["y", "x"], output[0])
        dataset["velocity_y"] = (["y", "x"], output[1])

        return dataset

    return _motion_method_wrapper


def xarray_nowcast(nowcast_method):
    @wraps(nowcast_method)
    def _nowcast_method_wrapper(
        dataset: xr.Dataset, timesteps: int | list[int], *args, **kwargs
    ) -> xr.Dataset:
        precip_var = dataset.attrs["precip_var"]
        precip = dataset[precip_var].values
        velocity = np.stack([dataset["velocity_x"], dataset["velocity_y"]])

        output = nowcast_method(precip, velocity, timesteps, *args, **kwargs)
        metadata = dataset[precip_var].attrs
        last_timestamp = (
            dataset.time[-1].values.astype("datetime64[us]").astype(datetime)
        )
        time_metadata = dataset["time"].attrs
        timestep_seconds = (
            dataset["time"][1].values.astype("datetime64[us]").astype(datetime)
            - dataset["time"][0].values.astype("datetime64[us]").astype(datetime)
        ).total_seconds()
        dataset = dataset.drop_vars([precip_var]).drop_dims(["time"])
        if isinstance(timesteps, int):
            timesteps = list(range(1, timesteps + 1))
        next_timestamps = [
            last_timestamp + timedelta(seconds=timestep_seconds * i) for i in timesteps
        ]
        dataset = dataset.assign_coords(
            {"time": (["time"], next_timestamps, time_metadata)}
        )
        if output.ndim == 4:
            dataset = dataset.assign_coords(
                {
                    "ens_number": (
                        ["ens_number"],
                        list(range(1, output.shape[0] + 1)),
                        {
                            "long_name": "ensemble member",
                            "standard_name": "realization",
                            "units": "",
                        },
                    )
                }
            )
            dataset[precip_var] = (["ens_number", "time", "y", "x"], output, metadata)
        else:
            dataset[precip_var] = (["time", "y", "x"], output, metadata)

        return dataset

    return _nowcast_method_wrapper
