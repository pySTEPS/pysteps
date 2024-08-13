from functools import wraps

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
