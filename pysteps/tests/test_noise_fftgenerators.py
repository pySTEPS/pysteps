import numpy as np

from pysteps.noise import fftgenerators
from pysteps.tests.helpers import get_precipitation_fields


precip_dataset = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=False,
    upscale=2000,
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]


# XR: all tests assume a 2D field, so we select the first timestep, these tests need to be changed when fftgenerators support xarray DataArrays
def test_noise_param_2d_fft_filter():
    fft_filter = fftgenerators.initialize_param_2d_fft_filter(
        precip_dataarray.isel(time=0).values
    )

    assert isinstance(fft_filter, dict)
    assert all([key in fft_filter for key in ["field", "input_shape", "model", "pars"]])

    out = fftgenerators.generate_noise_2d_fft_filter(fft_filter)

    assert isinstance(out, np.ndarray)
    assert out.shape == precip_dataarray.isel(time=0).shape


def test_noise_nonparam_2d_fft_filter():
    fft_filter = fftgenerators.initialize_nonparam_2d_fft_filter(
        precip_dataarray.isel(time=0).values
    )

    assert isinstance(fft_filter, dict)
    assert all([key in fft_filter for key in ["field", "input_shape"]])

    out = fftgenerators.generate_noise_2d_fft_filter(fft_filter)

    assert isinstance(out, np.ndarray)
    assert out.shape == precip_dataarray.isel(time=0).shape


def test_noise_nonparam_2d_ssft_filter():
    fft_filter = fftgenerators.initialize_nonparam_2d_ssft_filter(
        precip_dataarray.isel(time=0).values
    )

    assert isinstance(fft_filter, dict)
    assert all([key in fft_filter for key in ["field", "input_shape"]])

    out = fftgenerators.generate_noise_2d_ssft_filter(fft_filter)

    assert isinstance(out, np.ndarray)
    assert out.shape == precip_dataarray.isel(time=0).shape


def test_noise_nonparam_2d_nested_filter():
    fft_filter = fftgenerators.initialize_nonparam_2d_nested_filter(
        precip_dataarray.isel(time=0).values
    )

    assert isinstance(fft_filter, dict)
    assert all([key in fft_filter for key in ["field", "input_shape"]])

    out = fftgenerators.generate_noise_2d_ssft_filter(fft_filter)

    assert isinstance(out, np.ndarray)
    assert out.shape == precip_dataarray.isel(time=0).shape
