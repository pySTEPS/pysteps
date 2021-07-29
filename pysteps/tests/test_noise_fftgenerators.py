import numpy as np

from pysteps.noise import fftgenerators
from pysteps.tests.helpers import get_precipitation_fields


PRECIP = get_precipitation_fields(
    num_prev_files=0,
    num_next_files=0,
    return_raw=False,
    metadata=False,
    upscale=2000,
)
PRECIP = PRECIP.filled()


def test_noise_param_2d_fft_filter():

    fft_filter = fftgenerators.initialize_param_2d_fft_filter(PRECIP)

    assert isinstance(fft_filter, dict)
    assert all([key in fft_filter for key in ["field", "input_shape", "model", "pars"]])

    out = fftgenerators.generate_noise_2d_fft_filter(fft_filter)

    assert isinstance(out, np.ndarray)
    assert out.shape == PRECIP.shape


def test_noise_nonparam_2d_fft_filter():

    fft_filter = fftgenerators.initialize_nonparam_2d_fft_filter(PRECIP)

    assert isinstance(fft_filter, dict)
    assert all([key in fft_filter for key in ["field", "input_shape"]])

    out = fftgenerators.generate_noise_2d_fft_filter(fft_filter)

    assert isinstance(out, np.ndarray)
    assert out.shape == PRECIP.shape


def test_noise_nonparam_2d_ssft_filter():

    fft_filter = fftgenerators.initialize_nonparam_2d_ssft_filter(PRECIP)

    assert isinstance(fft_filter, dict)
    assert all([key in fft_filter for key in ["field", "input_shape"]])

    out = fftgenerators.generate_noise_2d_ssft_filter(fft_filter)

    assert isinstance(out, np.ndarray)
    assert out.shape == PRECIP.shape
