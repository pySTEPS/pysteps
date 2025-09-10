# -*- coding: utf-8 -*-

import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.verification import spatialscores

precip_dataset = get_precipitation_fields(num_prev_files=1, return_raw=True)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]

# XR: the scorring code has not been made xarray compatible, so we need to convert to numpy arrays. Once changed we can properly test these scores with xarray DataArrays
# BUG: the tests for BMSE below reverse the arrays with [::-1], this should be fixed in the scoring code
test_data = [
    (
        precip_dataarray.isel(time=0).values,
        precip_dataarray.isel(time=1).values,
        "FSS",
        [1],
        [10],
        None,
        0.85161531,
    ),
    (
        precip_dataarray.isel(time=0).values[::-1],
        precip_dataarray.isel(time=1).values[::-1],
        "BMSE",
        [1],
        None,
        "Haar",
        0.99989651,
    ),
]


@pytest.mark.parametrize("X_f, X_o, name, thrs, scales, wavelet, expected", test_data)
def test_intensity_scale(X_f, X_o, name, thrs, scales, wavelet, expected):
    """Test the intensity_scale."""
    if name == "BMSE":
        pytest.importorskip("pywt")

    assert_array_almost_equal(
        spatialscores.intensity_scale(X_f, X_o, name, thrs, scales, wavelet)[0][0],
        expected,
    )


precip_dataset = get_precipitation_fields(num_next_files=3, return_raw=True)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]

test_data = [
    (
        precip_dataarray.isel(time=slice(0, 2)).values,
        precip_dataarray.isel(
            time=slice(
                2,
            )
        ).values,
        "FSS",
        [1],
        [10],
        None,
    ),
    (
        precip_dataarray.isel(time=slice(0, 2)).values,
        precip_dataarray.isel(
            time=slice(
                2,
            )
        ).values,
        "BMSE",
        [1],
        None,
        "Haar",
    ),
]


@pytest.mark.parametrize("R1, R2, name, thrs, scales, wavelet", test_data)
def test_intensity_scale_methods(R1, R2, name, thrs, scales, wavelet):
    """
    Test the intensity_scale merge."""
    if name == "BMSE":
        pytest.importorskip("pywt")

    # expected reult
    int = spatialscores.intensity_scale_init(name, thrs, scales, wavelet)
    spatialscores.intensity_scale_accum(int, R1[0], R1[1])
    spatialscores.intensity_scale_accum(int, R2[0], R2[1])
    expected = spatialscores.intensity_scale_compute(int)[0][0]

    # init
    int_1 = spatialscores.intensity_scale_init(name, thrs, scales, wavelet)
    int_2 = spatialscores.intensity_scale_init(name, thrs, scales, wavelet)

    # accum
    spatialscores.intensity_scale_accum(int_1, R1[0], R1[1])
    spatialscores.intensity_scale_accum(int_2, R2[0], R2[1])

    # merge
    int = spatialscores.intensity_scale_merge(int_1, int_2)

    # compute
    score = spatialscores.intensity_scale_compute(int)[0][0]

    assert_array_almost_equal(score, expected)
