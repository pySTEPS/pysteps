# -*- coding: utf-8 -*-

import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.verification import spatialscores

R = get_precipitation_fields(num_prev_files=1, return_raw=True)
test_data = [
    (R[0], R[1], "FSS", [1], [10], None, 0.85161531),
    (R[0], R[1], "BMSE", [1], None, "Haar", 0.99989651),
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


R = get_precipitation_fields(num_prev_files=3, return_raw=True)
test_data = [
    (R[:2], R[2:], "FSS", [1], [10], None),
    (R[:2], R[2:], "BMSE", [1], None, "Haar"),
]


@pytest.mark.parametrize("R1, R2, name, thrs, scales, wavelet", test_data)
def test_intensity_scale_methods(R1, R2, name, thrs, scales, wavelet):
    """Test the intensity_scale merge."""
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
