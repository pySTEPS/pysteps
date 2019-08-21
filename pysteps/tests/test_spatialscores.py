# -*- coding: utf-8 -*-

import pytest
from numpy.testing import assert_array_almost_equal

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.verification import spatialscores

try:
    import pywt

    PYWT_IMPORTED = True
except ImportError:
    PYWT_IMPORTED = False

R = get_precipitation_fields(num_prev_files=1, return_raw=True)
test_data = [(R[0], R[1], "FSS", [1], [10], None, 0.85161531)]
if PYWT_IMPORTED:
    test_data.append((R[0], R[1], "BMSE", [1], None, "Haar", 0.99989651))


@pytest.mark.parametrize("X_f, X_o, name, thrs, scales, wavelet, expected",
                         test_data)
def test_intensity_scale(X_f, X_o, name, thrs, scales, wavelet, expected):
    """Test the intensity_scale."""
    assert_array_almost_equal(
        spatialscores.intensity_scale(X_f, X_o, name,
                                      thrs, scales, wavelet)[0][0],
        expected,
    )
