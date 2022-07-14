# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.verification.salscores import sal
from pysteps.utils import to_rainrate, to_reflectivity

test_data = [
    (to_rainrate, 1 / 15),
    (to_reflectivity, None),
]


@pytest.mark.parametrize("converter, thr_factor", test_data)
class TestSAL:
    pytest.importorskip("pandas")
    pytest.importorskip("skimage")

    def test_sal_zeros(self, converter, thr_factor):
        """Test the SAL verification method."""
        precip, metadata = get_precipitation_fields(
            num_prev_files=0, log_transform=False, metadata=True
        )
        precip, metadata = converter(precip.filled(np.nan), metadata)
        result = sal(precip * 0, precip * 0, thr_factor)
        assert np.isnan(result).all()
        result = sal(precip * 0, precip, thr_factor)
        assert result[:2] == (-2, -2)
        assert np.isnan(result[2])
        result = sal(precip, precip * 0, thr_factor)
        assert result[:2] == (2, 2)
        assert np.isnan(result[2])

    def test_sal_same_image(self, converter, thr_factor):
        """Test the SAL verification method."""
        precip, metadata = get_precipitation_fields(
            num_prev_files=0, log_transform=False, metadata=True
        )
        precip, metadata = converter(precip.filled(np.nan), metadata)
        result = sal(precip, precip, thr_factor)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert np.allclose(result, [0, 0, 0])

    def test_sal_translation(self, converter, thr_factor):
        precip, metadata = get_precipitation_fields(
            num_prev_files=0, log_transform=False, metadata=True
        )
        precip, metadata = converter(precip.filled(np.nan), metadata)
        precip_translated = np.roll(precip, 10, axis=0)
        result = sal(precip, precip_translated, thr_factor)
        assert np.allclose(result[0], 0)
        assert np.allclose(result[1], 0)
        assert not np.allclose(result[2], 0)
