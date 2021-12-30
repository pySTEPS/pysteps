# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.verification import sal
from pysteps.utils import to_reflectivity


def test_sal():
    """Test the SAL verification method."""
    pytest.importorskip("pandas")
    pytest.importorskip("skimage")
    precip, metadata = get_precipitation_fields(num_prev_files=0, metadata=True)
    precip, metadata = to_reflectivity(precip, metadata)
    # same image
    result = sal.sal(precip, precip)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert np.allclose(result, [0, 0, 0])
    # with displacement
    precip_translated = np.roll(precip, 10, axis=0)
    result = sal.sal(precip, precip_translated)
    assert np.allclose(result[0], 0)
    assert np.allclose(result[1], 0)
    assert not np.allclose(result[2], 0)
