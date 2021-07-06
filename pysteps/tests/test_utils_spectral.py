import numpy as np
import pytest
from pysteps.utils import spectral

_rapsd_input_fields = [
    np.random.uniform(size=(255, 255)),
    np.random.uniform(size=(256, 256)),
    np.random.uniform(size=(255, 256)),
    np.random.uniform(size=(256, 255)),
]


@pytest.mark.parametrize("field", _rapsd_input_fields)
def test_rapsd(field):
    rapsd, freq = spectral.rapsd(field, return_freq=True)

    m, n = field.shape
    l = max(m, n)

    if l % 2 == 0:
        assert len(rapsd) == int(l / 2)
    else:
        assert len(rapsd) == int(l / 2 + 1)
    assert len(rapsd) == len(freq)
    assert np.all(freq >= 0.0)
