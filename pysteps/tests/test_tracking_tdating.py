# -*- coding: utf-8 -*-

import datetime as dt

import numpy as np
import pytest

from pysteps.tracking.tdating import dating
from pysteps.utils import to_reflectivity
from pysteps.tests.helpers import get_precipitation_fields

arg_names = "source"

arg_values = [
    ("mch"),
]


@pytest.mark.parametrize(arg_names, arg_values)
def test_tracking_tdating_dating(source):

    pytest.importorskip("skimage")
    pytest.importorskip("pandas")

    input, metadata = get_precipitation_fields(0, 2, True, True, 4000, source)
    input, __ = to_reflectivity(input, metadata)

    timelist = metadata["timestamps"]

    output = dating(input, timelist)

    # Check output format
    assert isinstance(output, tuple)
    assert len(output) == 3
    assert isinstance(output[0], list)
    assert isinstance(output[1], list)
    assert isinstance(output[2], list)
    assert isinstance(output[2][0], np.ndarray)
    assert output[2][0].ndim == 2
    assert output[2][0].shape == input.shape[1:]
