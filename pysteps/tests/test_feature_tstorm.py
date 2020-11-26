# -*- coding: utf-8 -*-

import datetime as dt

import numpy as np
import pytest

from pysteps.feature.tstorm import detection
from pysteps.utils import to_reflectivity
from pysteps.tests.helpers import get_precipitation_fields

arg_names = ("source", "output_feat")

arg_values = [
    ("mch", False),
    ("mch", True),
]


@pytest.mark.parametrize(arg_names, arg_values)
def test_feature_tstorm_detection(source, output_feat):

    pytest.importorskip("skimage")
    pytest.importorskip("pandas")

    input, metadata = get_precipitation_fields(0, 0, True, True, None, source)
    input = input.squeeze()
    input, __ = to_reflectivity(input, metadata)

    output = detection(input, output_feat=output_feat)

    if output_feat:
        assert isinstance(output, np.ndarray)
        assert output.ndim == 2
        assert output.shape[1] == 2
    else:
        assert isinstance(output, tuple)
        assert len(output) == 2
        assert isinstance(output[1], np.ndarray)
        assert output[1].ndim == 2
        assert output[1].shape == input.shape
