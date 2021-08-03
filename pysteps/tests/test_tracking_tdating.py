# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pysteps.tracking.tdating import dating
from pysteps.utils import to_reflectivity
from pysteps.tests.helpers import get_precipitation_fields

arg_names = ("source", "dry_input")

arg_values = [
    ("mch", False),
    ("mch", False),
    ("mch", True),
]


@pytest.mark.parametrize(arg_names, arg_values)
def test_tracking_tdating_dating(source, dry_input):
    pytest.importorskip("skimage")
    pandas = pytest.importorskip("pandas")

    if not dry_input:
        input, metadata = get_precipitation_fields(0, 2, True, True, 4000, source)
        input, __ = to_reflectivity(input, metadata)
    else:
        input = np.zeros((3, 50, 50))
        metadata = {"timestamps": ["00", "01", "02"]}

    timelist = metadata["timestamps"]

    output = dating(input, timelist, mintrack=1)

    # Check output format
    assert isinstance(output, tuple)
    assert len(output) == 3
    assert isinstance(output[0], list)
    assert isinstance(output[1], list)
    assert isinstance(output[2], list)
    assert len(output[1]) == input.shape[0]
    assert len(output[2]) == input.shape[0]
    assert isinstance(output[1][0], pandas.DataFrame)
    assert isinstance(output[2][0], np.ndarray)
    assert output[1][0].shape[1] == 9
    assert output[2][0].shape == input.shape[1:]
    if not dry_input:
        assert len(output[0]) > 0
        assert isinstance(output[0][0], pandas.DataFrame)
        assert output[0][0].shape[1] == 9
    else:
        assert len(output[0]) == 0
        assert output[1][0].shape[0] == 0
        assert output[2][0].sum() == 0
