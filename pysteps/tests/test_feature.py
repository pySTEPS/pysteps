import pytest
import numpy as np
from pysteps import feature
from pysteps.tests.helpers import get_precipitation_fields

methods = ["blob", "shitomasi"]


@pytest.mark.parametrize("method", methods)
def test_feature(method):
    input, metadata = get_precipitation_fields(0, 0, True, True, None, "mch")

    detector = feature.get_method(method)

    output = detector(input.squeeze())

    assert isinstance(output, np.ndarray)
    assert output.ndim == 2
    assert output.shape[0] > 0
    assert output.shape[1] == 2


# TODO: remove this
if __name__ == "__main__":
    test_feature("blob")
    test_feature("shitomasi")
