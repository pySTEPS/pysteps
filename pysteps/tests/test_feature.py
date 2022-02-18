import pytest
import numpy as np
from pysteps import feature
from pysteps.tests.helpers import get_precipitation_fields

arg_names = ["method", "max_num_features"]
arg_values = [("blob", None), ("blob", 5), ("shitomasi", None), ("shitomasi", 5)]


@pytest.mark.parametrize(arg_names, arg_values)
def test_feature(method, max_num_features):
    if method == "blob":
        pytest.importorskip("skimage")
    if method == "shitomasi":
        pytest.importorskip("cv2")

    input_field, _ = get_precipitation_fields(0, 0, True, True, None, "mch")

    detector = feature.get_method(method)

    kwargs = {"max_num_features": max_num_features}
    output = detector(input_field.squeeze(), **kwargs)

    assert isinstance(output, np.ndarray)
    assert output.ndim == 2
    assert output.shape[0] > 0
    if max_num_features is not None:
        assert output.shape[0] <= max_num_features
    assert output.shape[1] == 2
