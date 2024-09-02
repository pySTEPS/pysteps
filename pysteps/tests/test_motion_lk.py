# coding: utf-8

"""
"""

import numpy as np
import pytest

from pysteps import motion, verification
from pysteps.tests.helpers import get_precipitation_fields

lk_arg_names = (
    "lk_kwargs",
    "fd_method",
    "dense",
    "nr_std_outlier",
    "k_outlier",
    "size_opening",
    "decl_scale",
    "verbose",
)

lk_arg_values = [
    ({}, "shitomasi", True, 3, 30, 3, 20, False),  # defaults
    ({}, "shitomasi", False, 3, 30, 3, 20, True),  # sparse ouput, verbose
    ({}, "shitomasi", False, 0, 30, 3, 20, False),  # sparse ouput, all outliers
    (
        {},
        "shitomasi",
        True,
        3,
        None,
        0,
        0,
        False,
    ),  # global outlier detection, no filtering, no declutering
    ({}, "shitomasi", True, 0, 30, 3, 20, False),  # all outliers
    ({}, "blob", True, 3, 30, 3, 20, False),  # blob detection
    ({}, "tstorm", True, 3, 30, 3, 20, False),  # tstorm detection
]


@pytest.mark.parametrize(lk_arg_names, lk_arg_values)
def test_lk(
    lk_kwargs,
    fd_method,
    dense,
    nr_std_outlier,
    k_outlier,
    size_opening,
    decl_scale,
    verbose,
):
    """Tests Lucas-Kanade optical flow."""

    pytest.importorskip("cv2")
    if fd_method == "blob":
        pytest.importorskip("skimage")
    if fd_method == "tstorm":
        pytest.importorskip("skimage")
        pytest.importorskip("pandas")

    # inputs
    dataset = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=False,
        metadata=True,
        upscale=2000,
    )
    precip_var = dataset.attrs["precip_var"]

    # Retrieve motion field
    oflow_method = motion.get_method("LK")
    output_dataset = oflow_method(
        dataset,
        lk_kwargs=lk_kwargs,
        fd_method=fd_method,
        dense=dense,
        nr_std_outlier=nr_std_outlier,
        k_outlier=k_outlier,
        size_opening=size_opening,
        decl_scale=decl_scale,
        verbose=verbose,
    )

    # Check format of ouput
    if dense:
        output = np.stack(
            [output_dataset["velocity_x"].values, output_dataset["velocity_y"].values]
        )
        assert isinstance(output, np.ndarray)
        assert output.ndim == 3
        assert output.shape[0] == 2
        assert output.shape[1:] == dataset[precip_var].values[0].shape
        if nr_std_outlier == 0:
            assert output.sum() == 0
    else:
        output = output_dataset
        assert isinstance(output, tuple)
        assert len(output) == 2
        assert isinstance(output[0], np.ndarray)
        assert isinstance(output[1], np.ndarray)
        assert output[0].ndim == 2
        assert output[1].ndim == 2
        assert output[0].shape[1] == 2
        assert output[1].shape[1] == 2
        assert output[0].shape[0] == output[1].shape[0]
        if nr_std_outlier == 0:
            assert output[0].shape[0] == 0
            assert output[1].shape[0] == 0
