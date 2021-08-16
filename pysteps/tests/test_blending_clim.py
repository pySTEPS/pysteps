# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pysteps.blending.clim import save_weights, calc_clim_weights
import random
from datetime import datetime, timedelta
from os.path import join, exists

random.seed(12356)
n_cascade_levels = 7
model_names = ["alaro13", "arome13"]
default_start_weights = [0.8, 0.5]
""" Helper functions """


def generate_random_weights(n_cascade_levels, n_models=1):
    """
    Generate random weights which decay exponentially with scale.
    """
    start_weights = np.array([random.uniform(0.5, 0.99) for i in range(n_models)])
    powers = np.arange(1, n_cascade_levels + 1)
    return pow(start_weights[:, np.newaxis], powers)


def generate_fixed_weights(n_cascade_levels, n_models=1):
    """
    Generate weights starting at default_start_weights which decay exponentially with scale.
    """
    start_weights = np.resize(default_start_weights, n_models)
    powers = np.arange(1, n_cascade_levels + 1)
    return pow(start_weights[:, np.newaxis], powers)


""" Test arguments """

clim_arg_names = ("startdatestr", "enddatestr", "n_models", "expected_weights_today")

test_enddates = ["20210701235500", "20210702010000", "20200930235500"]

clim_arg_values = [
    (
        "20210701230000",
        "20210701235500",
        1,
        {
            "mean_weights": generate_fixed_weights(n_cascade_levels),
            "n": 12,
            "last_validtime": datetime.strptime(test_enddates[0], "%Y%m%d%H%M%S"),
        },
    ),
    (
        "20210701230000",
        "20210702000000",
        1,
        {
            "mean_weights": generate_fixed_weights(n_cascade_levels),
            "n": 1,
            "last_validtime": datetime.strptime(test_enddates[1], "%Y%m%d%H%M%S"),
        },
    ),
    (
        "20200801000000",
        "20200930235500",
        1,
        {
            "mean_weights": generate_fixed_weights(n_cascade_levels),
            "n": 288,
            "last_validtime": datetime.strptime(test_enddates[2], "%Y%m%d%H%M%S"),
        },
    ),
    (
        "20210701230000",
        "20210701235500",
        2,
        {
            "mean_weights": generate_fixed_weights(n_cascade_levels, 2),
            "n": 12,
            "last_validtime": datetime.strptime(test_enddates[0], "%Y%m%d%H%M%S"),
        },
    ),
    (
        "20210701230000",
        "20210702000000",
        2,
        {
            "mean_weights": generate_fixed_weights(n_cascade_levels, 2),
            "n": 1,
            "last_validtime": datetime.strptime(test_enddates[1], "%Y%m%d%H%M%S"),
        },
    ),
    (
        "20200801000000",
        "20200930235500",
        2,
        {
            "mean_weights": generate_fixed_weights(n_cascade_levels, 2),
            "n": 288,
            "last_validtime": datetime.strptime(test_enddates[2], "%Y%m%d%H%M%S"),
        },
    ),
]


@pytest.mark.parametrize(clim_arg_names, clim_arg_values)
def test_save_weights(
    startdatestr, enddatestr, n_models, expected_weights_today, tmpdir
):
    """Test if the weights are saved correctly and the daily average is computed"""

    # get validtime
    currentdate = datetime.strptime(startdatestr, "%Y%m%d%H%M%S")
    enddate = datetime.strptime(enddatestr, "%Y%m%d%H%M%S")
    timestep = timedelta(minutes=5)

    outdir_path = tmpdir

    while currentdate <= enddate:
        current_weights = generate_fixed_weights(n_cascade_levels, n_models)
        save_weights(current_weights, currentdate, outdir_path, window_length=2)
        currentdate += timestep

    weights_today_file = join(outdir_path, "NWP_weights_today.npy")
    assert exists(weights_today_file)

    weights_today = load(weights_today_file)

    # Check type
    assert type(weights_today) == type({})
    assert "mean_weights" in weights_today
    assert "n" in weights_today
    assert "last_validtime" in weights_today
    assertEqual(weights_today, expected_weights_today)


if __name__ == "__main__":
    save_weights(
        generate_fixed_weights(n_cascade_levels, 1),
        datetime.strptime("20200801000000", "%Y%m%d%H%M%S"),
        "/tmp/",
    )
