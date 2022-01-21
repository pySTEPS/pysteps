# -*- coding: utf-8 -*-


from datetime import datetime, timedelta
from os.path import join, exists
import pickle
import random

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from pysteps.blending.clim import save_skill, calc_clim_skill


random.seed(12356)
n_cascade_levels = 7
model_names = ["alaro13", "arome13"]
default_start_skill = [0.8, 0.5]

# Helper functions
def generate_fixed_skill(n_cascade_levels, n_models=1):
    """
    Generate skill starting at default_start_skill which decay exponentially with scale.
    """
    start_skill = np.resize(default_start_skill, n_models)
    powers = np.arange(1, n_cascade_levels + 1)
    return pow(start_skill[:, np.newaxis], powers)


# Test arguments
clim_arg_names = ("startdatestr", "enddatestr", "n_models", "expected_skill_today")

test_enddates = ["20210701235500", "20210702000000", "20200930235500"]

clim_arg_values = [
    (
        "20210701230000",
        "20210701235500",
        1,
        {
            "mean_skill": generate_fixed_skill(n_cascade_levels),
            "n": 12,
            "last_validtime": datetime.strptime(test_enddates[0], "%Y%m%d%H%M%S"),
        },
    ),
    (
        "20210701235500",
        "20210702000000",
        1,
        {
            "mean_skill": generate_fixed_skill(n_cascade_levels),
            "n": 1,
            "last_validtime": datetime.strptime(test_enddates[1], "%Y%m%d%H%M%S"),
        },
    ),
    (
        "20200801000000",
        "20200930235500",
        1,
        {
            "mean_skill": generate_fixed_skill(n_cascade_levels),
            "n": 288,
            "last_validtime": datetime.strptime(test_enddates[2], "%Y%m%d%H%M%S"),
        },
    ),
    (
        "20210701230000",
        "20210701235500",
        2,
        {
            "mean_skill": generate_fixed_skill(n_cascade_levels, 2),
            "n": 12,
            "last_validtime": datetime.strptime(test_enddates[0], "%Y%m%d%H%M%S"),
        },
    ),
    (
        "20210701230000",
        "20210702000000",
        2,
        {
            "mean_skill": generate_fixed_skill(n_cascade_levels, 2),
            "n": 1,
            "last_validtime": datetime.strptime(test_enddates[1], "%Y%m%d%H%M%S"),
        },
    ),
    (
        "20200801000000",
        "20200930235500",
        2,
        {
            "mean_skill": generate_fixed_skill(n_cascade_levels, 2),
            "n": 288,
            "last_validtime": datetime.strptime(test_enddates[2], "%Y%m%d%H%M%S"),
        },
    ),
]


@pytest.mark.parametrize(clim_arg_names, clim_arg_values)
def test_save_skill(startdatestr, enddatestr, n_models, expected_skill_today, tmpdir):
    """Test if the skill are saved correctly and the daily average is computed"""

    # get validtime
    currentdate = datetime.strptime(startdatestr, "%Y%m%d%H%M%S")
    enddate = datetime.strptime(enddatestr, "%Y%m%d%H%M%S")
    timestep = timedelta(minutes=5)

    outdir_path = tmpdir

    while currentdate <= enddate:
        current_skill = generate_fixed_skill(n_cascade_levels, n_models)
        print("Saving skill: ", current_skill, currentdate, outdir_path)
        save_skill(
            current_skill, currentdate, outdir_path, n_models=n_models, window_length=2
        )
        currentdate += timestep

    skill_today_file = join(outdir_path, "NWP_skill_today.pkl")
    assert exists(skill_today_file)
    with open(skill_today_file, "rb") as f:
        skill_today = pickle.load(f)

    # Check type
    assert isinstance(skill_today, dict)
    assert "mean_skill" in skill_today
    assert "n" in skill_today
    assert "last_validtime" in skill_today
    assert_array_equal(skill_today["mean_skill"], expected_skill_today["mean_skill"])
    assert skill_today["n"] == expected_skill_today["n"]
    assert skill_today["last_validtime"] == expected_skill_today["last_validtime"]

    # Finally, check if the clim skill calculation returns an array of values
    clim_skill = calc_clim_skill(
        outdir_path=tmpdir,
        n_cascade_levels=n_cascade_levels,
        n_models=n_models,
        window_length=2,
    )

    assert clim_skill.shape[0] == n_models
    assert clim_skill.shape[1] == n_cascade_levels


if __name__ == "__main__":
    save_skill(
        generate_fixed_skill(n_cascade_levels, 1),
        datetime.strptime("20200801000000", "%Y%m%d%H%M%S"),
        "./tmp/",
    )
