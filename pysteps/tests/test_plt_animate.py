# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest
from unittest.mock import patch

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.visualization.animations import animate


PRECIP, METADATA = get_precipitation_fields(
    num_prev_files=2,
    num_next_files=0,
    return_raw=True,
    metadata=True,
    upscale=2000,
)

VALID_ARGS = (
    ([PRECIP], {}),
    ([PRECIP], {"title": "title"}),
    ([PRECIP], {"timestamps_obs": METADATA["timestamps"]}),
    ([PRECIP], {"geodata": METADATA, "map_kwargs": {"plot_map": None}}),
    ([PRECIP], {"motion_field": np.ones((2, *PRECIP.shape[1:]))}),
    (
        [PRECIP],
        {"precip_kwargs": {"units": "mm/h", "colorbar": True, "colorscale": "pysteps"}},
    ),
    ([PRECIP, PRECIP], {}),
    ([PRECIP, PRECIP], {"title": "title"}),
    ([PRECIP, PRECIP], {"timestamps_obs": METADATA["timestamps"]}),
    ([PRECIP, PRECIP], {"timestamps_obs": METADATA["timestamps"], "timestep_min": 5}),
    ([PRECIP, PRECIP], {"ptype": "prob", "prob_thr": 1}),
    ([PRECIP, PRECIP], {"ptype": "mean"}),
    ([PRECIP, np.stack((PRECIP, PRECIP))], {"ptype": "ensemble"}),
)


@pytest.mark.parametrize(["anim_args", "anim_kwargs"], VALID_ARGS)
def test_animate(anim_args, anim_kwargs):
    with patch("matplotlib.pyplot.show"):
        animate(*anim_args, **anim_kwargs)


VALUEERROR_ARGS = (
    ([PRECIP], {"timestamps_obs": METADATA["timestamps"][:2]}),
    ([PRECIP], {"motion_plot": "test"}),
    ([PRECIP, PRECIP], {"ptype": "prob"}),
)


@pytest.mark.parametrize(["anim_args", "anim_kwargs"], VALUEERROR_ARGS)
def test_animate_valueerrors(anim_args, anim_kwargs):
    with pytest.raises(ValueError):
        animate(*anim_args, **anim_kwargs)


TYPEERROR_ARGS = (
    ([PRECIP], {"timestamps": METADATA["timestamps"]}),
    ([PRECIP], {"plotanimation": True}),
    ([PRECIP], {"units": "mm/h"}),
    ([PRECIP], {"colorbar": True}),
    ([PRECIP], {"colorscale": "pysteps"}),
    ([PRECIP, PRECIP], {"type": "ensemble"}),
)


@pytest.mark.parametrize(["anim_args", "anim_kwargs"], TYPEERROR_ARGS)
def test_animate_typeerrors(anim_args, anim_kwargs):
    with pytest.raises(TypeError):
        animate(*anim_args, **anim_kwargs)


def test_animate_save(tmp_path):
    animate(
        PRECIP,
        np.stack((PRECIP, PRECIP)),
        display_animation=False,
        savefig=True,
        path_outputs=tmp_path,
        fig_dpi=10,
    )
    assert len(os.listdir(tmp_path)) == 9
