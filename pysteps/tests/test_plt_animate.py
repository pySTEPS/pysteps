# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.visualization.animations import animate


def test_animate(tmp_path):

    # test data
    precip, metadata = get_precipitation_fields(
        num_prev_files=2,
        num_next_files=0,
        return_raw=True,
        metadata=True,
        upscale=2000,
    )

    # obs only
    animate(precip)
    animate(precip, title="title")
    animate(precip, timestamps_obs=metadata["timestamps"])
    animate(precip, geodata=metadata, map_kwargs={"plot_map": None})
    with pytest.raises(ValueError):
        animate(precip, timestamps_obs=metadata["timestamps"][:2])
    animate(precip, motion_field=np.ones((2, *precip.shape[1:])))
    with pytest.raises(ValueError):
        animate(precip, motion_plot="test")

    # with forecast
    animate(precip, precip)
    animate(precip, precip, title="title")
    animate(precip, precip, timestamps_obs=metadata["timestamps"])
    animate(precip, precip, timestamps_obs=metadata["timestamps"], timestep_min=5)
    with pytest.raises(ValueError):
        animate(precip, precip, ptype="prob")
    animate(precip, precip, ptype="prob", prob_thr=1)
    animate(precip, precip, ptype="mean")
    animate(precip, np.stack((precip, precip)), ptype="ensemble")

    # save frames
    animate(
        precip,
        np.stack((precip, precip)),
        display_animation=False,
        savefig=True,
        path_outputs=tmp_path,
        fig_dpi=10,
    )
    assert len(os.listdir(tmp_path)) == 9
