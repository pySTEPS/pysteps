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

arg_names_multistep = ("source", "len_timesteps")
arg_values_multistep = [
    ("mch", 6),
]


@pytest.mark.parametrize(arg_names_multistep, arg_values_multistep)
def test_tracking_tdating_dating_multistep(source, len_timesteps):
    pytest.importorskip("skimage")

    input_fields, metadata = get_precipitation_fields(
        0, len_timesteps, True, True, 4000, source
    )
    input_fields, __ = to_reflectivity(input_fields, metadata)

    timelist = metadata["timestamps"]

    # First half of timesteps
    tracks_1, cells, labels = dating(
        input_fields[0 : len_timesteps // 2],
        timelist[0 : len_timesteps // 2],
        mintrack=1,
    )
    # Second half of timesteps
    tracks_2, cells, _ = dating(
        input_fields[len_timesteps // 2 - 2 :],
        timelist[len_timesteps // 2 - 2 :],
        mintrack=1,
        start=2,
        cell_list=cells,
        label_list=labels,
    )

    # Since we are adding cells, number of tracks should increase
    assert len(tracks_1) <= len(tracks_2)

    # Tracks should be continuous in time so time difference should not exceed timestep
    max_track_step = max([t.time.diff().max().seconds for t in tracks_2 if len(t) > 1])
    timestep = np.diff(timelist).max().seconds
    assert max_track_step <= timestep

    # IDs of unmatched cells should increase in every timestep
    for prev_df, cur_df in zip(cells[:-1], cells[1:]):
        prev_ids = set(prev_df.ID)
        cur_ids = set(cur_df.ID)
        new_ids = list(cur_ids - prev_ids)
        prev_unmatched = list(prev_ids - cur_ids)
        if len(prev_unmatched):
            assert np.all(np.array(new_ids) > max(prev_unmatched))


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
