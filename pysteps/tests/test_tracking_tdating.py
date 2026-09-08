# -*- coding: utf-8 -*-

import numpy as np
import pytest
import xarray as xr

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.tracking.tdating import dating
from pysteps.utils import to_reflectivity

arg_names = ("source", "dry_input", "output_splits_merges")

arg_values = [
    ("mch", False, False),
    ("mch", False, False),
    ("mch", True, False),
    ("mch", False, True),
]

arg_names_multistep = ("source", "len_timesteps", "output_splits_merges")
arg_values_multistep = [
    ("mch", 6, False),
    ("mch", 6, True),
]


@pytest.mark.parametrize(arg_names_multistep, arg_values_multistep)
def test_tracking_tdating_dating_multistep(source, len_timesteps, output_splits_merges):
    pytest.importorskip("skimage")

    dataset_input = get_precipitation_fields(0, len_timesteps, True, 4000, source)
    dataset_input = to_reflectivity(dataset_input)

    # First half of timesteps
    tracks_1, cells, labels = dating(
        dataset_input.isel(time=slice(0, len_timesteps // 2)),
        mintrack=1,
        output_splits_merges=output_splits_merges,
    )
    # Second half of timesteps
    tracks_2, cells, _ = dating(
        dataset_input.isel(time=slice(len_timesteps // 2 - 2, None)),
        mintrack=1,
        start=2,
        cell_list=cells,
        label_list=labels,
        output_splits_merges=output_splits_merges,
    )

    # Since we are adding cells, number of tracks should increase
    assert len(tracks_1) <= len(tracks_2)

    # Tracks should be continuous in time so time difference should not exceed timestep
    max_track_step = max([t.time.diff().max().seconds for t in tracks_2 if len(t) > 1])
    timestep = np.diff(dataset_input.time.values).max() / np.timedelta64(1, "s")
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
def test_tracking_tdating_dating(source, dry_input, output_splits_merges):
    pytest.importorskip("skimage")
    pandas = pytest.importorskip("pandas")

    if not dry_input:
        dataset_input = get_precipitation_fields(0, 2, True, 4000, source)
        dataset_input = to_reflectivity(dataset_input)
    else:
        dataset_input = xr.Dataset(
            data_vars={"precip_intensity": (["time", "y", "x"], np.zeros((3, 50, 50)))},
            attrs={"precip_var": "precip_intensity"},
        )

    cell_column_length = 9
    if output_splits_merges:
        cell_column_length = 15

    output = dating(
        dataset_input, mintrack=1, output_splits_merges=output_splits_merges
    )

    # Check output format
    assert isinstance(output, tuple)
    assert len(output) == 3
    assert isinstance(output[0], list)
    assert isinstance(output[1], list)
    assert isinstance(output[2], list)
    assert len(output[1]) == dataset_input.sizes["time"]
    assert len(output[2]) == dataset_input.sizes["time"]
    assert isinstance(output[1][0], pandas.DataFrame)
    assert isinstance(output[2][0], np.ndarray)
    assert output[1][0].shape[1] == cell_column_length
    assert output[2][0].shape == (dataset_input.sizes["y"], dataset_input.sizes["x"])
    if not dry_input:
        assert len(output[0]) > 0
        assert isinstance(output[0][0], pandas.DataFrame)
        assert output[0][0].shape[1] == cell_column_length
    else:
        assert len(output[0]) == 0
        assert output[1][0].shape[0] == 0
        assert output[2][0].sum() == 0
