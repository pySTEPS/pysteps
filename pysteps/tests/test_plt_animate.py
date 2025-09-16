# -*- coding: utf-8 -*-

import os

import xarray as xr
import pytest
from unittest.mock import patch
from pysteps.tests.helpers import get_precipitation_fields
from pysteps.visualization.animations import animate
from datetime import datetime


precip_dataset: xr.Dataset = get_precipitation_fields(
    num_prev_files=2,
    num_next_files=0,
    return_raw=True,
    upscale=2000,
)

precip_var = precip_dataset.attrs["precip_var"]
precip_dataarray = precip_dataset[precip_var]

geodata = {
    "projection": precip_dataset.attrs["projection"],
    "x1": precip_dataset.x.values[0],
    "x2": precip_dataset.x.values[-1],
    "y1": precip_dataset.y.values[0],
    "y2": precip_dataset.y.values[-1],
    "yorigin": "lower",
}

motion_fields_dataset = xr.Dataset(
    data_vars={
        "velocity_x": xr.ones_like(precip_dataarray.isel(time=0)),
        "velocity_y": xr.ones_like(precip_dataarray.isel(time=0)),
    }
)

ensemble_forecast = xr.concat([precip_dataset, precip_dataset], dim="ens_member")

ensemble_forecast = ensemble_forecast.assign_coords(ens_member=[0, 1])

# Need to convert timestamp objects to datetime as animations.py calls strfime on timestamp obs
# Only datetime64us converts to datetime cleanly. Other variants convert to int.
timestamps_obs = precip_dataset.time.values.astype("datetime64[us]").astype(datetime)

# NOTE:
# calling .values on precip_dataarray to convert it to a numpy array is required each time
# animate uses numerical indexing. For consistency it has been applied everywhere.
VALID_ARGS = (
    ([precip_dataarray.values], {}),
    ([precip_dataarray.values], {"title": "title"}),
    ([precip_dataarray.values], {"timestamps_obs": timestamps_obs}),
    ([precip_dataarray.values], {"geodata": geodata, "map_kwargs": {"plot_map": None}}),
    (
        [precip_dataarray.values],
        {"motion_field": motion_fields_dataset.to_array().values},
    ),
    (
        [precip_dataarray.values],
        {"precip_kwargs": {"units": "mm/h", "colorbar": True, "colorscale": "pysteps"}},
    ),
    ([precip_dataarray.values, precip_dataarray.values], {}),
    ([precip_dataarray.values, precip_dataarray.values], {"title": "title"}),
    (
        [precip_dataarray.values, precip_dataarray.values],
        {"timestamps_obs": timestamps_obs},
    ),
    (
        [precip_dataarray.values, precip_dataarray.values],
        {"timestamps_obs": timestamps_obs},
    ),
    (
        [precip_dataarray.values, precip_dataarray.values],
        {"ptype": "prob", "prob_thr": 1},
    ),
    ([precip_dataarray.values, precip_dataarray.values], {"ptype": "mean"}),
    # XR: Not passing in an ensemble forecast here technically, test still works
    ([ensemble_forecast[precip_var][0]], {"ptype": "ensemble"}),
)


@pytest.mark.parametrize(["anim_args", "anim_kwargs"], VALID_ARGS)
def test_animate(anim_args, anim_kwargs):
    with patch("matplotlib.pyplot.show"):
        animate(*anim_args, **anim_kwargs)


VALUEERROR_ARGS = (
    ([precip_dataarray.values], {"timestamps_obs": timestamps_obs[:2]}),
    ([precip_dataarray.values], {"motion_plot": "test"}),
    ([precip_dataarray.values, precip_dataarray.values], {"ptype": "prob"}),
)


@pytest.mark.parametrize(["anim_args", "anim_kwargs"], VALUEERROR_ARGS)
def test_animate_valueerrors(anim_args, anim_kwargs):
    with pytest.raises(ValueError):
        animate(*anim_args, **anim_kwargs)


TYPEERROR_ARGS = (
    ([precip_dataarray.values], {"timestamps": timestamps_obs[:2]}),
    ([precip_dataarray.values], {"plotanimation": True}),
    ([precip_dataarray.values], {"units": "mm/h"}),
    ([precip_dataarray.values], {"colorbar": True}),
    ([precip_dataarray.values], {"colorscale": "pysteps"}),
    ([ensemble_forecast], {"type": "ensemble"}),
)


@pytest.mark.parametrize(["anim_args", "anim_kwargs"], TYPEERROR_ARGS)
def test_animate_typeerrors(anim_args, anim_kwargs):
    with pytest.raises(TypeError):
        animate(*anim_args, **anim_kwargs)


def test_animate_save(tmp_path):
    animate(
        precip_dataset[precip_var],
        ensemble_forecast[precip_var],
        display_animation=False,
        savefig=True,
        path_outputs=tmp_path,
        fig_dpi=10,
    )
    assert len(os.listdir(tmp_path)) == 9
