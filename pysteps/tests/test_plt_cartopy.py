# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pytest

from pysteps.tests.helpers import get_precipitation_fields
from pysteps.utils import to_rainrate
from pysteps.visualization import plot_precip_field

plt_arg_names = ("source", "map_kwargs", "pass_geodata")

plt_arg_values = [
    ("mch", {"drawlonlatlines": False, "lw": 0.5, "plot_map": None}, False),
    ("mch", {"drawlonlatlines": False, "lw": 0.5, "plot_map": "cartopy"}, False),
    ("mch", {"drawlonlatlines": False, "lw": 0.5}, True),
    ("mch", {"drawlonlatlines": True, "lw": 1.0}, True),
    ("bom", {"drawlonlatlines": True, "lw": 0.5}, True),
    ("fmi", {"drawlonlatlines": True, "lw": 0.5}, True),
    ("knmi", {"drawlonlatlines": True, "lw": 0.5}, True),
    ("opera", {"drawlonlatlines": True, "lw": 0.5}, True),
    ("mrms", {"drawlonlatlines": True, "lw": 0.5}, True),
    ("saf", {"drawlonlatlines": True, "lw": 0.5}, True),
]


@pytest.mark.parametrize(plt_arg_names, plt_arg_values)
def test_visualization_plot_precip_field(source, map_kwargs, pass_geodata):
    dataset = get_precipitation_fields(0, 0, True, None, source)
    dataset = to_rainrate(dataset)

    precip_var = dataset.attrs["precip_var"]
    field = dataset[precip_var].values
    field = field.squeeze()
    geodata = {
        "projection": dataset.attrs["projection"],
        "x1": dataset.x.values[0],
        "x2": dataset.x.values[-1],
        "y1": dataset.y.values[0],
        "y2": dataset.y.values[-1],
        "yorigin": "lower",
    }
    if not pass_geodata:
        geodata = None

    plot_precip_field(field, ptype="intensity", geodata=geodata, map_kwargs=map_kwargs)


if __name__ == "__main__":
    for i, args in enumerate(plt_arg_values):
        test_visualization_plot_precip_field(*args)
        plt.show()
