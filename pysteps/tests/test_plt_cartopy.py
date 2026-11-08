# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
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


def test_visualization_plot_precip_field_no_extra_axes():
    """The regular axis replaced by the map must not be left in the figure."""
    pytest.importorskip("cartopy")
    pytest.importorskip("pyproj")
    from cartopy.mpl.geoaxes import GeoAxes

    geodata = {
        "projection": "+proj=stere +lat_0=90 +lon_0=25 +lat_ts=60 +ellps=WGS84",
        "x1": 0.0,
        "x2": 500000.0,
        "y1": 6500000.0,
        "y2": 7000000.0,
        "yorigin": "upper",
    }
    field = np.linspace(0.0, 10.0, 20 * 20).reshape(20, 20)

    fig = plt.figure()
    try:
        ax = plot_precip_field(
            field,
            geodata=geodata,
            colorbar=False,
            map_kwargs={"cartopy_scale": "110m", "drawlonlatlines": False},
        )
        assert isinstance(ax, GeoAxes)
        assert fig.axes == [ax]
    finally:
        plt.close(fig)


if __name__ == "__main__":
    for i, args in enumerate(plt_arg_values):
        test_visualization_plot_precip_field(*args)
        plt.show()
