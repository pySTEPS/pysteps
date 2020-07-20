# -*- coding: utf-8 -*-

import pytest

from pysteps.visualization import plot_precip_field
from pysteps.utils import to_rainrate
from pysteps.tests.helpers import get_precipitation_fields
import matplotlib.pyplot as pl

pytest.importorskip("cartopy")

plt_arg_names = ("source", "map", "drawlonlatlines", "lw")

plt_arg_values = [
    ("mch", "cartopy", False, 0.5),
    ("mch", "cartopy", True, 1.0),
    ("bom", "cartopy", True, 0.5),
    ("fmi", "cartopy", True, 0.5),
    ("knmi", "cartopy", True, 0.5),
    ("opera", "cartopy", True, 0.5),
    ("mrms", "cartopy", True, 0.5),
]


@pytest.mark.parametrize(plt_arg_names, plt_arg_values)
def test_visualization_plot_precip_field(source, map, drawlonlatlines, lw):

    field, metadata = get_precipitation_fields(0, 0, True, True, None, source)
    field = field.squeeze()
    field, __ = to_rainrate(field, metadata)

    ax = plot_precip_field(
        field,
        type="intensity",
        geodata=metadata,
        map=map,
        drawlonlatlines=drawlonlatlines,
        lw=lw,
    )
    pl.close()


if __name__ == "__main__":

    for i, args in enumerate(plt_arg_values):
        test_visualization_plot_precip_field(*args)
        pl.show()
