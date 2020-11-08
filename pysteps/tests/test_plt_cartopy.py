# -*- coding: utf-8 -*-

import pytest

from pysteps.visualization import plot_precip_field
from pysteps.utils import to_rainrate
from pysteps.tests.helpers import get_precipitation_fields
import matplotlib.pyplot as plt


plt_arg_names = ("source", "drawlonlatlines", "lw", "pass_geodata")

plt_arg_values = [
    ("mch", False, 0.5, False),
    ("mch", False, 0.5, True),
    ("mch", True, 1.0, True),
    ("bom", True, 0.5, True),
    ("fmi", True, 0.5, True),
    ("knmi", True, 0.5, True),
    ("opera", True, 0.5, True),
    ("mrms", True, 0.5, True),
    ("saf", True, 0.5, True),
]


@pytest.mark.parametrize(plt_arg_names, plt_arg_values)
def test_visualization_plot_precip_field(source, drawlonlatlines, lw, pass_geodata):

    field, metadata = get_precipitation_fields(0, 0, True, True, None, source)
    field = field.squeeze()
    field, __ = to_rainrate(field, metadata)

    if not pass_geodata:
        metadata = None

    ax = plot_precip_field(
        field,
        type="intensity",
        geodata=metadata,
        drawlonlatlines=drawlonlatlines,
        lw=lw,
    )


if __name__ == "__main__":

    for i, args in enumerate(plt_arg_values):
        test_visualization_plot_precip_field(*args)
        plt.show()
