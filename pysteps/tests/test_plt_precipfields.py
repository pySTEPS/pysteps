# -*- coding: utf-8 -*-

import pytest

from pysteps.visualization import plot_precip_field
from pysteps.utils import conversion
from pysteps.postprocessing import ensemblestats
from pysteps.tests.helpers import get_precipitation_fields
import matplotlib.pyplot as pl

plt_arg_names = (
    "source",
    "type",
    "bbox",
    "colorscale",
    "probthr",
    "title",
    "colorbar",
    "axis",
)

plt_arg_values = [
    ("mch", "intensity", None, "pysteps", None, None, False, "off"),
    ("mch", "depth", None, "pysteps", None, "Title", True, "on"),
    ("mch", "prob", None, "pysteps", 0.1, None, True, "on"),
    ("mch", "intensity", None, "STEPS-BE", None, None, True, "on"),
    ("mch", "intensity", None, "BOM-RF3", None, None, True, "on"),
    ("bom", "intensity", None, "pysteps", None, None, True, "on"),
    ("fmi", "intensity", None, "pysteps", None, None, True, "on"),
    ("knmi", "intensity", None, "pysteps", None, None, True, "on"),
    (
        "knmi",
        "intensity",
        [2e2, -4.1e3, 5e2, -3.8e3],
        "pysteps",
        None,
        None,
        True,
        "on",
    ),
    ("opera", "intensity", None, "pysteps", None, None, True, "on"),
    ("saf", "intensity", None, "pysteps", None, None, True, "on"),
]


@pytest.mark.parametrize(plt_arg_names, plt_arg_values)
def test_visualization_plot_precip_field(
    source, type, bbox, colorscale, probthr, title, colorbar, axis,
):

    if type == "intensity":

        field, metadata = get_precipitation_fields(0, 0, True, True, None, source)
        field = field.squeeze()
        field, metadata = conversion.to_rainrate(field, metadata)

    elif type == "depth":

        field, metadata = get_precipitation_fields(0, 0, True, True, None, source)
        field = field.squeeze()
        field, metadata = conversion.to_raindepth(field, metadata)

    elif type == "prob":

        field, metadata = get_precipitation_fields(0, 10, True, True, None, source)
        field, metadata = conversion.to_rainrate(field, metadata)
        field = ensemblestats.excprob(field, probthr)

    ax = plot_precip_field(
        field,
        type=type,
        bbox=bbox,
        geodata=metadata,
        colorscale=colorscale,
        probthr=probthr,
        units=metadata["unit"],
        title=title,
        colorbar=colorbar,
        axis=axis,
    )
    pl.close()


if __name__ == "__main__":

    for i, args in enumerate(plt_arg_values):
        test_visualization_plot_precip_field(*args)
        pl.show()
