# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pysteps.postprocessing import ensemblestats
from pysteps.tests.helpers import get_precipitation_fields
from pysteps.utils import conversion
from pysteps.visualization import plot_precip_field

plt_arg_names = (
    "source",
    "plot_type",
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
    ("knmi", "intensity", None, "STEPS-NL", None, None, True, "on"),
    ("knmi", "intensity", [300, 300, 500, 500], "pysteps", None, None, True, "on"),
    ("opera", "intensity", None, "pysteps", None, None, True, "on"),
    ("saf", "intensity", None, "pysteps", None, None, True, "on"),
]


@pytest.mark.parametrize(plt_arg_names, plt_arg_values)
def test_visualization_plot_precip_field(
    source, plot_type, bbox, colorscale, probthr, title, colorbar, axis
):
    if plot_type == "intensity":
        dataset = get_precipitation_fields(0, 0, True, None, source)
        dataset = conversion.to_rainrate(dataset)

    elif plot_type == "depth":
        dataset = get_precipitation_fields(0, 0, True, None, source)
        dataset = conversion.to_raindepth(dataset)

    elif plot_type == "prob":
        dataset = get_precipitation_fields(0, 10, True, None, source)
        dataset = conversion.to_rainrate(dataset)

    precip_var = dataset.attrs["precip_var"]
    field = dataset[precip_var].values
    if plot_type == "prob":
        field = ensemblestats.excprob(field, probthr)

    field = field.squeeze()
    field_orig = field.copy()
    ax = plot_precip_field(
        field.copy(),
        ptype=plot_type,
        bbox=bbox,
        geodata=None,
        colorscale=colorscale,
        probthr=probthr,
        units=dataset[precip_var].attrs["units"],
        title=title,
        colorbar=colorbar,
        axis=axis,
    )

    # Check that plot_precip_field does not modify the input data
    field_orig = np.ma.masked_invalid(field_orig)
    field_orig.data[field_orig.mask] = -100
    field = np.ma.masked_invalid(field)
    field.data[field.mask] = -100
    assert np.array_equal(field_orig.data, field.data)


if __name__ == "__main__":
    for i, args in enumerate(plt_arg_values):
        test_visualization_plot_precip_field(*args)
        plt.show()
