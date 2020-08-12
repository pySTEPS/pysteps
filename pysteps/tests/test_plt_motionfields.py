# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pysteps import motion
from pysteps.visualization import plot_precip_field, quiver, streamplot
from pysteps.tests.helpers import get_precipitation_fields


arg_names_quiver = (
    "source",
    "map",
    "drawlonlatlines",
    "lw",
    "axis",
    "step",
    "quiver_kwargs",
    "upscale",
)

arg_values_quiver = [
    (None, None, False, 0.5, "off", 10, None, None),
    ("bom", None, False, 0.5, "on", 10, None, 4000),
    ("bom", "cartopy", True, 0.5, "on", 10, None, 4000),
    ("mch", "cartopy", False, 0.5, "on", 20, None, 2000),
    ("bom", "basemap", False, 0.5, "off", 10, None, 4000),
]


@pytest.mark.parametrize(arg_names_quiver, arg_values_quiver)
def test_visualization_motionfields_quiver(
    source, map, drawlonlatlines, lw, axis, step, quiver_kwargs, upscale,
):

    if map == "cartopy":
        pytest.importorskip("cartopy")
    elif map == "basemap":
        pytest.importorskip("basemap")

    if source is not None:
        fields, geodata = get_precipitation_fields(0, 2, False, True, upscale, source)
        ax = plot_precip_field(fields[-1], map=map, geodata=geodata,)
        oflow_method = motion.get_method("LK")
        UV = oflow_method(fields)

    else:
        shape = (100, 100)
        geodata = None
        ax = None
        u = np.ones(shape[1]) * shape[0]
        v = np.arange(0, shape[0])
        U, V = np.meshgrid(u, v)
        UV = np.concatenate([U[None, :], V[None, :]])

    __ = quiver(UV, ax, map, geodata, drawlonlatlines, lw, axis, step, quiver_kwargs)


arg_names_streamplot = (
    "source",
    "map",
    "drawlonlatlines",
    "lw",
    "axis",
    "streamplot_kwargs",
    "upscale",
)

arg_values_streamplot = [
    (None, None, False, 0.5, "off", None, None),
    ("bom", None, False, 0.5, "on", None, 4000),
    ("bom", "cartopy", True, 0.5, "on", {"density": 0.1}, 4000),
]


@pytest.mark.parametrize(arg_names_streamplot, arg_values_streamplot)
def test_visualization_motionfields_streamplot(
    source, map, drawlonlatlines, lw, axis, streamplot_kwargs, upscale,
):

    if map == "cartopy":
        pytest.importorskip("cartopy")
    elif map == "basemap":
        pytest.importorskip("basemap")

    if source is not None:
        fields, geodata = get_precipitation_fields(0, 2, False, True, upscale, source)
        ax = plot_precip_field(fields[-1], map=map, geodata=geodata,)
        oflow_method = motion.get_method("LK")
        UV = oflow_method(fields)

    else:
        shape = (100, 100)
        geodata = None
        ax = None
        u = np.ones(shape[1]) * shape[0]
        v = np.arange(0, shape[0])
        U, V = np.meshgrid(u, v)
        UV = np.concatenate([U[None, :], V[None, :]])

    __ = streamplot(UV, ax, map, geodata, drawlonlatlines, lw, axis, streamplot_kwargs)


if __name__ == "__main__":

    for i, args in enumerate(arg_values_quiver):
        test_visualization_motionfields_quiver(*args)
        plt.show()

    for i, args in enumerate(arg_values_streamplot):
        test_visualization_motionfields_streamplot(*args)
        plt.show()
