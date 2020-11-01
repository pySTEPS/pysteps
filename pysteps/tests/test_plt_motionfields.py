# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pysteps import motion
from pysteps.visualization import plot_precip_field, quiver, streamplot
from pysteps.tests.helpers import get_precipitation_fields


arg_names_quiver = (
    "source",
    "axis",
    "step",
    "quiver_kwargs",
    "plot_map",
    "drawlonlatlines",
    "lw",
    "upscale",
)

arg_values_quiver = [
    (None, "off", 10, None, None, False, 0.5, None),
    ("bom", "on", 10, None, None, False, 0.5, 4000),
    ("bom", "on", 10, None, "cartopy", True, 0.5, 4000),
    ("mch", "on", 20, None, "cartopy", False, 0.5, 2000),
    ("bom", "off", 10, None, "basemap", False, 0.5, 4000),
]


@pytest.mark.parametrize(arg_names_quiver, arg_values_quiver)
def test_visualization_motionfields_quiver(
    source, axis, step, quiver_kwargs, plot_map, drawlonlatlines, lw, upscale,
):

    if plot_map == "cartopy":
        pytest.importorskip("cartopy")
    elif plot_map == "basemap":
        pytest.importorskip("basemap")

    if source is not None:
        fields, geodata = get_precipitation_fields(0, 2, False, True, upscale, source)
        ax = plot_precip_field(fields[-1], geodata=geodata, plot_map=plot_map,)
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

    __ = quiver(
        UV,
        ax,
        geodata,
        axis,
        step,
        quiver_kwargs,
        plot_map=plot_map,
        drawlonlatlines=drawlonlatlines,
        lw=lw,
    )


arg_names_streamplot = (
    "source",
    "axis",
    "streamplot_kwargs",
    "plot_map",
    "drawlonlatlines",
    "lw",
    "upscale",
)

arg_values_streamplot = [
    (None, "off", None, None, False, 0.5, None),
    ("bom", "on", None, None, False, 0.5, 4000),
    ("bom", "on", {"density": 0.1}, "cartopy", True, 0.5, 4000),
]


@pytest.mark.parametrize(arg_names_streamplot, arg_values_streamplot)
def test_visualization_motionfields_streamplot(
    source, axis, streamplot_kwargs, plot_map, drawlonlatlines, lw, upscale,
):

    if plot_map == "cartopy":
        pytest.importorskip("cartopy")
    elif plot_map == "basemap":
        pytest.importorskip("basemap")

    if source is not None:
        fields, geodata = get_precipitation_fields(0, 2, False, True, upscale, source)
        ax = plot_precip_field(fields[-1], geodata=geodata, plot_map=plot_map,)
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

    __ = streamplot(
        UV,
        ax,
        geodata,
        axis,
        streamplot_kwargs,
        plot_map=plot_map,
        drawlonlatlines=drawlonlatlines,
        lw=lw,
    )


if __name__ == "__main__":

    for i, args in enumerate(arg_values_quiver):
        test_visualization_motionfields_quiver(*args)
        plt.show()

    for i, args in enumerate(arg_values_streamplot):
        test_visualization_motionfields_streamplot(*args)
        plt.show()
