# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.quiver import QuiverKey

from pysteps import motion
from pysteps.visualization import plot_precip_field, quiver, streamplot
from pysteps.visualization.motionfields import motion_plot
from pysteps.tests.helpers import get_precipitation_fields

arg_names_quiver = (
    "source",
    "axis",
    "step",
    "quiver_kwargs",
    "map_kwargs",
    "upscale",
    "pass_geodata",
)

arg_values_quiver = [
    (None, "off", 10, {}, {"drawlonlatlines": False, "lw": 0.5}, None, False),
    ("bom", "on", 10, {}, {"drawlonlatlines": False, "lw": 0.5}, 4000, False),
    ("bom", "on", 10, {}, {"drawlonlatlines": True, "lw": 0.5}, 4000, True),
    ("mch", "on", 20, {}, {"drawlonlatlines": False, "lw": 0.5}, 2000, True),
]


@pytest.mark.parametrize(arg_names_quiver, arg_values_quiver)
def test_visualization_motionfields_quiver(
    source, axis, step, quiver_kwargs, map_kwargs, upscale, pass_geodata
):
    pytest.importorskip("cv2")
    if source is not None:
        fields, geodata = get_precipitation_fields(0, 2, False, True, upscale, source)
        if not pass_geodata:
            geodata = None
        ax = plot_precip_field(fields[-1], geodata=geodata)
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

    UV_orig = UV.copy()
    __ = quiver(UV, ax, geodata, axis, step, quiver_kwargs, map_kwargs=map_kwargs)

    # Check that quiver does not modify the input data
    assert np.array_equal(UV, UV_orig)


arg_names_streamplot = (
    "source",
    "axis",
    "streamplot_kwargs",
    "map_kwargs",
    "upscale",
    "pass_geodata",
)

arg_values_streamplot = [
    (None, "off", {}, {"drawlonlatlines": False, "lw": 0.5}, None, False),
    ("bom", "on", {}, {"drawlonlatlines": False, "lw": 0.5}, 4000, False),
    ("bom", "on", {"density": 0.5}, {"drawlonlatlines": True, "lw": 0.5}, 4000, True),
]


@pytest.mark.parametrize(arg_names_streamplot, arg_values_streamplot)
def test_visualization_motionfields_streamplot(
    source, axis, streamplot_kwargs, map_kwargs, upscale, pass_geodata
):
    pytest.importorskip("cv2")
    if source is not None:
        fields, geodata = get_precipitation_fields(0, 2, False, True, upscale, source)
        if not pass_geodata:
            pass_geodata = None
        ax = plot_precip_field(fields[-1], geodata=geodata)
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

    UV_orig = UV.copy()
    __ = streamplot(UV, ax, geodata, axis, streamplot_kwargs, map_kwargs=map_kwargs)

    # Check that streamplot does not modify the input data
    assert np.array_equal(UV, UV_orig)


def _get_quiverkeys(ax):
    return [artist for artist in ax.get_children() if isinstance(artist, QuiverKey)]


def _constant_motion_field(u, v, shape=(50, 50)):
    return np.stack([np.full(shape, u, dtype=float), np.full(shape, v, dtype=float)])


def test_visualization_motionfields_quiver_no_quiverkey_by_default():
    UV = _constant_motion_field(3.0, 4.0)
    ax = quiver(UV, step=10)
    assert _get_quiverkeys(ax) == []
    plt.close("all")


def test_visualization_motionfields_quiverkey_defaults():
    # max magnitude is 5 (3-4-5 triangle), so the default key length is 5
    UV = _constant_motion_field(3.0, 4.0)
    ax = quiver(UV, step=10, quiverkey_kwargs={})

    quiverkeys = _get_quiverkeys(ax)
    assert len(quiverkeys) == 1
    qk = quiverkeys[0]
    assert qk.U == 5
    assert qk.text.get_text() == "5 px timestep$^{-1}$"
    assert qk.coord == "axes"
    assert qk.labelpos == "E"
    plt.close("all")


@pytest.mark.parametrize(
    "u, v, expected_U",
    [(12.3, 0.0, 10), (0.27, 0.0, 0.3), (0.0, 0.0, 1), (np.nan, np.nan, 1)],
)
def test_visualization_motionfields_quiverkey_default_length(u, v, expected_U):
    UV = _constant_motion_field(u, v)
    ax = quiver(UV, step=10, quiverkey_kwargs={})
    qk = _get_quiverkeys(ax)[0]
    assert qk.U == pytest.approx(expected_U)
    plt.close("all")


def test_visualization_motionfields_quiverkey_custom():
    UV = _constant_motion_field(3.0, 4.0)
    quiverkey_kwargs = {
        "X": 0.1,
        "Y": 0.2,
        "U": 2,
        "label": "10 km/h",
        "labelpos": "N",
        "coordinates": "figure",
        "color": "red",
    }
    ax = quiver(UV, step=10, quiverkey_kwargs=quiverkey_kwargs)

    qk = _get_quiverkeys(ax)[0]
    assert (qk.X, qk.Y, qk.U) == (0.1, 0.2, 2)
    assert qk.text.get_text() == "10 km/h"
    assert qk.labelpos == "N"
    assert qk.coord == "figure"
    assert qk.color == "red"
    # the input dictionary is not modified
    assert quiverkey_kwargs["U"] == 2 and "X" in quiverkey_kwargs
    plt.close("all")


def test_visualization_motionfields_quiverkey_streamplot_raises():
    UV = _constant_motion_field(3.0, 4.0)
    with pytest.raises(ValueError):
        motion_plot(UV, plot_type="streamplot", quiverkey_kwargs={})
    plt.close("all")


if __name__ == "__main__":
    for i, args in enumerate(arg_values_quiver):
        test_visualization_motionfields_quiver(*args)
        plt.show()

    for i, args in enumerate(arg_values_streamplot):
        test_visualization_motionfields_streamplot(*args)
        plt.show()
