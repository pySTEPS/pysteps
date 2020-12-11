# -*- coding: utf-8 -*-
"""
pysteps.visualization.tstorm
============================

Methods for plotting thunderstorm cells.

Created on Wed Nov  4 11:09:44 2020

@author: mfeldman

.. autosummary::
    :toctree: ../generated/

    plot_track
    plot_cart_contour
"""
import matplotlib.pyplot as plt
import numpy as np

try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False


def plot_track(track_list, geodata=None):
    """

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    Parameters
    ----------
    track_list: list
        List of tracks provided by DATing.
    geodata: dictionary or None, optional
        Optional dictionary containing geographical information about
        the field. If not None, plots the contours in a georeferenced frame.

    Returns
    -------
    ax: fig Axes_
        Figure axes.
    """
    ax = plt.gca()

    if geodata is not None:

        def pix2coord(nx, ny):
            x = geodata["x1"] + geodata["xpixelsize"] * nx
            if geodata["yorigin"] == "lower":
                y = geodata["y1"] + geodata["ypixelsize"] * ny
            else:
                y = geodata["y2"] - geodata["ypixelsize"] * ny
            return x, y

    else:

        def pix2coord(nx, ny):
            return nx, ny

    color = iter(plt.cm.spring(np.linspace(0, 1, len(track_list))))
    for track in track_list:
        cen_x, cen_y = pix2coord(track.cen_x, track.cen_y)
        ax.plot(cen_x, cen_y, c=next(color))
    return ax


def plot_cart_contour(contours, geodata=None):
    """
    Plots input image with identified cell contours. Optionally points of interest added.

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    Parameters
    ----------
    contours: list or dataframe-element
        list of identified cell contours.
    geodata: dictionary or None, optional
        Optional dictionary containing geographical information about
        the field. If not None, plots the contours in a georeferenced frame.

    Returns
    -------
    ax: fig Axes_
        Figure axes.
    """
    ax = plt.gca()

    if geodata is not None:

        def pix2coord(nx, ny):
            x = geodata["x1"] + geodata["xpixelsize"] * nx
            if geodata["yorigin"] == "lower":
                y = geodata["y1"] + geodata["ypixelsize"] * ny
            else:
                y = geodata["y2"] - geodata["ypixelsize"] * ny
            return x, y

    else:

        def pix2coord(nx, ny):
            return nx, ny

    contours = list(contours)
    for contour in contours:
        for c in contour:
            x, y = pix2coord(c[:, 1], c[:, 0])
            p1 = ax.plot(x, y, color="black")
        # else: p1=plt.plot(contour[:,1], contour[:,0], color='black')
    return ax
