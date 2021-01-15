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

################################
# track and contour plots zorder
# - precipitation: 40


def plot_track(track_list, geodata=None, ref_shape=None):
    """
    Plot storm tracks.

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    Parameters
    ----------
    track_list: list
        List of tracks provided by DATing.
    geodata: dictionary or None, optional
        Optional dictionary containing geographical information about
        the field. If not None, plots the contours in a georeferenced frame.
    ref_shape: (vertical, horizontal)
        Shape of the 2D precipitation field used to find the cells' contours.
        This is only needed only if `geodata=None`.

        IMPORTANT: If `geodata=None` it is assumed that the y-origin of the reference
        precipitation fields is the upper-left corner (yorigin="upper").

    Returns
    -------
    ax: fig Axes_
        Figure axes.
    """
    ax = plt.gca()
    pix2coord = _pix2coord_factory(geodata, ref_shape)

    color = iter(plt.cm.spring(np.linspace(0, 1, len(track_list))))
    for track in track_list:
        cen_x, cen_y = pix2coord(track.cen_x, track.cen_y)
        ax.plot(cen_x, cen_y, c=next(color), zorder=40)
    return ax


def plot_cart_contour(contours, geodata=None, ref_shape=None):
    """
    Plots input image with identified cell contours.
    Also, this function can be user to add points of interest to a plot.

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    Parameters
    ----------
    contours: list or dataframe-element
        list of identified cell contours.
    geodata: dictionary or None, optional
        Optional dictionary containing geographical information about
        the field. If not None, plots the contours in a georeferenced frame.
    ref_shape: (vertical, horizontal)
        Shape of the 2D precipitation field used to find the cells' contours.
        This is only needed only if `geodata=None`.

        IMPORTANT: If `geodata=None` it is assumed that the y-origin of the reference
        precipitation fields is the upper-left corner (yorigin="upper").

    Returns
    -------
    ax: fig Axes_
        Figure axes.
    """
    ax = plt.gca()
    pix2coord = _pix2coord_factory(geodata, ref_shape)

    contours = list(contours)
    for contour in contours:
        for c in contour:
            x, y = pix2coord(c[:, 1], c[:, 0])
            ax.plot(x, y, color="black", zorder=40)
    return ax


def _pix2coord_factory(geodata, ref_shape):
    """Construct the pix2coord transformation function."""
    if geodata is not None:

        def pix2coord(x_input, y_input):
            x = geodata["x1"] + geodata["xpixelsize"] * x_input
            if geodata["yorigin"] == "lower":
                y = geodata["y1"] + geodata["ypixelsize"] * y_input
            else:
                y = geodata["y2"] - geodata["ypixelsize"] * y_input
            return x, y

    else:
        if ref_shape is None:
            raise ValueError("'ref_shape' can't be None when not geodata is available.")

        # Default pix2coord function when no geographical information is present.
        def pix2coord(x_input, y_input):
            # yorigin is "upper" by default
            return x_input, ref_shape[0] - y_input

    return pix2coord
