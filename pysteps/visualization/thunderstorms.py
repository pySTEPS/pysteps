#!/usr/bin/env python3
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pysteps.visualization import plot_precip_field


def plot_track(track_list, xsize, ysize, poix=None, poiy=None):
    """

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    Parameters
    ----------
    track_list : list
        List of tracks provided by DATing.
    xsize : int
        size of domain in x-direction.
    ysize : int
        size of domain in y-direction.
    poix : array of int, optional
        X-location of points of interest to plot with tracks (e.g. radar locations).
        The default is None.
    poiy : array of int, optional
        Y-location of points of interest to plot with tracks (e.g. radar locations).
        The default is None.

    Returns
    -------
    ax : fig Axes_
        Figure axes.
    """
    ax = plt.gca()
    ax.set_ylim(0, ysize)
    ax.set_xlim(0, xsize)
    if poix is not None and poiy is not None:
        p1 = ax.scatter(poix, poiy, s=None, c="black")
    color = iter(plt.cm.spring(np.linspace(0, 1, len(track_list))))
    for track in track_list:
        p2 = ax.plot(track.max_x, track.max_y, c=next(color))
    return ax


def plot_cart_contour(contours, poix=None, poiy=None):
    """
    Plots input image with identified cell contours. Optionally points of interest added.

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    Parameters
    ----------
    contours : list or dataframe-element
        list of identified cell contours.
    poix : array-like, optional
        1-D array of x-locations of points of interest. The default is None.
    poiy : array-like, optional
        1-D array, same length as poix, y-locations of points of interest.
        The default is None.

    Returns
    -------
    ax : fig Axes_
        Figure axes.
    """
    ax = plt.gca()
    p1 = ax.scatter(poix, poiy, s=None, c="black")
    contours = list(contours)
    for contour in contours:
        for c in contour:
            p1 = ax.plot(c[:, 1], c[:, 0], color="black")
        # else: p1=plt.plot(contour[:,1], contour[:,0], color='black')
    return ax
