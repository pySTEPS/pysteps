"""Functions to plot motion fields."""

import matplotlib.pylab as plt
import matplotlib.colors as colors

import numpy as np

def quiver(UV, ax=None, geodata=None, **kwargs):
    """Function to plot a motion field as arrows.

    Parameters
    ----------
    UV : array-like
        Array of shape (2,m,n) containing the input motion field.
    ax : axis object
        Optional axis object to use for plotting.
    geodata : dictionary
        Optional dictionary containing geographical information about the field.
        If geodata is not None, it must contain the following key-value pairs:

        .. tabularcolumns:: |p{1.5cm}|L|

        +-----------------+----------------------------------------------------+
        |        Key      |                  Value                             |
        +=================+====================================================+
        |   projection    | PROJ.4-compatible projection definition            |
        +-----------------+----------------------------------------------------+
        |    x1           | x-coordinate of the lower-left corner of the data  |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    y1           | y-coordinate of the lower-left corner of the data  |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    x2           | x-coordinate of the upper-right corner of the data |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    y2           | y-coordinate of the upper-right corner of the data |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    yorigin      | a string specifying the location of the first      |
        |                 | element in the data raster w.r.t. y-axis:          |
        |                 | 'upper' = upper border, 'lower' = lower border     |
        +-----------------+----------------------------------------------------+

    Other Parameters
    ----------------
    step : int
        Optional resample step to control the density of the arrows.
        Default : 20
    color : string
        Optional color of the arrows. This is a synonym for the PolyCollection
        facecolor kwarg in matplotlib.collections.
        Default : black

    Returns
    -------
    out : axis object
        Figure axes. Needed if one wants to add e.g. text inside the plot.

    """
    if ax is None:
        ax = plt

    # defaults
    step        = kwargs.get("step", 20)

    quiver_keys = ["scale", "scale_units", "width", "headwidth", "headlength", 
                   "headaxislength", "minshaft", "minlength", "pivot", "color"]
    kwargs_quiver = {k: kwargs[k] for k in set(quiver_keys).intersection(kwargs)}

    kwargs_quiver["color"] = kwargs.get("color", "black")

    # prepare x y coordinates
    if geodata is not None:
        x = np.linspace(geodata['x1'], geodata['x2'], UV.shape[2])
        y = np.linspace(geodata['y1'], geodata['y2'], UV.shape[1])
    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1])

    # reduce number of vectors to plot
    UV_ = UV[:, 0:UV.shape[1]:step, 0:UV.shape[2]:step]
    y_ = y[0:UV.shape[1]:step]
    x_ = x[0:UV.shape[2]:step]

    ax.quiver(x_, np.flipud(y_), UV_[0,:,:], -UV_[1,:,:], angles='xy',
              zorder=1e6, **kwargs_quiver)

    axes = plt.gca() if ax == plt else ax

    if geodata is None:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])

    return axes

def streamplot(UV, ax=None, geodata=None, **kwargs):
    """Function to plot a motion field as streamlines.

    Parameters
    ----------
    UV : array-like
        Array of shape (2, m,n) containing the input motion field.
    ax : axis object
        Optional axis object to use for plotting.
    geodata : dictionary
        Optional dictionary containing geographical information about the field.
        If geodata is not None, it must contain the following key-value pairs:

        .. tabularcolumns:: |p{1.5cm}|L|

        +-----------------+----------------------------------------------------+
        |        Key      |                  Value                             |
        +=================+====================================================+
        |   projection    | PROJ.4-compatible projection definition            |
        +-----------------+----------------------------------------------------+
        |    x1           | x-coordinate of the lower-left corner of the data  |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    y1           | y-coordinate of the lower-left corner of the data  |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    x2           | x-coordinate of the upper-right corner of the data |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    y2           | y-coordinate of the upper-right corner of the data |
        |                 | raster (meters)                                    |
        +-----------------+----------------------------------------------------+
        |    yorigin      | a string specifying the location of the first      |
        |                 | element in the data raster w.r.t. y-axis:          |
        |                 | 'upper' = upper border, 'lower' = lower border     |
        +-----------------+----------------------------------------------------+

    Other Parameters
    ----------------
    density : float
        Controls the closeness of streamlines.
        Default : 1.5
    color : string
        Optional streamline color. This is a synonym for the PolyCollection
        facecolor kwarg in matplotlib.collections.
        Default : black

    Returns
    -------
    out : axis object
        Figure axes. Needed if one wants to add e.g. text inside the plot.

    """
    if ax is None:
        ax = plt

    # defaults
    density     = kwargs.get("density", 1.5)
    color       = kwargs.get("color", "black")

    # prepare x y coordinates
    if geodata is not None:
        x = np.linspace(geodata['x1'], geodata['x2'], UV.shape[2])
        y = np.linspace(geodata['y1'], geodata['y2'], UV.shape[1])
    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1],0,-1)

    ax.streamplot(x, np.flipud(y), UV[0,:,:], -UV[1,:,:], density=density,
                  color=color, zorder=1e6)

    axes = plt.gca() if ax == plt else ax

    if geodata is None:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])

    return axes
