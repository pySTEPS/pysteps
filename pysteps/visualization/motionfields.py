"""
pysteps.visualization.motionfields
==================================

Functions to plot motion fields.

.. autosummary::
    :toctree: ../generated/

    quiver
    streamplot
"""

import matplotlib.pylab as plt
import matplotlib.colors as colors

import numpy as np

from . import basemaps
from . import utils
    
def quiver(UV, ax=None, map=None, geodata=None, drawlonlatlines=False, 
            basemap_resolution='l', cartopy_scale="50m", lw=0.5, 
            cartopy_subplot=(1,1,1), axis="on", **kwargs):
    """Function to plot a motion field as arrows.

    Parameters
    ----------
    UV : array-like
        Array of shape (2,m,n) containing the input motion field.
    ax : axis object
        Optional axis object to use for plotting.
    map : {'basemap', 'cartopy'}, optional
        Optional method for plotting a map: 'basemap' or 'cartopy'. The former
        uses `mpl_toolkits.basemap`_, while the latter uses cartopy_.
    geodata : dictionary
        Optional dictionary containing geographical information about the field.
        If geodata is not None, it must contain the following key-value pairs:
    drawlonlatlines : bool, optional
        If set to True, draw longitude and latitude lines. Applicable if map is
        'basemap' or 'cartopy'.
    basemap_resolution : str, optional
        The resolution of the basemap, see the documentation of
        `mpl_toolkits.basemap`_.
        Applicable if map is 'basemap'.
    cartopy_scale : {'10m', '50m', '110m'}, optional
        The scale (resolution) of the map. The available options are '10m',
        '50m', and '110m'. Applicable if map is 'cartopy'.
    lw: float, optional
        Linewidth of the map (administrative boundaries and coastlines).
    cartopy_subplot : tuple or SubplotSpec_ instance, optional
        Cartopy subplot. Applicable if map is 'cartopy'.
    axis : {'off','on'}, optional
        Whether to turn off or on the x and y axis.   
        
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
    if map is not None and geodata is None:
        raise ValueError("map!=None but geodata=None")

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
        extent = (geodata['x1'],geodata['x2'], geodata['y1'],geodata['y2'])
    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1])
    
    # draw basemaps
    if map is not None:
        ax,_ = basemaps.plot_geography(map, geodata["projection"], 
                extent, drawlonlatlines, basemap_resolution, 
                cartopy_scale, lw, cartopy_subplot)
    else:
        ax = plt    
        
    # reduce number of vectors to plot
    UV_ = UV[:, 0:UV.shape[1]:step, 0:UV.shape[2]:step]
    y_ = y[0:UV.shape[1]:step]
    x_ = x[0:UV.shape[2]:step]
    
    # plot quiver
    ax.quiver(x_, np.flipud(y_), UV_[0,:,:], -UV_[1,:,:], angles='xy',
              zorder=1e6, **kwargs_quiver)

    if geodata is None or axis == "off":
        axes = plt.gca()
        axes.xaxis.set_ticks([])
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticks([])
        axes.yaxis.set_ticklabels([])

    return plt.gca()

def streamplot(UV, ax=None, map=None, geodata=None, drawlonlatlines=False, 
                basemap_resolution='l', cartopy_scale="50m", lw=0.5, 
                cartopy_subplot=(1,1,1), axis="on", **kwargs):
    """Function to plot a motion field as streamlines.

    Parameters
    ----------
    UV : array-like
        Array of shape (2, m,n) containing the input motion field.
    ax : axis object
        Optional axis object to use for plotting.
    map : {'basemap', 'cartopy'}, optional
        Optional method for plotting a map: 'basemap' or 'cartopy'. The former
        uses `mpl_toolkits.basemap`_, while the latter uses cartopy_.
    geodata : dictionary
        Optional dictionary containing geographical information about the field.
        If geodata is not None, it must contain the following key-value pairs:
    drawlonlatlines : bool, optional
        If set to True, draw longitude and latitude lines. Applicable if map is
        'basemap' or 'cartopy'.
    basemap_resolution : str, optional
        The resolution of the basemap, see the documentation of
        `mpl_toolkits.basemap`_.
        Applicable if map is 'basemap'.
    cartopy_scale : {'10m', '50m', '110m'}, optional
        The scale (resolution) of the map. The available options are '10m',
        '50m', and '110m'. Applicable if map is 'cartopy'.
    lw: float, optional
        Linewidth of the map (administrative boundaries and coastlines).
    cartopy_subplot : tuple or SubplotSpec_ instance, optional
        Cartopy subplot. Applicable if map is 'cartopy'.
    axis : {'off','on'}, optional
        Whether to turn off or on the x and y axis.

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
    if map is not None and geodata is None:
        raise ValueError("map!=None but geodata=None")

    # defaults
    density     = kwargs.get("density", 1.5)
    color       = kwargs.get("color", "black")

    # prepare x y coordinates
    if geodata is not None:
        x = np.linspace(geodata['x1'], geodata['x2'], UV.shape[2])
        y = np.linspace(geodata['y1'], geodata['y2'], UV.shape[1])
        extent = (geodata['x1'],geodata['x2'], geodata['y1'],geodata['y2'])
    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1])
    
    # draw basemaps
    if map is not None:
        ax,_ = basemaps.plot_geography(map, geodata["projection"], 
                        extent, drawlonlatlines, basemap_resolution, 
                        cartopy_scale, lw, cartopy_subplot)
    else:
        ax = plt 
    
    # plot streamplot
    ax.streamplot(x, np.flipud(y), UV[0,:,:], -UV[1,:,:], density=density,
                  color=color, zorder=1e6)

    if geodata is None or axis == "off":
        axes = plt.gca()
        axes.xaxis.set_ticks([])
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticks([])
        axes.yaxis.set_ticklabels([])

    return plt.gca()
