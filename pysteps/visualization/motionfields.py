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
import numpy as np
from pysteps.exceptions import UnsupportedSomercProjection

from . import basemaps
from . import utils


def quiver(
    UV,
    ax=None,
    map=None,
    geodata=None,
    drawlonlatlines=False,
    lw=0.5,
    axis="on",
    step=20,
    quiver_kwargs=None,
    **kwargs,
):
    """Function to plot a motion field as arrows.

    .. _`mpl_toolkits.basemap`: https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap
    .. _cartopy: https://scitools.org.uk/cartopy/docs/latest/
    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html?highlight=subplotspec#matplotlib.gridspec.SubplotSpec

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
        Optional dictionary containing geographical information about
        the field.
        
        If geodata is not None, it must contain the following key-value pairs:
        
        .. tabularcolumns:: |p{1.5cm}|L|

        +----------------+----------------------------------------------------+
        |        Key     |                  Value                             |
        +================+====================================================+
        |   projection   | PROJ.4-compatible projection definition            |
        +----------------+----------------------------------------------------+
        |    x1          | x-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y1          | y-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    x2          | x-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y2          | y-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    yorigin      | a string specifying the location of the first     |
        |                 | element in the data raster w.r.t. y-axis:         |
        |                 | 'upper' = upper border, 'lower' = lower border    |
        +-----------------+---------------------------------------------------+
    drawlonlatlines : bool, optional
        If set to True, draw longitude and latitude lines. Applicable if map is
        'basemap' or 'cartopy'.
    lw: float, optional
        Linewidth of the map (administrative boundaries and coastlines).
    axis : {'off','on'}, optional
        Whether to turn off or on the x and y axis.
    step : int
        Optional resample step to control the density of the arrows.
    quiver_kwargs : dict, optional
      Optional dictionary containing keyword arguments for the quiver method.
      See the documentation of matplotlib.pyplot.quiver.
    
    
    Other parameters
    ----------------
    Optional parameters are contained in **kwargs. See basemaps.plot_geography.

    Returns
    -------
    out : axis object
        Figure axes. Needed if one wants to add e.g. text inside the plot.

    """
    if map is not None:
        FutureWarning(
            "'map' argument will be renamed to 'plot_map' in 1.4.0. Use 'plot_map' to silence this warning."
        )
        plot_map = map
    else:
        plot_map = kwargs.pop("plot_map", None)

    if plot_map is not None and geodata is None:
        raise ValueError("map!=None but geodata=None")

    if quiver_kwargs is None:
        quiver_kwargs = dict()

    # prepare x y coordinates
    reproject = False
    if geodata is not None:
        xmin = min((geodata["x1"], geodata["x2"]))
        xmax = max((geodata["x1"], geodata["x2"]))
        x = np.linspace(xmin, xmax, UV.shape[2])
        xpixelsize = np.abs(x[1] - x[0])
        x += xpixelsize / 2.0

        ymin = min((geodata["y1"], geodata["y2"]))
        ymax = max((geodata["y1"], geodata["y2"]))
        y = np.linspace(ymin, ymax, UV.shape[1])
        ypixelsize = np.abs(y[1] - y[0])
        y += ypixelsize / 2.0

        extent = (geodata["x1"], geodata["x2"], geodata["y1"], geodata["y2"])

        # check geodata and project if different from axes
        if ax is not None and plot_map is None:
            if type(ax).__name__ == "GeoAxesSubplot":
                try:
                    ccrs = utils.proj4_to_cartopy(geodata["projection"])
                except UnsupportedSomercProjection:
                    # Define fall-back projection for Swiss data(EPSG:3035)
                    # This will work reasonably well for Europe only.
                    t_proj4str = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs"
                    reproject = True
            elif type(ax).__name__ == "Basemap":
                utils.proj4_to_basemap(geodata["projection"])

            if reproject:
                geodata = utils.reproject_geodata(
                    geodata, t_proj4str, return_grid="coords"
                )
                extent = (geodata["x1"], geodata["x2"], geodata["y1"], geodata["y2"])
                X, Y = geodata["X_grid"], geodata["Y_grid"]

    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1])

    if not reproject:
        X, Y = np.meshgrid(x, y)

    # draw basemaps
    if plot_map is not None:
        try:
            ax = basemaps.plot_geography(
                plot_map,
                geodata["projection"],
                extent,
                lw=lw,
                drawlonlatlines=drawlonlatlines,
                **kwargs,
            )

        except UnsupportedSomercProjection:
            # Define default fall-back projection for Swiss data(EPSG:3035)
            # This will work reasonably well for Europe only.
            t_proj4str = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs"
            geodata = utils.reproject_geodata(geodata, t_proj4str, return_grid="coords")
            extent = (geodata["x1"], geodata["x2"], geodata["y1"], geodata["y2"])
            X, Y = geodata["X_grid"], geodata["Y_grid"]

            ax = basemaps.plot_geography(
                plot_map,
                geodata["projection"],
                extent,
                lw=lw,
                drawlonlatlines=drawlonlatlines,
                **kwargs,
            )
    else:
        ax = plt.gca()

    # reduce number of vectors to plot
    skip = (slice(None, None, step), slice(None, None, step))
    dx = UV[0, :, :][skip]
    dy = UV[1, :, :][skip]
    X = X[skip]
    Y = Y[skip]

    if geodata is None or geodata["yorigin"] == "upper":
        Y = np.flipud(Y)
        dy *= -1

    # plot quiver
    ax.quiver(
        X, Y, dx, dy, angles="xy", zorder=1e6, **quiver_kwargs,
    )
    if geodata is None or axis == "off":
        axes = plt.gca()
        axes.xaxis.set_ticks([])
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticks([])
        axes.yaxis.set_ticklabels([])

    return plt.gca()


def streamplot(
    UV,
    ax=None,
    map=None,
    geodata=None,
    drawlonlatlines=False,
    lw=0.5,
    axis="on",
    streamplot_kwargs=None,
    **kwargs,
):
    """Function to plot a motion field as streamlines.

    .. _`mpl_toolkits.basemap`: https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap

    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html?highlight=subplotspec#matplotlib.gridspec.SubplotSpec

    .. _cartopy: https://scitools.org.uk/cartopy/docs/latest/


    Parameters
    ----------
    UV : array-like
        Array of shape (2, m,n) containing the input motion field.
    ax : axis object
        Optional axis object to use for plotting.
    map : {'basemap', 'cartopy'}, optional
        Optional method for plotting a map: 'basemap' or 'cartopy'.
        The former uses `mpl_toolkits.basemap`_, while the latter uses cartopy_.
    geodata : dictionary
        Optional dictionary containing geographical information about
        the field.
        If geodata is not None, it must contain the following key-value pairs:
        
        .. tabularcolumns:: |p{1.5cm}|L|

        +----------------+----------------------------------------------------+
        |        Key     |                  Value                             |
        +================+====================================================+
        |   projection   | PROJ.4-compatible projection definition            |
        +----------------+----------------------------------------------------+
        |    x1          | x-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y1          | y-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    x2          | x-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y2          | y-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    yorigin     | a string specifying the location of the first      |
        |                | element in the data raster w.r.t. y-axis:          |
        |                | 'upper' = upper border, 'lower' = lower border     |
        +----------------+----------------------------------------------------+
    drawlonlatlines : bool, optional
        If set to True, draw longitude and latitude lines. Applicable if map is
        'basemap' or 'cartopy'.
    lw: float, optional
        Linewidth of the map (administrative boundaries and coastlines).
    axis : {'off','on'}, optional
        Whether to turn off or on the x and y axis.
    streamplot_kwargs : dict, optional
      Optional dictionary containing keyword arguments for the streamplot method.
      See the documentation of matplotlib.pyplot.streamplot.
    
    Other parameters
    ----------------
    Optional parameters are contained in **kwargs. See basemaps.plot_geography.

    Returns
    -------
    out : axis object
        Figure axes. Needed if one wants to add e.g. text inside the plot.

    """
    if map is not None:
        FutureWarning(
            "'map' argument will be renamed to 'plot_map' in 1.4.0. Use 'plot_map' to silence this warning."
        )
        plot_map = map
    else:
        plot_map = kwargs.pop("plot_map", None)

    if plot_map is not None and geodata is None:
        raise ValueError("map!=None but geodata=None")

    if streamplot_kwargs is None:
        streamplot_kwargs = dict()

    # prepare x y coordinates
    reproject = False
    if geodata is not None:
        xmin = min((geodata["x1"], geodata["x2"]))
        xmax = max((geodata["x1"], geodata["x2"]))
        x = np.linspace(xmin, xmax, UV.shape[2])
        xpixelsize = np.abs(x[1] - x[0])
        x += xpixelsize / 2.0

        ymin = min((geodata["y1"], geodata["y2"]))
        ymax = max((geodata["y1"], geodata["y2"]))
        y = np.linspace(ymin, ymax, UV.shape[1])
        ypixelsize = np.abs(y[1] - y[0])
        y += ypixelsize / 2.0

        extent = (geodata["x1"], geodata["x2"], geodata["y1"], geodata["y2"])

        # check geodata and project if different from axes
        if ax is not None and plot_map is None:
            if type(ax).__name__ == "GeoAxesSubplot":
                try:
                    ccrs = utils.proj4_to_cartopy(geodata["projection"])
                except UnsupportedSomercProjection:
                    # Define fall-back projection for Swiss data(EPSG:3035)
                    # This will work reasonably well for Europe only.
                    t_proj4str = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs"
                    reproject = True
            elif type(ax).__name__ == "Basemap":
                utils.proj4_to_basemap(geodata["projection"])

            if reproject:
                geodata = utils.reproject_geodata(
                    geodata, t_proj4str, return_grid="coords"
                )
                extent = (geodata["x1"], geodata["x2"], geodata["y1"], geodata["y2"])
                X, Y = geodata["X_grid"], geodata["Y_grid"]
                x = X[0, :]
                y = Y[:, 0]
    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1])

    # draw basemaps
    if plot_map is not None:
        try:
            ax = basemaps.plot_geography(
                plot_map,
                geodata["projection"],
                extent,
                lw=lw,
                drawlonlatlines=drawlonlatlines,
                **kwargs,
            )
        except UnsupportedSomercProjection:
            # Define default fall-back projection for Swiss data(EPSG:3035)
            # This will work reasonably well for Europe only.
            t_proj4str = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs"
            geodata = utils.reproject_geodata(geodata, t_proj4str, return_grid="coords")
            extent = (geodata["x1"], geodata["x2"], geodata["y1"], geodata["y2"])
            X, Y = geodata["X_grid"], geodata["Y_grid"]
            x = X[0, :]
            y = Y[:, 0]

            ax = basemaps.plot_geography(
                plot_map,
                geodata["projection"],
                extent,
                lw=lw,
                drawlonlatlines=drawlonlatlines,
                **kwargs,
            )
    else:
        ax = plt.gca()

    dx = UV[0, :, :]
    dy = UV[1, :, :]

    if geodata is None or geodata["yorigin"] == "upper":
        y = y[::-1]
        dy *= -1

    # plot streamplot
    ax.streamplot(
        x, y, dx, dy, zorder=1e6, **streamplot_kwargs,
    )

    if geodata is None or axis == "off":
        axes = plt.gca()
        axes.xaxis.set_ticks([])
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticks([])
        axes.yaxis.set_ticklabels([])

    return plt.gca()
