# -*- coding: utf-8 -*-
"""
pysteps.visualization.basemaps
==============================

Methods for plotting geographical maps using Cartopy.

.. autosummary::
    :toctree: ../generated/

    plot_geography
    plot_map_cartopy
"""
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import warnings
from pysteps.exceptions import MissingOptionalDependency


try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    CARTOPY_IMPORTED = True
except ImportError:
    CARTOPY_IMPORTED = False
try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False

from . import utils


VALID_BASEMAPS = ["cartopy"]


def plot_geography(
    proj4str, extent, lw=0.5, drawlonlatlines=False, drawlonlatlabels=True, **kwargs
):
    """
    Plot geographical map using cartopy_ in a chosen projection.

    .. _cartopy: https://scitools.org.uk/cartopy/docs/latest

    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html

    Parameters
    ----------
    proj4str: str
        The PROJ.4-compatible projection string.
    extent: scalars (left, right, bottom, top)
        The bounding box in proj4str coordinates.
    lw: float, optional
        Linewidth of the map (administrative boundaries and coastlines).
    drawlonlatlines: bool, optional
        If set to True, draw longitude and latitude lines.
    drawlonlatlabels: bool, optional
        If set to True, draw longitude and latitude labels.  Valid only if
        'drawlonlatlines' is True.

    Other parameters
    ----------------
    plot_map: {'cartopy', None}, optional
        The type of basemap, either 'cartopy_' or None. If None, the figure
        axis is returned without any basemap drawn. Default ``'cartopy'``.
    scale: {'10m', '50m', '110m'}, optional
        The scale (resolution). Applicable if 'plot_map' is 'cartopy'.
        The available options are '10m', '50m', and '110m'. Default ``'50m'``.
    subplot: tuple of int (nrows, ncols, index) or SubplotSpec_ instance, optional
        The subplot where to plot the basemap.
        By the default, the basemap will replace the current axis.

    Returns
    -------
    ax: fig Axes_
        Cartopy axes.
    """

    plot_map = kwargs.get("plot_map", "cartopy")

    if plot_map is None:
        return plt.gca()

    if plot_map not in VALID_BASEMAPS:
        raise ValueError(
            f"unsupported plot_map method {plot_map}. Supported basemaps: "
            f"{VALID_BASEMAPS}"
        )

    if plot_map == "cartopy" and not CARTOPY_IMPORTED:
        raise MissingOptionalDependency(
            "the cartopy package is required to plot the geographical map "
            " but it is not installed"
        )

    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "the pyproj package is required to plot the geographical map"
            "but it is not installed"
        )

    # if plot_map == "cartopy": # not really an option for the moment
    cartopy_scale = kwargs.get("scale", "50m")
    cartopy_subplot = kwargs.get("subplot", None)
    crs = utils.proj4_to_cartopy(proj4str)

    # Replace current axis
    if cartopy_subplot is None:
        cax = plt.gca()
        cartopy_subplot = cax.get_subplotspec()
        cax.clear()
        cax.set_axis_off()

    ax = plot_map_cartopy(
        crs,
        extent,
        cartopy_scale,
        drawlonlatlines=drawlonlatlines,
        drawlonlatlabels=drawlonlatlabels,
        lw=lw,
        subplot=cartopy_subplot,
    )

    return ax


def plot_map_cartopy(
    crs,
    extent,
    scale,
    drawlonlatlines=False,
    drawlonlatlabels=True,
    lw=0.5,
    subplot=None,
):
    """
    Plot coastlines, countries, rivers and meridians/parallels using cartopy_.

    .. _cartopy: https://scitools.org.uk/cartopy/docs/latest

    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html

    Parameters
    ----------
    crs: object
        Instance of a crs class defined in cartopy.crs.
        It can be created using utils.proj4_to_cartopy.
    extent: scalars (left, right, bottom, top)
        The coordinates of the bounding box.
    drawlonlatlines: bool
        Whether to plot longitudes and latitudes.
    drawlonlatlabels: bool, optional
        If set to True, draw longitude and latitude labels. Valid only if
        'drawlonlatlines' is True.
    scale: {'10m', '50m', '110m'}
        The scale (resolution) of the map. The available options are '10m',
        '50m', and '110m'.
    lw: float
        Line width.
    subplot: tuple of int (nrows, ncols, index) or SubplotSpec_ instance, optional
        The subplot where to place the basemap.
        By the default, the basemap will replace the current axis.

    Returns
    -------
    ax: axes
        Cartopy axes. Compatible with matplotlib.
    """
    if not CARTOPY_IMPORTED:
        raise MissingOptionalDependency(
            "the cartopy package is required to plot the geographical map"
            " but it is not installed"
        )

    # Replace current axis
    if subplot is None:
        cax = plt.gca()
        subplot = cax.get_subplotspec()
        cax.clear()
        cax.set_axis_off()

    if isinstance(subplot, gridspec.SubplotSpec):
        ax = plt.subplot(subplot, projection=crs)
    else:
        ax = plt.subplot(*subplot, projection=crs)

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "ocean",
            scale="50m" if scale == "10m" else scale,
            edgecolor="none",
            facecolor=np.array([0.59375, 0.71484375, 0.8828125]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            scale=scale,
            edgecolor="none",
            facecolor=np.array([0.9375, 0.9375, 0.859375]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "coastline",
            scale=scale,
            edgecolor="black",
            facecolor="none",
            linewidth=lw,
        ),
        zorder=2,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "lakes",
            scale=scale,
            edgecolor="none",
            facecolor=np.array([0.59375, 0.71484375, 0.8828125]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "rivers_lake_centerlines",
            scale=scale,
            edgecolor=np.array([0.59375, 0.71484375, 0.8828125]),
            facecolor="none",
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural",
            "admin_0_boundary_lines_land",
            scale=scale,
            edgecolor="black",
            facecolor="none",
            linewidth=lw,
        ),
        zorder=2,
    )
    if scale in ["10m", "50m"]:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "reefs",
                scale="10m",
                edgecolor="black",
                facecolor="none",
                linewidth=lw,
            ),
            zorder=2,
        )
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "minor_islands",
                scale="10m",
                edgecolor="black",
                facecolor="none",
                linewidth=lw,
            ),
            zorder=2,
        )

    if drawlonlatlines:
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=drawlonlatlabels, dms=True
        )
        gl.top_labels = gl.right_labels = False
        gl.y_inline = gl.x_inline = False
        gl.rotate_labels = False

    ax.set_extent(extent, crs)

    return ax
