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
    from cartopy.mpl.geoaxes import GeoAxesSubplot

    CARTOPY_IMPORTED = True
except ImportError:
    CARTOPY_IMPORTED = False
try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False

from . import utils

VALID_BASEMAPS = ("cartopy",)


#########################
# Basemap features zorder
# - ocean: 0
# - land: 0
# - lakes: 0
# - rivers_lake_centerlines: 0
# - coastline: 15
# - cultural: 15
# - reefs: 15
# - minor_islands: 15


def plot_geography(
    proj4str,
    extent,
    lw=0.5,
    drawlonlatlines=False,
    drawlonlatlabels=True,
    plot_map="cartopy",
    scale="50m",
    subplot=None,
    **kwargs,
):
    """
    Plot geographical map in a chosen projection using cartopy.

    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html

    Parameters
    ----------
    proj4str: str
        The PROJ.4-compatible projection string.
    extent: scalars (left, right, bottom, top)
        The bounding box in proj4str coordinates.
    lw: float, optional`
        Linewidth of the map (administrative boundaries and coastlines).
    drawlonlatlines: bool, optional
        If set to True, draw longitude and latitude lines.
    drawlonlatlabels: bool, optional
        If set to True, draw longitude and latitude labels.  Valid only if
        'drawlonlatlines' is True.
    plot_map: {'cartopy', None}, optional
        The type of basemap, either 'cartopy' or None. If None, the figure
        axis is returned without any basemap drawn. Default `'cartopy'`.
    scale: {'10m', '50m', '110m'}, optional
        The scale (resolution). Applicable if 'plot_map' is 'cartopy'.
        The available options are '10m', '50m', and '110m'. Default ``'50m'``.
    subplot: tuple of int (nrows, ncols, index) or SubplotSpec_ instance, optional
        The subplot where to plot the basemap.
        By the default, the basemap will replace the current axis.

    Returns
    -------
    ax: fig Axes
        Cartopy axes.
    """

    if len(kwargs) > 0:
        warnings.warn(
            "plot_geography: The following keywords are ignored:\n"
            + str(kwargs)
            + "\nIn version 1.5, passing unsupported arguments will raise an error.",
            DeprecationWarning,
        )

    if plot_map is None:
        return plt.gca()

    if plot_map not in VALID_BASEMAPS:
        raise ValueError(
            f"unsupported plot_map method {plot_map}. Supported basemaps: "
            f"{VALID_BASEMAPS}"
        )

    if plot_map == "cartopy" and not CARTOPY_IMPORTED:
        warnings.warn(
            "The cartopy package is required to plot the geographical map but it is "
            "not installed. Ignoring the geographic information."
        )
        return plt.gca()

    if not PYPROJ_IMPORTED:
        warnings.warn(
            "the pyproj package is required to plot the geographical map "
            "but it is not installed"
        )
        return plt.gca()

    crs = utils.proj4_to_cartopy(proj4str)

    ax = plot_map_cartopy(
        crs,
        extent,
        scale,
        drawlonlatlines=drawlonlatlines,
        drawlonlatlabels=drawlonlatlabels,
        lw=lw,
        subplot=subplot,
    )

    return ax


def plot_map_cartopy(
    crs,
    extent,
    cartopy_scale,
    drawlonlatlines=False,
    drawlonlatlabels=True,
    lw=0.5,
    subplot=None,
):
    """
    Plot coastlines, countries, rivers and meridians/parallels using cartopy.

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
    cartopy_scale: {'10m', '50m', '110m'}
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

    if subplot is None:
        ax = plt.gca()
    else:
        if isinstance(subplot, gridspec.SubplotSpec):
            ax = plt.subplot(subplot, projection=crs)
        else:
            ax = plt.subplot(*subplot, projection=crs)

    if not isinstance(ax, GeoAxesSubplot):
        ax = plt.subplot(ax.get_subplotspec(), projection=crs)
        # cax.clear()
        ax.set_axis_off()

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "ocean",
            scale="50m" if cartopy_scale == "10m" else cartopy_scale,
            edgecolor="none",
            facecolor=np.array([0.59375, 0.71484375, 0.8828125]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            scale=cartopy_scale,
            edgecolor="none",
            facecolor=np.array([0.9375, 0.9375, 0.859375]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "coastline",
            scale=cartopy_scale,
            edgecolor="black",
            facecolor="none",
            linewidth=lw,
        ),
        zorder=15,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "lakes",
            scale=cartopy_scale,
            edgecolor="none",
            facecolor=np.array([0.59375, 0.71484375, 0.8828125]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "rivers_lake_centerlines",
            scale=cartopy_scale,
            edgecolor=np.array([0.59375, 0.71484375, 0.8828125]),
            facecolor="none",
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural",
            "admin_0_boundary_lines_land",
            scale=cartopy_scale,
            edgecolor="black",
            facecolor="none",
            linewidth=lw,
        ),
        zorder=15,
    )
    if cartopy_scale in ["10m", "50m"]:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "reefs",
                scale="10m",
                edgecolor="black",
                facecolor="none",
                linewidth=lw,
            ),
            zorder=15,
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
            zorder=15,
        )

    if drawlonlatlines:
        grid_lines = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=drawlonlatlabels, dms=True
        )
        grid_lines.top_labels = grid_lines.right_labels = False
        grid_lines.y_inline = grid_lines.x_inline = False
        grid_lines.rotate_labels = False

    ax.set_extent(extent, crs)

    return ax
