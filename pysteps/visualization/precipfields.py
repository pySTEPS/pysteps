# -- coding: utf-8 --
"""
pysteps.visualization.precipfields
==================================

Methods for plotting precipitation fields.

.. autosummary::
    :toctree: ../generated/

    plot_precip_field
    get_colormap
"""
import copy
import warnings

import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm, colors

from pysteps.visualization.utils import get_geogrid, get_basemap_axis

PRECIP_VALID_TYPES = ("intensity", "depth", "prob")
PRECIP_VALID_UNITS = ("mm/h", "mm", "dBZ")


############################
# precipitation plots zorder
# - precipitation: 10


def plot_precip_field(
    precip,
    ptype="intensity",
    ax=None,
    geodata=None,
    units="mm/h",
    bbox=None,
    colorscale="pysteps",
    probthr=None,
    title=None,
    colorbar=True,
    axis="on",
    cax=None,
    map_kwargs=None,
):
    """
    Function to plot a precipitation intensity or probability field with a
    colorbar.

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html

    Parameters
    ----------
    precip: array-like
        Two-dimensional array containing the input precipitation field or an
        exceedance probability map.
    ptype: {'intensity', 'depth', 'prob'}, optional
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'depth' = precipitation depth (accumulation) field,
        'prob' = exceedance probability field.
    geodata: dictionary or None, optional
        Optional dictionary containing geographical information about
        the field. Required is map is not None.

        If geodata is not None, it must contain the following key-value pairs:

        .. tabularcolumns:: |p{1.5cm}|L|

        +-----------------+---------------------------------------------------+
        |        Key      |                  Value                            |
        +=================+===================================================+
        |    projection   | PROJ.4-compatible projection definition           |
        +-----------------+---------------------------------------------------+
        |    x1           | x-coordinate of the lower-left corner of the data |
        |                 | raster                                            |
        +-----------------+---------------------------------------------------+
        |    y1           | y-coordinate of the lower-left corner of the data |
        |                 | raster                                            |
        +-----------------+---------------------------------------------------+
        |    x2           | x-coordinate of the upper-right corner of the     |
        |                 | data raster                                       |
        +-----------------+---------------------------------------------------+
        |    y2           | y-coordinate of the upper-right corner of the     |
        |                 | data raster                                       |
        +-----------------+---------------------------------------------------+
        |    yorigin      | a string specifying the location of the first     |
        |                 | element in the data raster w.r.t. y-axis:         |
        |                 | 'upper' = upper border, 'lower' = lower border    |
        +-----------------+---------------------------------------------------+
    units : {'mm/h', 'mm', 'dBZ'}, optional
        Units of the input array. If ptype is 'prob', this specifies the unit of
        the intensity threshold.
    bbox : tuple, optional
        Four-element tuple specifying the coordinates of the bounding box. Use
        this for plotting a subdomain inside the input grid. The coordinates are
        of the form (lower left x, lower left y ,upper right x, upper right y).
        If 'geodata' is not None, the bbox is in map coordinates, otherwise
        it represents image pixels.
    colorscale : {'pysteps', 'STEPS-BE', 'BOM-RF3'}, optional
        Which colorscale to use. Applicable if units is 'mm/h', 'mm' or 'dBZ'.
    probthr : float, optional
        Intensity threshold to show in the color bar of the exceedance
        probability map.
        Required if ptype is "prob" and colorbar is True.
    title : str, optional
        If not None, print the title on top of the plot.
    colorbar : bool, optional
        If set to True, add a colorbar on the right side of the plot.
    axis : {'off','on'}, optional
        Whether to turn off or on the x and y axis.
    cax : Axes_ object, optional
        Axes into which the colorbar will be drawn. If no axes is provided
        the colorbar axes are created next to the plot.

    Other parameters
    ----------------
    map_kwargs: dict
        Optional parameters that need to be passed to
        :py:func:`pysteps.visualization.basemaps.plot_geography`.

    Returns
    -------
    ax : fig Axes_
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    """

    if map_kwargs is None:
        map_kwargs = {}

    if ptype not in PRECIP_VALID_TYPES:
        raise ValueError(
            f"Invalid precipitation type '{ptype}'."
            f"Supported: {str(PRECIP_VALID_TYPES)}"
        )

    if units not in PRECIP_VALID_UNITS:
        raise ValueError(
            f"Invalid precipitation units '{units}."
            f"Supported: {str(PRECIP_VALID_UNITS)}"
        )

    if ptype == "prob" and colorbar and probthr is None:
        raise ValueError("ptype='prob' but probthr not specified")

    if len(precip.shape) != 2:
        raise ValueError("The input is not two-dimensional array")

    # Assumes the input dimensions are lat/lon
    nlat, nlon = precip.shape

    x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
        nlat, nlon, geodata=geodata
    )

    ax = get_basemap_axis(extent, ax=ax, geodata=geodata, map_kwargs=map_kwargs)

    precip = np.ma.masked_invalid(precip)
    # plot rainfield
    if regular_grid:
        im = _plot_field(precip, ax, ptype, units, colorscale, extent, origin=origin)
    else:
        im = _plot_field(
            precip, ax, ptype, units, colorscale, extent, x_grid=x_grid, y_grid=y_grid
        )

    plt.title(title)

    # add colorbar
    if colorbar:
        # get colormap and color levels
        _, _, clevs, clevs_str = get_colormap(ptype, units, colorscale)
        if ptype in ["intensity", "depth"]:
            extend = "max"
        else:
            extend = "neither"
        cbar = plt.colorbar(
            im, ticks=clevs, spacing="uniform", extend=extend, shrink=0.8, cax=cax
        )
        if clevs_str is not None:
            cbar.ax.set_yticklabels(clevs_str)

        if ptype == "intensity":
            cbar.set_label(f"Precipitation intensity [{units}]")
        elif ptype == "depth":
            cbar.set_label(f"Precipitation depth [{units}]")
        else:
            cbar.set_label(f"P(R > {probthr:.1f} {units})")

    if geodata is None or axis == "off":
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])

    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    return ax


def _plot_field(
    precip, ax, ptype, units, colorscale, extent, origin=None, x_grid=None, y_grid=None
):
    precip = precip.copy()

    # Get colormap and color levels
    cmap, norm, _, _ = get_colormap(ptype, units, colorscale)

    if (x_grid is None) or (y_grid is None):
        im = ax.imshow(
            precip,
            cmap=cmap,
            norm=norm,
            extent=extent,
            interpolation="nearest",
            origin=origin,
            zorder=10,
        )
    else:
        im = ax.pcolormesh(
            x_grid,
            y_grid,
            precip,
            cmap=cmap,
            norm=norm,
            zorder=10,
        )

    return im


def get_colormap(ptype, units="mm/h", colorscale="pysteps"):
    """
    Function to generate a colormap (cmap) and norm.

    Parameters
    ----------
    ptype : {'intensity', 'depth', 'prob'}, optional
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'depth' = precipitation depth (accumulation) field,
        'prob' = exceedance probability field.
    units : {'mm/h', 'mm', 'dBZ'}, optional
        Units of the input array. If ptype is 'prob', this specifies the unit of
        the intensity threshold.
    colorscale : {'pysteps', 'STEPS-BE', 'BOM-RF3'}, optional
        Which colorscale to use. Applicable if units is 'mm/h', 'mm' or 'dBZ'.

    Returns
    -------
    cmap : Colormap instance
        colormap
    norm : colors.Normalize object
        Colors norm
    clevs: list(float)
        List of precipitation values defining the color limits.
    clevs_str: list(str)
        List of precipitation values defining the color limits (with correct
        number of decimals).
    """
    if ptype in ["intensity", "depth"]:
        # Get list of colors
        color_list, clevs, clevs_str = _get_colorlist(units, colorscale)

        cmap = colors.LinearSegmentedColormap.from_list(
            "cmap", color_list, len(clevs) - 1
        )

        if colorscale == "BOM-RF3":
            cmap.set_over("black", 1)
        if colorscale == "pysteps":
            cmap.set_over("darkred", 1)
        if colorscale == "STEPS-BE":
            cmap.set_over("black", 1)
        norm = colors.BoundaryNorm(clevs, cmap.N)

        cmap.set_bad("gray", alpha=0.5)
        cmap.set_under("none")

        return cmap, norm, clevs, clevs_str

    if ptype == "prob":
        cmap = copy.copy(plt.get_cmap("OrRd", 10))
        cmap.set_bad("gray", alpha=0.5)
        cmap.set_under("none")
        clevs = np.linspace(0, 1, 11)
        clevs[0] = 1e-3  # to set zeros to transparent
        norm = colors.BoundaryNorm(clevs, cmap.N)
        clevs_str = [f"{clev:.1f}" for clev in clevs]
        return cmap, norm, clevs, clevs_str

    return cm.get_cmap("jet"), colors.Normalize(), None, None


def _get_colorlist(units="mm/h", colorscale="pysteps"):
    """
    Function to get a list of colors to generate the colormap.

    Parameters
    ----------
    units : str
        Units of the input array (mm/h, mm or dBZ)
    colorscale : str
        Which colorscale to use (BOM-RF3, pysteps, STEPS-BE)

    Returns
    -------
    color_list : list(str)
        List of color strings.

    clevs : list(float)
        List of precipitation values defining the color limits.

    clevs_str : list(str)
        List of precipitation values defining the color limits
        (with correct number of decimals).
    """

    if colorscale == "BOM-RF3":
        color_list = np.array(
            [
                (255, 255, 255),  # 0.0
                (245, 245, 255),  # 0.2
                (180, 180, 255),  # 0.5
                (120, 120, 255),  # 1.5
                (20, 20, 255),  # 2.5
                (0, 216, 195),  # 4.0
                (0, 150, 144),  # 6.0
                (0, 102, 102),  # 10
                (255, 255, 0),  # 15
                (255, 200, 0),  # 20
                (255, 150, 0),  # 30
                (255, 100, 0),  # 40
                (255, 0, 0),  # 50
                (200, 0, 0),  # 60
                (120, 0, 0),  # 75
                (40, 0, 0),
            ]
        )  # > 100
        color_list = color_list / 255.0
        if units == "mm/h":
            clevs = [
                0.0,
                0.2,
                0.5,
                1.5,
                2.5,
                4,
                6,
                10,
                15,
                20,
                30,
                40,
                50,
                60,
                75,
                100,
                150,
            ]
        elif units == "mm":
            clevs = [
                0.0,
                0.2,
                0.5,
                1.5,
                2.5,
                4,
                5,
                7,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
            ]
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)
    elif colorscale == "pysteps":
        # pinkHex = '#%02x%02x%02x' % (232, 215, 242)
        redgrey_hex = "#%02x%02x%02x" % (156, 126, 148)
        color_list = [
            redgrey_hex,
            "#640064",
            "#AF00AF",
            "#DC00DC",
            "#3232C8",
            "#0064FF",
            "#009696",
            "#00C832",
            "#64FF00",
            "#96FF00",
            "#C8FF00",
            "#FFFF00",
            "#FFC800",
            "#FFA000",
            "#FF7D00",
            "#E11900",
        ]
        if units in ["mm/h", "mm"]:
            clevs = [
                0.08,
                0.16,
                0.25,
                0.40,
                0.63,
                1,
                1.6,
                2.5,
                4,
                6.3,
                10,
                16,
                25,
                40,
                63,
                100,
                160,
            ]
        elif units == "dBZ":
            clevs = np.arange(10, 65, 5)
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)
    elif colorscale == "STEPS-BE":
        color_list = [
            "cyan",
            "deepskyblue",
            "dodgerblue",
            "blue",
            "chartreuse",
            "limegreen",
            "green",
            "darkgreen",
            "yellow",
            "gold",
            "orange",
            "red",
            "magenta",
            "darkmagenta",
        ]
        if units in ["mm/h", "mm"]:
            clevs = [0.1, 0.25, 0.4, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100]
        elif units == "dBZ":
            clevs = np.arange(10, 65, 5)
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)

    else:
        print("Invalid colorscale", colorscale)
        raise ValueError("Invalid colorscale " + colorscale)

    # Generate color level strings with correct amount of decimal places
    clevs_str = _dynamic_formatting_floats(clevs)

    return color_list, clevs, clevs_str


def _dynamic_formatting_floats(float_array, colorscale="pysteps"):
    """Function to format the floats defining the class limits of the colorbar."""
    float_array = np.array(float_array, dtype=float)

    labels = []
    for label in float_array:
        if 0.1 <= label < 1:
            if colorscale == "pysteps":
                formatting = ",.2f"
            else:
                formatting = ",.1f"
        elif 0.01 <= label < 0.1:
            formatting = ",.2f"
        elif 0.001 <= label < 0.01:
            formatting = ",.3f"
        elif 0.0001 <= label < 0.001:
            formatting = ",.4f"
        elif label >= 1 and label.is_integer():
            formatting = "i"
        else:
            formatting = ",.1f"

        if formatting != "i":
            labels.append(format(label, formatting))
        else:
            labels.append(str(int(label)))

    return labels
