# -- coding: utf-8 --
"""
pysteps.visualization.motionfields
==================================

Functions to plot motion fields.

.. autosummary::
    :toctree: ../generated/

    motion_plot
    quiver
    streamplot
"""

from pysteps.visualization import utils

VALID_PLOT_TYPES = ("quiver", "streamplot", "stream")


#################################
# Motion plots zorder definitions
# - quiver: 20
# - stream function: 30


def motion_plot(
    uv_motion_field,
    plot_type="quiver",
    ax=None,
    geodata=None,
    axis="on",
    plot_kwargs=None,
    map_kwargs=None,
    step=20,
):
    """
    Function to plot a motion field as arrows (quiver) or as stream lines (streamplot).

    .. _`quiver_doc`: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.htm

    .. _`streamplot_doc`: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.streamplot.html

    Parameters
    ----------
    uv_motion_field: array-like
        Array of shape (2,m,n) containing the input motion field.
    plot_type: str
        Plot type. "quiver" or "streamplot".
    ax: axis object
        Optional axis object to use for plotting.
    geodata: dictionary or None
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
    axis: {'off','on'}, optional
        Whether to turn off or on the x and y axis.
    step: int
        Optional resample step to control the density of the arrows.
    plot_kwargs: dict, optional
      Optional dictionary containing keyword arguments passed to `quiver()` or
      `streamplot`.
      For more information, see the `quiver_doc`_ and `streamplot_doc`_ matplotlib's
      documentation.
    map_kwargs: dict
        Optional parameters that need to be passed to
        :py:func:`pysteps.visualization.basemaps.plot_geography`.

    Returns
    -------
    out: axis object
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    """
    if plot_type not in VALID_PLOT_TYPES:
        raise ValueError(
            f"Invalid plot_type: {plot_type}.\nSupported: {str(VALID_PLOT_TYPES)}"
        )

    if plot_kwargs is None:
        plot_kwargs = {}
    if map_kwargs is None:
        map_kwargs = {}

    # Assumes the input dimensions are lat/lon
    _, nlat, nlon = uv_motion_field.shape

    x_grid, y_grid, extent, _, _ = utils.get_geogrid(nlat, nlon, geodata=geodata)

    ax = utils.get_basemap_axis(extent, ax=ax, geodata=geodata, map_kwargs=map_kwargs)

    ###########################################################
    # Undersample the number of grid points to use in the plots
    skip = (slice(None, None, step), slice(None, None, step))
    dx = uv_motion_field[0, :, :][skip]
    dy = uv_motion_field[1, :, :][skip].copy()
    x_grid = x_grid[skip]
    y_grid = y_grid[skip]

    # If we have yorigin"="upper" we flip the y axes for the motion field in the y axis.
    if geodata is None or geodata["yorigin"] == "upper":
        dy *= -1

    if plot_type.lower() == "quiver":
        ax.quiver(x_grid, y_grid, dx, dy, angles="xy", zorder=20, **plot_kwargs)
    else:
        ax.streamplot(x_grid, y_grid, dx, dy, zorder=30, **plot_kwargs)

    # Quiver sometimes do not produce tight axes
    ax.autoscale(enable=True, axis="both", tight=True)

    if geodata is None or axis == "off":
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])

    return ax


def quiver(
    uv_motion_field,
    ax=None,
    geodata=None,
    axis="on",
    step=20,
    quiver_kwargs=None,
    map_kwargs=None,
):
    """Function to plot a motion field as arrows.
    Wrapper for :func:`pysteps.visualization.motionfields.motion_plot` passing
    `plot_type="quiver"`.

    .. _`quiver_doc`: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html

    Parameters
    ----------
    uv_motion_field: array-like
        Array of shape (2, m,n) containing the input motion field.
    quiver_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the quiver method.
      This argument is passed to
      See the `quiver_doc`_ matplotlib's documentation.

    Other parameters
    ----------------
    See :py::func:`pysteps.visualization.motionfields.motion_plot`.

    Returns
    -------
    out: axis object0
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    """
    if quiver_kwargs is None:
        quiver_kwargs = dict()

    return motion_plot(
        uv_motion_field,
        plot_type="quiver",
        ax=ax,
        geodata=geodata,
        axis=axis,
        step=step,
        plot_kwargs=quiver_kwargs,
        map_kwargs=map_kwargs,
    )


def streamplot(
    uv_motion_field,
    ax=None,
    geodata=None,
    axis="on",
    streamplot_kwargs=None,
    map_kwargs=None,
    step=20,
):
    """Function to plot a motion field as streamlines.
    Wrapper for :func:`pysteps.visualization.motionfields.motion_plot` passing
    `plot_type="streamplot"`.

    .. _`streamplot_doc`: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.streamplot.html

    Parameters
    ----------
    uv_motion_field: array-like
        Array of shape (2, m,n) containing the input motion field.
    streamplot_kwargs: dict, optional
      Optional dictionary containing keyword arguments for the quiver method.
      This argument is passed to
      See the `streamplot_doc`_ matplotlib's documentation.

    Other parameters
    ----------------
    See :py:func:`pysteps.visualization.motionfields.motion_plot`.

    Returns
    -------
    out: axis object
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    """

    if streamplot_kwargs is None:
        streamplot_kwargs = dict()

    return motion_plot(
        uv_motion_field,
        plot_type="streamplot",
        ax=ax,
        geodata=geodata,
        axis=axis,
        step=step,
        plot_kwargs=streamplot_kwargs,
        map_kwargs=map_kwargs,
    )
