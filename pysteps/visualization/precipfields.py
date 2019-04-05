"""
pysteps.visualization.precipfields
==================================

Methods for plotting precipitation fields.

.. autosummary::
    :toctree: ../generated/

    plot_precip_field
    get_colormap
"""

import matplotlib.pylab as plt
from matplotlib import cm, colors, gridspec
import numpy as np
from pysteps.exceptions import UnsupportedSomercProjection

from . import basemaps
from . import utils

def plot_precip_field(R, type="intensity", map=None, geodata=None, units='mm/h',
                      colorscale='pysteps', probthr=None, title=None,
                      colorbar=True, drawlonlatlines=False, basemap_resolution='l',
                      basemap_scale_args=None, cartopy_scale="50m", lw=0.5,
                      cartopy_subplot=(1,1,1), axis="on", cax=None, **kwargs):
    """
    Function to plot a precipitation intensity or probability field with a
    colorbar.

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html

    .. _cartopy: https://scitools.org.uk/cartopy/docs/latest

    .. _mpl_toolkits.basemap: https://matplotlib.org/basemap

    Parameters
    ----------
    R : array-like
        Two-dimensional array containing the input precipitation field or an
        exceedance probability map.
    type : {'intensity', 'depth', 'prob'}, optional
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'depth' = precipitation depth (accumulation) field,
        'prob' = exceedance probability field.
    map : {'basemap', 'cartopy'}, optional
        Optional method for plotting a map: 'basemap' or 'cartopy'. The former
        uses `mpl_toolkits.basemap`_, while the latter uses cartopy_.
    geodata : dictionary, optional
        Optional dictionary containing geographical information about the field.
        If geodata is not None, it must contain the following key-value pairs:
        
        .. tabularcolumns:: |p{1.5cm}|L|

        +-----------------+----------------------------------------------------+
        |        Key      |                  Value                             |
        +=================+====================================================+
        |    projection   | PROJ.4-compatible projection definition            |
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
    units : {'mm/h', 'mm', 'dBZ'}, optional
        Units of the input array. If type is 'prob', this specifies the unit of 
        the intensity threshold.
    colorscale : {'pysteps', 'STEPS-BE', 'BOM-RF3'}, optional
        Which colorscale to use. Applicable if units is 'mm/h', 'mm' or 'dBZ'.
    probthr : float, optional
      Intensity threshold to show in the color bar of the exceedance probability
      map. Required if type is "prob" and colorbar is True.
    title : str, optional
        If not None, print the title on top of the plot.
    colorbar : bool, optional
        If set to True, add a colorbar on the right side of the plot.
    drawlonlatlines : bool, optional
        If set to True, draw longitude and latitude lines. Applicable if map is
        'basemap' or 'cartopy'.
    basemap_resolution : str, optional
        The resolution of the basemap, see the documentation of
        `mpl_toolkits.basemap`_.
        Applicable if map is 'basemap'.
    basemap_scale_args : list, optional
        If not None, a map scale bar is drawn with basemap_scale_args supplied
        to mpl_toolkits.basemap.Basemap.drawmapscale.
    cartopy_scale : {'10m', '50m', '110m'}, optional
        The scale (resolution) of the map. The available options are '10m',
        '50m', and '110m'. Applicable if map is 'cartopy'.
    lw: float, optional
        Linewidth of the map (administrative boundaries and coastlines).
    cartopy_subplot : tuple or SubplotSpec_ instance, optional
        Cartopy subplot. Applicable if map is 'cartopy'.
    axis : {'off','on'}, optional
        Whether to turn off or on the x and y axis.
    cax : Axes_ object, optional
        Axes into which the colorbar will be drawn. If no axes is provided
        the colorbar axes are created next to the plot.

    Returns
    -------
    ax : fig Axes_
        Figure axes. Needed if one wants to add e.g. text inside the plot.

    """
    if type not in ["intensity", "depth", "prob"]:
        raise ValueError("invalid type '%s', must be 'intensity', 'depth' or 'prob'" % type)
    if units not in ["mm/h", "mm", "dBZ"]:
        raise ValueError("invalid units '%s', must be 'mm/h', 'mm' or 'dBZ'" % units)
    if type == "prob" and colorbar and probthr is None:
        raise ValueError("type='prob' but probthr not specified")
    if map is not None and geodata is None:
        raise ValueError("map!=None but geodata=None")
    if len(R.shape) != 2:
        raise ValueError("the input is not two-dimensional array")

    # get colormap and color levels
    cmap, norm, clevs, clevsStr = get_colormap(type, units, colorscale)
    
    # extract extent and origin
    if geodata is not None:
        extent = (geodata['x1'],geodata['x2'], geodata['y1'],geodata['y2'])
        origin = geodata["yorigin"]
    else:
        extent = (0, R.shape[1]-1, 0, R.shape[0]-1)
        origin = "upper"
    
    # plot geography
    if map is not None:
        try:
            ax = basemaps.plot_geography(map, geodata["projection"], 
                            extent, R.shape, drawlonlatlines, basemap_resolution, 
                            basemap_scale_args, cartopy_scale, lw, cartopy_subplot)
            regular_grid = True
        except UnsupportedSomercProjection:        
            # Define default fall-back projection for Swiss data(EPSG:3035)
            # This will work reasonably well for Europe only.
            t_proj4str = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs"
            geodata = utils.reproject_geodata(geodata, t_proj4str, return_grid="quadmesh")
            extent = (geodata['x1'], geodata['x2'], geodata['y1'], geodata['y2'])
            X, Y = geodata["X_grid"], geodata["Y_grid"]
            regular_grid = geodata["regular_grid"]
            
            ax = basemaps.plot_geography(map, geodata["projection"],
                            extent, R.shape, drawlonlatlines, basemap_resolution,
                            basemap_scale_args, cartopy_scale, lw, cartopy_subplot)
    else:
        regular_grid = True
        ax = plt.gca()
    
    # plot rainfield
    if regular_grid:
        im = _plot_field(R, ax, type, units, colorscale, extent=extent, origin=origin)
    else:
        if origin == "upper":
            Y = np.flipud(Y)
        im = _plot_field_pcolormesh(X, Y, R, ax, type, units, colorscale)

    # plot radar domain mask
    mask = np.ones(R.shape)
    mask[~np.isnan(R)] = np.nan # Fully transparent within the radar domain
    ax.imshow(mask, cmap=colors.ListedColormap(['gray']), alpha=0.5,
              zorder=1e6, extent=extent, origin=origin)
              
    # ax.pcolormesh(X, Y, np.flipud(mask), cmap=colors.ListedColormap(['gray']),
                    # alpha=0.5, zorder=1e6)
    # TODO: pcolormesh doesn't work properly with the alpha parameter

    if title is not None:
        plt.title(title)

    # add colorbar
    if colorbar:
        cbar = plt.colorbar(im, ticks=clevs, spacing='uniform', norm=norm,
                extend="max" if type in ["intensity", "depth"] else "neither",
                shrink=0.8, cax=cax)
        if clevsStr != None:
            cbar.ax.set_yticklabels(clevsStr)

        if type == "intensity":
            cbar.ax.set_title(units, fontsize=10)
            cbar.set_label("Precipitation intensity")
        elif type == "depth":
            cbar.ax.set_title(units, fontsize=10)
            cbar.set_label("Precipitation depth")
        else:
            cbar.set_label("P(R > %.1f %s)" % (probthr, units))
    
    if geodata is None or axis == "off":
        axes = plt.gca()
        axes.xaxis.set_ticks([])
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticks([])
        axes.yaxis.set_ticklabels([])

    return plt.gca()

def _plot_field(R, ax, type, units, colorscale, extent, origin=None):
    R = R.copy()

    # Get colormap and color levels
    cmap, norm, clevs, clevsStr = get_colormap(type, units, colorscale)

    # Plot precipitation field
    # transparent where no precipitation or the probability is zero
    if type in ["intensity", "depth"]:
        if units in ['mm/h', 'mm']:
            R[R < 0.1] = np.nan
        elif units == 'dBZ':
            R[R < 10] = np.nan
    else:
        R[R < 1e-3] = np.nan

    vmin,vmax = [None, None] if type in ["intensity", "depth"] else [0.0, 1.0]

    im = ax.imshow(R, cmap=cmap, norm=norm, extent=extent, interpolation='nearest',
                   vmin=vmin, vmax=vmax, origin=origin, zorder=1)

    return im

def _plot_field_pcolormesh(X, Y, R, ax, type, units, colorscale):
    R = R.copy()

    # Get colormap and color levels
    cmap, norm, clevs, clevsStr = get_colormap(type, units, colorscale)

    # Plot precipitation field
    # transparent where no precipitation or the probability is zero
    if type in ["intensity", "depth"]:
        if units in ['mm/h', 'mm']:
            R[R < 0.1] = np.nan
        elif units == 'dBZ':
            R[R < 10] = np.nan
    else:
        R[R < 1e-3] = np.nan

    vmin,vmax = [None, None] if type in ["intensity", "depth"] else [0.0, 1.0]

    im = ax.pcolormesh(X, Y, R, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, zorder=1)

    return im

def get_colormap(type, units='mm/h', colorscale='pysteps'):
    """Function to generate a colormap (cmap) and norm.

    Parameters
    ----------
    type : {'intensity', 'depth', 'prob'}, optional
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'depth' = precipitation depth (accumulation) field,
        'prob' = exceedance probability field.
    units : {'mm/h', 'mm', 'dBZ'}, optional
        Units of the input array. If type is 'prob', this specifies the unit of 
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
    clevsStr: list(str)
        List of precipitation values defining the color limits (with correct
        number of decimals).

    """
    if type in ["intensity", "depth"]:
        # Get list of colors
        color_list,clevs,clevsStr = _get_colorlist(units, colorscale)

        cmap = colors.LinearSegmentedColormap.from_list("cmap", color_list, len(clevs)-1)

        if colorscale == 'BOM-RF3':
            cmap.set_over('black',1)
        if colorscale == 'pysteps':
            cmap.set_over('darkred',1)
        if colorscale == 'STEPS-BE':
            cmap.set_over('black',1)
        norm = colors.BoundaryNorm(clevs, cmap.N)

        return cmap, norm, clevs, clevsStr

    elif type == "prob":
        cmap = plt.get_cmap("OrRd", 10)
        return cmap, colors.Normalize(vmin=0, vmax=1), None, None
    else:
        return cm.jet, colors.Normalize(), None, None

def _get_colorlist(units='mm/h', colorscale='pysteps'):
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

    clevsStr : list(str)
        List of precipitation values defining the color limits
        (with correct number of decimals).

    """

    if colorscale == "BOM-RF3":
        color_list = np.array([(255, 255, 255),  # 0.0
                               (245, 245, 255),  # 0.2
                               (180, 180, 255),  # 0.5
                               (120, 120, 255),  # 1.5
                               (20,  20, 255),   # 2.5
                               (0, 216, 195),    # 4.0
                               (0, 150, 144),    # 6.0
                               (0, 102, 102),    # 10
                               (255, 255,   0),  # 15
                               (255, 200,   0),  # 20
                               (255, 150,   0),  # 30
                               (255, 100,   0),  # 40
                               (255,   0,   0),  # 50
                               (200,   0,   0),  # 60
                               (120,   0,   0),  # 75
                               (40,   0,   0)])  # > 100
        color_list = color_list/255.
        if units == 'mm/h':
            clevs = [0.,0.2, 0.5, 1.5, 2.5, 4, 6, 10, 15, 20, 30, 40, 50, 60, 75,
                    100, 150]
        elif units == "mm":
            clevs = [0.,0.2, 0.5, 1.5, 2.5, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40,
                    45, 50]
        else:
            raise ValueError('Wrong units in get_colorlist: %s' % units)
    elif colorscale == 'pysteps':
        pinkHex = '#%02x%02x%02x' % (232, 215, 242)
        redgreyHex = '#%02x%02x%02x' % (156, 126, 148)
        color_list = [redgreyHex, "#640064","#AF00AF","#DC00DC","#3232C8","#0064FF","#009696","#00C832",
        "#64FF00","#96FF00","#C8FF00","#FFFF00","#FFC800","#FFA000","#FF7D00","#E11900"]
        if units in ['mm/h', 'mm']:
            clevs= [0.08,0.16,0.25,0.40,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100,160]
        elif units == 'dBZ':
            clevs = np.arange(10,65,5)
        else:
            raise ValueError('Wrong units in get_colorlist: %s' % units)
    elif colorscale == 'STEPS-BE':
        color_list = ['cyan','deepskyblue','dodgerblue','blue','chartreuse','limegreen','green','darkgreen','yellow','gold','orange','red','magenta','darkmagenta']
        if units in ['mm/h', 'mm']:
            clevs = [0.1,0.25,0.4,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100]
        elif units == 'dBZ':
            clevs = np.arange(10,65,5)
        else:
            raise ValueError('Wrong units in get_colorlist: %s' % units)

    else:
        print('Invalid colorscale', colorscale)
        raise ValueError("Invalid colorscale " + colorscale)

    # Generate color level strings with correct amount of decimal places
    clevsStr = []
    clevsStr = _dynamic_formatting_floats(clevs, )

    return color_list, clevs, clevsStr

def _dynamic_formatting_floats(floatArray, colorscale='pysteps'):
    """
    Function to format the floats defining the class limits of the colorbar.
    """
    floatArray = np.array(floatArray, dtype=float)

    labels = []
    for label in floatArray:
        if label >= 0.1 and label < 1:
            if colorscale == 'pysteps':
                formatting = ',.2f'
            else:
                formatting = ',.1f'
        elif label >= 0.01 and label < 0.1:
            formatting = ',.2f'
        elif label >= 0.001 and label < 0.01:
            formatting = ',.3f'
        elif label >= 0.0001 and label < 0.001:
            formatting = ',.4f'
        elif label >= 1 and label.is_integer():
            formatting = 'i'
        else:
            formatting = ',.1f'

        if formatting != 'i':
            labels.append(format(label, formatting))
        else:
            labels.append(str(int(label)))

    return labels
