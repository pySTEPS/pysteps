"""Methods for plotting precipitation fields."""

import matplotlib.pylab as plt
from matplotlib import cm, colors
import numpy as np

from pysteps.exceptions import MissingOptionalDependency

try:
    from mpl_toolkits.basemap import Basemap
    basemap_imported = True
except ImportError:
    basemap_imported = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    cartopy_imported = True
except ImportError:
    cartopy_imported = False
try:
    import pyproj
    pyproj_imported = True
except ImportError:
    pyproj_imported = False
from . import utils

def plot_precip_field(R, type="intensity", map=None, geodata=None, units='mm/h',
                      colorscale='MeteoSwiss', probthr=None, title=None,
                      colorbar=True, drawlonlatlines=False, basemap_resolution='l',
                      cartopy_scale="50m"):
    """Function to plot a precipitation intensity or probability field with a
    colorbar.

    Parameters
    ----------
    R : array-like
        Two-dimensional array containing the input precipitation field or an
        exceedance probability map.

    Other parameters
    ----------------
    type : str
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'prob' = exceedance probability field.
    map : str
        Optional method for plotting a map: 'basemap' or 'cartopy'. The former
        uses mpl_toolkits.basemap (https://matplotlib.org/basemap), and the
        latter uses cartopy (https://scitools.org.uk/cartopy/docs/latest).
    geodata : dictionary
        Optional dictionary containing geographical information about the field.
        If geodata is not None, it must contain the following key-value pairs:

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
    units : str
        Units of the input array (mm/h or dBZ). If type is 'prob', this specifies
        the unit of the intensity threshold.
    colorscale : str
        Which colorscale to use (MeteoSwiss, STEPS-BE). Applicable if units is
        'mm/h' or 'dBZ'.
    probthr : float
      Intensity threshold to show in the color bar of the exceedance probability
      map. Required if type is "prob" and colorbar is True.
    title : str
        If not None, print the title on top of the plot.
    colorbar : bool
        If set to True, add a colorbar on the right side of the plot.
    drawlonlatlines : bool
        If set to True, draw longitude and latitude lines. Applicable if map is
        'basemap' or 'cartopy'.
    basemap_resolution : str
        The resolution of the basemap, see the documentation of mpl_toolkits.basemap.
        Applicable if map is 'basemap'.
    cartopy_scale : str
        The scale (resolution) of the map. The available options are '10m',
        '50m', and '110m'. Applicable if map is 'cartopy'.

    Returns
    -------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.

    """
    if type not in ["intensity", "prob"]:
        raise ValueError("invalid type '%s', must be 'intensity' or 'prob'" % type)
    if units not in ["mm/h", "dBZ"]:
        raise ValueError("invalid units '%s', must be 'mm/h' or 'dBZ'" % units)
    if type == "prob" and colorbar and probthr is None:
        raise ValueError("type='prob' but probthr not specified")
    if map is not None and geodata is None:
        raise ValueError("map!=None but geodata=None")
    if map is not None and map not in ["basemap", "cartopy"]:
        raise ValueError("unknown map method %s: must be 'basemap' or 'cartopy'" % map)
    if map == "basemap" and not basemap_imported:
        raise MissingOptionalDependency(
            "map='basemap' option passed to plot_precip_field function"
            "but the basemap package is not installed")
    if map == "cartopy" and not cartopy_imported:
        raise MissingOptionalDependency(
            "map='cartopy' option passed to plot_precip_field function"
            "but the cartopy package is not installed")
    if map is not None and not pyproj_imported:
        raise MissingOptionalDependency(
            "map!=None option passed to plot_precip_field function"
            "but the pyproj package is not installed")
    if len(R.shape) != 2:
        raise ValueError("the input is not two-dimensional array")

    # Get colormap and color levels
    cmap, norm, clevs, clevsStr = get_colormap(type, units, colorscale)

    if map is None:
        # Extract extent for imshow function
        if geodata is not None:
            extent = np.array([geodata['x1'],geodata['x2'],
                               geodata['y1'],geodata['y2']])
            origin = geodata["yorigin"]
        else:
            extent = np.array([0, R.shape[1]-1, 0, R.shape[0]-1])
            origin = "upper"

        # Plot radar domain mask
        mask = np.ones(R.shape)
        mask[~np.isnan(R)] = np.nan # Fully transparent within the radar domain
        plt.imshow(mask, cmap=colors.ListedColormap(['gray']),
                   extent=extent, origin=origin)

        im = _plot_field(R, plt.gca(), type, units, colorscale, geodata, extent=extent)
    else:
        if map == "basemap":
            pr = pyproj.Proj(geodata["projection"])
            ll_lon,ll_lat = pr(geodata["x1"], geodata["y1"], inverse=True)
            ur_lon,ur_lat = pr(geodata["x2"], geodata["y2"], inverse=True)

            bm_params = utils.proj4_to_basemap(geodata["projection"])

            bm_params["llcrnrlon"]  = ll_lon
            bm_params["llcrnrlat"]  = ll_lat
            bm_params["urcrnrlon"]  = ur_lon
            bm_params["urcrnrlat"]  = ur_lat
            bm_params["resolution"] = basemap_resolution

            bm = _plot_map_basemap(bm_params, drawlonlatlines=drawlonlatlines)

            if geodata["yorigin"] == "upper":
                R = np.flipud(R)

            extent = None
            regular_grid = True
        else:
            x1,y1,x2,y2 = geodata["x1"],geodata["y1"],geodata["x2"],geodata["y2"]

            try:
                crs = utils.proj4_to_cartopy(geodata["projection"])
                regular_grid = True
            except:
                # Necessary since cartopy doesn't support the Swiss projection
                # TODO: remove once the somerc projection is supported in cartopy.

                # Define fall back projection
                laeastr = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs" # EPSG:3035
                laea = pyproj.Proj(laeastr)
                crs = utils.proj4_to_cartopy(laeastr)

                # Reproject swiss data on fall back projection
                # this will work reasonably well for Europe only.
                pr = pyproj.Proj(geodata["projection"])
                y_coord = np.linspace(y1, y2, R.shape[0] + 1)
                x_coord = np.linspace(x1, x2, R.shape[1] + 1)
                X, Y = np.meshgrid(x_coord, y_coord)
                x1, y1 = pyproj.transform(pr, laea, x1, y1)
                x2, y2 = pyproj.transform(pr, laea, x2, y2)
                X, Y = pyproj.transform(pr, laea, X.flatten(), Y.flatten())
                X = X.reshape((y_coord.size, x_coord.size))
                Y = Y.reshape((y_coord.size, x_coord.size))

                regular_grid = False

            bm = _plot_map_cartopy(crs, x1, y1, x2, y2, cartopy_scale,
                                   drawlonlatlines=drawlonlatlines)
            extent = (x1, x2, y2, y1)

        if regular_grid:
            im = _plot_field(R, bm, type, units, colorscale, geodata, extent=extent)
        else:
            im = _plot_field_pcolormesh(X, Y, np.flipud(R), bm, type, units, colorscale, geodata)

        # Plot radar domain mask
        mask = np.ones(R.shape)
        mask[~np.isnan(R)] = np.nan # Fully transparent within the radar domain
        bm.imshow(mask, cmap=colors.ListedColormap(['gray']), alpha=0.5,
                  zorder=1e6, extent=extent)
        # bm.pcolormesh(X, Y, np.flipud(mask), cmap=colors.ListedColormap(['gray']),
                        # alpha=0.5, zorder=1e6)
        # TODO: pcolormesh doesn't work properly with the alpha parameter

    if title is not None:
        plt.title(title)

    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(im, ticks=clevs, spacing='uniform', norm=norm,
                            extend="max" if type == "intensity" else "neither",
                            shrink=0.8)
        if clevsStr != None:
            cbar.ax.set_yticklabels(clevsStr)

        if type == "intensity":
            cbar.ax.set_title(units, fontsize=10)
            cbar.set_label("Precipitation intensity")
        else:
            cbar.set_label("P(R > %.1f %s)" % (probthr, units))

    if map is None:
        axes = plt.gca()
        if geodata is None:
            axes.xaxis.set_ticks([])
            axes.xaxis.set_ticklabels([])
            axes.yaxis.set_ticks([])
            axes.yaxis.set_ticklabels([])

    if map is None:
        return axes
    else:
        return bm

def _plot_field(R, ax, type, units, colorscale, geodata, extent):
    R = R.copy()

    # Get colormap and color levels
    cmap, norm, clevs, clevsStr = get_colormap(type, units, colorscale)

    # Extract extent for imshow function
#    if geodata is not None:
#        extent = np.array([geodata['x1']/geodata["xpixelsize"],geodata['x2']/geodata["xpixelsize"],
#                           geodata['y1']/geodata["ypixelsize"],geodata['y2']/geodata["ypixelsize"]])
#    else:
#        extent = np.array([0, R.shape[1], 0, R.shape[0]])

    # Plot precipitation field
    # transparent where no precipitation or the probability is zero
    if type == "intensity":
        if units == 'mm/h':
            R[R < 0.1] = np.nan
        elif units == 'dBZ':
            R[R < 10] = np.nan
    else:
        R[R < 1e-3] = np.nan

    vmin,vmax = [None, None] if type == "intensity" else [0.0, 1.0]

    im = ax.imshow(R, cmap=cmap, norm=norm, extent=extent, interpolation='nearest',
                   vmin=vmin, vmax=vmax, zorder=1)

    return im

def _plot_field_pcolormesh(X, Y, R, ax, type, units, colorscale, geodata):
    R = R.copy()

    # Get colormap and color levels
    cmap, norm, clevs, clevsStr = get_colormap(type, units, colorscale)

    # Plot precipitation field
    # transparent where no precipitation or the probability is zero
    if type == "intensity":
        if units == 'mm/h':
            R[R < 0.1] = np.nan
        elif units == 'dBZ':
            R[R < 10] = np.nan
    else:
        R[R < 1e-3] = np.nan

    vmin,vmax = [None, None] if type == "intensity" else [0.0, 1.0]

    im = plt.pcolormesh(X, Y, R, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, zorder=10)

    return im

def get_colormap(type, units='mm/h', colorscale='MeteoSwiss'):
    """Function to generate a colormap (cmap) and norm.

    Parameters
    ----------
    type : str
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'prob' = exceedance probability field.
    units : str
        Units of the input array (mm/h or dBZ).
    colorscale : str
        Which colorscale to use (MeteoSwiss, STEPS-BE). Applicable if units is
        'mm/h' or 'dBZ'.

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
    if type == "intensity":
        # Get list of colors
        colors_list,clevs,clevsStr = _get_colorlist(units, colorscale)

        cmap = colors.LinearSegmentedColormap.from_list("cmap", colors_list, len(clevs)-1)

        if colorscale == 'MeteoSwiss':
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

def _get_colorlist(units='mm/h', colorscale='MeteoSwiss'):
    """Function to get a list of colors to generate the colormap.

    Parameters
    ----------
    units : str
        Units of the input array (mm/h or dBZ)
    colorscale : str
        Which colorscale to use (MeteoSwiss, STEPS-BE)

    Returns
    -------
    color_list : list(str)
        List of color strings.
    clevs : list(float)
        List of precipitation values defining the color limits.
    clevsStr : list(str)
        List of precipitation values defining the color limits (with correct number of decimals).

    """

    if colorscale == 'STEPS-BE':
        color_list = ['cyan','deepskyblue','dodgerblue','blue','chartreuse','limegreen','green','darkgreen','yellow','gold','orange','red','magenta','darkmagenta']
        if units == 'mm/h':
            clevs = [0.1,0.25,0.4,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100]
        elif units == 'dBZ':
            clevs = np.arange(10,65,5)
        else:
            raise ValueError('Wrong units in get_colorlist')
    elif colorscale == 'MeteoSwiss':
        pinkHex = '#%02x%02x%02x' % (232, 215, 242)
        redgreyHex = '#%02x%02x%02x' % (156, 126, 148)
        color_list = [redgreyHex, "#640064","#AF00AF","#DC00DC","#3232C8","#0064FF","#009696","#00C832",
        "#64FF00","#96FF00","#C8FF00","#FFFF00","#FFC800","#FFA000","#FF7D00","#E11900"]
        if units == 'mm/h':
            clevs= [0.08,0.16,0.25,0.40,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100,160]
        elif units == 'dBZ':
            clevs = np.arange(10,65,5)
        else:
            raise ValueError('Wrong units in get_colorlist')
    else:
        print('Invalid colorscale', colorscale)
        raise ValueError("Invalid colorscale " + colorscale)

    # Generate color level strings with correct amount of decimal places
    clevsStr = []
    clevsStr = _dynamic_formatting_floats(clevs, )

    return color_list, clevs, clevsStr

def _dynamic_formatting_floats(floatArray, colorscale='MeteoSwiss'):
    ''' Function to format the floats defining the class limits of the colorbar.
    '''
    floatArray = np.array(floatArray, dtype=float)

    labels = []
    for label in floatArray:
        if label >= 0.1 and label < 1:
            if colorscale == 'MeteoSwiss':
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

def _plot_map_basemap(bm_params, drawlonlatlines=False, coastlinecolor=(0.3,0.3,0.3),
                  countrycolor=(0.3,0.3,0.3), continentcolor=(0.95,0.95,0.85),
                  lakecolor=(0.65,0.75,0.9), rivercolor=(0.65,0.75,0.9),
                  mapboundarycolor=(0.65,0.75,0.9)):
    bm = Basemap(**bm_params)

    if coastlinecolor is not None:
        bm.drawcoastlines(color=coastlinecolor, linewidth=0.1, zorder=0.1)
    if countrycolor is not None:
        bm.drawcountries(countrycolor, zorder=0.2)
    if rivercolor is not None:
        bm.drawrivers(zorder=0.2, color=rivercolor)
    if continentcolor is not None:
        bm.fillcontinents(color=continentcolor, lake_color=lakecolor, zorder=0)
    if mapboundarycolor is not None:
        bm.drawmapboundary(fill_color=mapboundarycolor, zorder=-1)
    if drawlonlatlines:
        bm.drawmeridians(np.linspace(bm.llcrnrlon, bm.urcrnrlon, 10),
                         color=(0.5,0.5,0.5), linewidth=0.5, labels=[1,0,0,1],
                         fmt="%.1f", fontsize=6)
        bm.drawparallels(np.linspace(bm.llcrnrlat, bm.urcrnrlat, 10),
                         color=(0.5,0.5,0.5), linewidth=0.5, labels=[1,0,0,1],
                         fmt="%.1f", fontsize=6)

    return bm

def _plot_map_cartopy(crs, x1, y1, x2, y2, scale, drawlonlatlines=False):
    ax = plt.axes(projection=crs)

    ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", scale = "50m" if scale is "10m" else scale,
        edgecolor="none", facecolor=np.array([0.59375, 0.71484375, 0.8828125])))
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land",
       scale=scale, edgecolor="none", facecolor=np.array([0.9375, 0.9375, 0.859375])))
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", scale=scale,
        edgecolor="black", facecolor="none", linewidth=0.25))
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", scale=scale,
        edgecolor="none", facecolor=np.array([0.59375, 0.71484375, 0.8828125])))
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines",
        scale=scale, edgecolor=np.array([ 0.59375, 0.71484375, 0.8828125]),
        facecolor="none"))
    ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land",
        scale=scale, edgecolor="black", facecolor="none", linewidth=0.25))

    if drawlonlatlines:
        ax.gridlines(crs=ccrs.PlateCarree())

    ax.set_extent([x1, x2, y1, y2], crs)

    return ax
