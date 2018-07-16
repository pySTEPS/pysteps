"""Methods to plot precipitation fields."""

import matplotlib.pylab as plt
import matplotlib.colors as colors
import numpy as np
try:
    from mpl_toolkits.basemap import Basemap
    basemap_imported = True
except ImportError:
    basemap_imported = False
try:
    import pyproj
    pyproj_imported = True
except ImportError:
    pyproj_imported = False
from . import utils

def plot_precip_field(R, with_basemap=False, geodata=None, units='mm/h', 
                      colorscale='MeteoSwiss', title=None, colorbar=True, 
                      basemap_resolution='l', drawlonlatlines=False):
    """Function to plot a precipitation field with a colorbar.
    
    Parameters
    ----------
    R : array-like
        Two-dimensional array containing the input precipitation field.
    with_basemap : bool
        If True, plot a basemap.
    geodata : dictionary
        Optional dictionary containing geographical information about the field. 
        If geodata is not None, it must contain the following key-value pairs:
        
        projection   PROJ.4-compatible projection definition
        x1           x-coordinate of the lower-left corner of the data raster (meters)
        y1           y-coordinate of the lower-left corner of the data raster (meters)
        x2           x-coordinate of the upper-right corner of the data raster (meters)
        y2           y-coordinate of the upper-right corner of the data raster (meters)
        yorigin      a string specifying the location of the first element in
                     the data raster w.r.t. y-axis:
                     'upper' = upper border
                     'lower' = lower border
    units : str
        Units of the input array (mm/h or dBZ)
    colorscale : str 
        Which colorscale to use (MeteoSwiss, STEPS-BE)
    title : str
        If not None, print the title on top of the plot.
    colorbar : bool
        If set to True, add a colorbar on the right side of the plot.
    basemap_resolution : str
        The resolution of the basemap, see the documentation of mpl_toolkits.basemap. 
        Applicable if with_basemap is True.
    drawlonlatlines : bool
        If set to True, draw longitude and latitude lines. Applicable if 
        with_basemap is True.
    
    Returns
    -------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    """
    if with_basemap and not basemap_imported:
        raise Exception("with_basemap=True but basemap not imported")
    if with_basemap and not pyproj_imported:
        raise Exception("with_basemap=True but pyproj not imported")
    
    if len(R.shape) != 2:
        raise ValueError("the input is not two-dimensional array")
    
    # Get colormap and color levels
    cmap, norm, clevs, clevsStr = get_colormap(units, colorscale)
    
    if not with_basemap:
        # Extract extent for imshow function
        if geodata is not None:
            extent = np.array([geodata['x1'],geodata['x2'],
                               geodata['y1'],geodata['y2']]) / 1000
            origin = geodata["yorigin"]
        else:
            extent = np.array([0, R.shape[1], 0, R.shape[0]])
            origin = "upper"
        
        # Plot radar domain mask
        mask = np.ones(R.shape)
        mask[~np.isnan(R)] = np.nan # Fully transparent within the radar domain
        plt.imshow(mask, cmap=colors.ListedColormap(['gray']), 
                   extent=extent, origin=origin)
        
        im = _plot_precip_field(R, plt.gca(), units, colorscale, geodata)
    else:
        bm_params = utils.parse_proj4_string(geodata["projection"], parse_type="basemap")
        
        pr = pyproj.Proj(geodata["projection"])
        ll_lon,ll_lat = pr(geodata["x1"], geodata["y1"], inverse=True)
        ur_lon,ur_lat = pr(geodata["x2"], geodata["y2"], inverse=True)
        
        bm_params["llcrnrlon"]  = ll_lon
        bm_params["llcrnrlat"]  = ll_lat
        bm_params["urcrnrlon"]  = ur_lon
        bm_params["urcrnrlat"]  = ur_lat
        bm_params["resolution"] = basemap_resolution
        
        if geodata["yorigin"] == "upper":
          R = np.flipud(R)
        
        bm = _plot_basemap(bm_params, drawlonlatlines=drawlonlatlines)
        im = _plot_precip_field(R, bm, units, colorscale, geodata)
        
        # Plot radar domain mask
        mask = np.ones(R.shape)
        mask[~np.isnan(R)] = np.nan # Fully transparent within the radar domain
        bm.imshow(mask, cmap=colors.ListedColormap(['gray']), alpha=0.5, zorder=1e6)
    
    if title is not None:
        plt.title(title)
    
    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(im, ticks=clevs, spacing='uniform', norm=norm, extend='max')
        cbar.ax.set_yticklabels(clevsStr)
        cbar.ax.set_title(units, fontsize=12)
    
    if not with_basemap:
        axes = plt.gca()
        if geodata is None:
            axes.xaxis.set_ticks([])
            axes.xaxis.set_ticklabels([])
            axes.yaxis.set_ticks([])
            axes.yaxis.set_ticklabels([])
    
    if not with_basemap:
        return axes
    else:
        return bm

def _plot_precip_field(R, ax, units, colorscale, geodata):
    R = R.copy()
    
    # Get colormap and color levels
    cmap, norm, clevs, clevsStr = get_colormap(units, colorscale)
    
    # Extract extent for imshow function
    if geodata is not None:
        extent = np.array([geodata['x1'],geodata['x2'],geodata['y1'],geodata['y2']]) / 1000
    else:
        extent = np.array([0, R.shape[1], 0, R.shape[0]])
    
    # Plot precipitation field
    if units == 'mm/h':
        R[R < 0.1] = np.nan # Transparent where no precipitation
    if units == 'dBZ':
        R[R < 10] = np.nan
    im = ax.imshow(R, cmap=cmap, norm=norm, extent=extent, interpolation='nearest')
    
    return im

def get_colormap(units='mm/h', colorscale='MeteoSwiss'):
    ''' Function to generate a colormap (cmap) and norm
    
    Parameters: 
    ----------  
    units : str
        Units of the input array (mm/h or dBZ)     
    colorscale : str 
        Which colorscale to use (MeteoSwiss, STEPS-BE)
        
    Returns
    -----------
    cmap : Colormap instance
        colormap
    norm : colors.Normalize object 
        Colors norm
    clevs: list(float)
        List of precipitation values defining the color limits.
    clevsStr: list(str)
        List of precipitation values defining the color limits (with correct number of decimals).
    
    '''
    # Get list of colors
    colors_list,clevs,clevsStr = _get_colorlist(units, colorscale)
    
    cmap = colors.LinearSegmentedColormap.from_list("cmap", colors_list, len(clevs)-1)
    
    if colorscale == 'MeteoSwiss':
        cmap.set_over('darkred',1)
    if colorscale == 'STEPS-BE':
        cmap.set_over('black',1)
    norm = colors.BoundaryNorm(clevs, cmap.N)    
    
    return cmap, norm, clevs, clevsStr
    
def _get_colorlist(units='mm/h', colorscale='MeteoSwiss'):
    """Function to get a list of colors to generate the colormap. 
    
    Optional kwargs: 
    ----------  
    units : str
        Units of the input array (mm/h or dBZ)     
    colorscale : str 
        Which colorscale to use (MeteoSwiss, STEPS-BE)
    
    Returns: 
    ----------
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
            print('Wrong units in get_colorlist')
            sys.exit(1)        
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
            print('Wrong units in get_colorlist')
            sys.exit(1)
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

def _plot_basemap(bm_params, drawlonlatlines=False, coastlinecolor=(1,1,1), 
                  countrycolor=(0.3,0.3,0.3), continentcolor=(1,1,1), 
                  lakecolor=(0.7,0.7,0.7), rivercolor=(0.7,0.7,0.7), 
                  mapboundarycolor=(0.7,0.7,0.7)):
    bm = Basemap(**bm_params)
    
    if coastlinecolor is not None:
        bm.drawcoastlines(color=coastlinecolor, zorder=0.1)
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
