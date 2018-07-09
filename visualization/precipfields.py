"""Methods to plot precipitation fields."""

import matplotlib.pylab as plt
import matplotlib.colors as colors

import numpy as np

def plot_precip_field(R, geodata=None, units='mmhr', colorscale='MeteoSwiss', title=None, 
                      colorbar=True):
    """Function to plot a precipitation field witha a colorbar. 
    
    Parameters 
    ---------- 
    R : array-like 
        Array of shape (m,n) containing the input precipitation field.
    geodata : dictionary
        Dictionary containing geographical information about the field.
    units : str
        Units of the input array (mmhr or dBZ)
    colorscale : str 
        Which colorscale to use (MeteoSwiss, STEPS-BE)
    colorbar : bool 
        Whether to add the colorbar or not. 
    
    Returns: 
    ----------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    
    """
    
    if len(R.shape) != 2:
        raise ValueError("the input is not two-dimensional array")
    
    Rplot = R.copy()    
    
    # Get colormap and color levels
    cmap, norm, clevs, clevsStr = get_colormap(units, colorscale)
    
    # Extract extent for imshow function
    if geodata is not None:
        extent = np.array([geodata['x1'],geodata['x2'],geodata['y1'],geodata['y2']])/1000
    else:
        extent = np.array([0, R.shape[1], 0, R.shape[0]])
        
    # Plot radar domain mask
    mask = np.ones(Rplot.shape)
    mask[~np.isnan(Rplot)] = np.nan # Fully transparent within the radar domain
    plt.imshow(mask, cmap=colors.ListedColormap(['gray']), extent=extent)
    
    # Plot precipitation field
    if units == 'mmhr':
        Rplot[Rplot < 0.1] = np.nan # Transparent where no precipitation
    if units == 'dBZ':
        Rplot[Rplot < 10] = np.nan
    im = plt.imshow(Rplot, cmap=cmap, norm=norm, extent=extent, interpolation='nearest')
    plt.title(title)
    
    axes = plt.gca()
    # Add colorbar
    if (colorbar == True):
        cbar = plt.colorbar(im, ticks=clevs, spacing='uniform', norm=norm, extend='max')
        cbar.ax.set_yticklabels(clevsStr)
        cbar.ax.set_title(units, fontsize=12)
        
    if geodata is None:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
    
    return axes
    
def get_colormap(units='mmhr', colorscale='MeteoSwiss'):
    ''' Function to generate a colormap (cmap) and norm
    
    Parameters: 
    ----------  
    units : str
        Units of the input array (mmhr or dBZ)     
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
    
def _get_colorlist(units='mmhr', colorscale='MeteoSwiss'):
    """Function to get a list of colors to generate the colormap. 
    
    Optional kwargs: 
    ----------  
    units : str
        Units of the input array (mmhr or dBZ)     
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
        if units == 'mmhr':
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
        if units == 'mmhr':
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