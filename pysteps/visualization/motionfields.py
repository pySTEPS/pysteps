"""Functions to plot precipitation fields."""

import matplotlib.pylab as plt
import matplotlib.colors as colors

import numpy as np

def quiver(UV, metadata=None, step=20):
    """Function to plot a motion field as arrows. 
    
    Parameters 
    ---------- 
    UV : array-like 
        Array of shape (2, m,n) containing the input motion field.
    metadata : dictionary
        Optional dictionary containing geographical information about the field. 
        If metadata is not None, it must contain the following key-value pairs:
        
        x1           x-coordinate of the lower-left corner of the data raster (meters)
        y1           y-coordinate of the lower-left corner of the data raster (meters)
        x2           x-coordinate of the upper-right corner of the data raster (meters)
        y2           y-coordinate of the upper-right corner of the data raster (meters)
        yorigin      a string specifying the location of the first element in
                     the data raster w.r.t. y-axis:
                     'upper' = upper border
                     'lower' = lower border
    step : int
        Optional resample step to reduce the density of the arrows.
        
    Returns: 
    ----------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    
    """
    
    # prepare x y coordinates
    if metadata is not None:
        x = np.arange(metadata['x1']/1000, metadata['x2']/1000)
        y = np.arange(metadata['y1']/1000, metadata['y2']/1000)
    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1])
    
    # reduce number of vectors to plot
    UV_ = UV[:, 0:UV.shape[1]:step, 0:UV.shape[2]:step]
    y_ = y[0:UV.shape[1]:step]
    x_ = x[0:UV.shape[2]:step]
    
    plt.quiver(x_, np.flipud(y_), UV_[0,:,:], -UV_[1,:,:], angles='xy')
        
    axes = plt.gca()
    
    if metadata is None:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
   
    return axes
    
def streamplot(UV, metadata=None):
    """Function to plot a motion field as streamlines. 
    
    Parameters 
    ---------- 
    UV : array-like 
        Array of shape (2, m,n) containing the input motion field.
    metadata : dictionary
        Optional dictionary containing geographical information about the field. 
        If metadata is not None, it must contain the following key-value pairs:
        
        x1           x-coordinate of the lower-left corner of the data raster (meters)
        y1           y-coordinate of the lower-left corner of the data raster (meters)
        x2           x-coordinate of the upper-right corner of the data raster (meters)
        y2           y-coordinate of the upper-right corner of the data raster (meters)
        yorigin      a string specifying the location of the first element in
                     the data raster w.r.t. y-axis:
                     'upper' = upper border
                     'lower' = lower border
    
    Returns: 
    ----------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    
    """
    
    # prepare x y coordinates
    if metadata is not None:
        x = np.arange(metadata['x1']/1000,metadata['x2']/1000)
        y = np.arange(metadata['y1']/1000,metadata['y2']/1000)
    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1],0,-1)
        
    plt.streamplot(x, np.flipud(y), UV[0,:,:], -UV[1,:,:], density=1.5, color='black')
    
    axes = plt.gca()
    
    if metadata is None:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
   
    return axes