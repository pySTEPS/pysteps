"""Functions to plot precipitation fields."""

import matplotlib.pylab as plt
import matplotlib.colors as colors

import numpy as np

def quiver(UV, geodata=None, **kwargs):
    """Function to plot a motion field as arrows. 
    
    Parameters 
    ---------- 
    UV : array-like 
        Array of shape (2, m,n) containing the input motion field.
    geodata : dictionary
        Optional dictionary containing geographical information about the field. 
        If geodata is not None, it must contain the following key-value pairs:
        
        x1           x-coordinate of the lower-left corner of the data raster (meters)
        y1           y-coordinate of the lower-left corner of the data raster (meters)
        x2           x-coordinate of the upper-right corner of the data raster (meters)
        y2           y-coordinate of the upper-right corner of the data raster (meters)
        yorigin      a string specifying the location of the first element in
                     the data raster w.r.t. y-axis:
                     'upper' = upper border
                     'lower' = lower border
                     
    Optional kwargs
    ---------------              
    step : int
        Optional resample step to control the density of the arrows.
        Default : 20
    color : string
        Optional color of the arrows. This is a synonym for the PolyCollection 
        facecolor kwarg in matplotlib.collections.
        Default : black
        
    Returns: 
    ----------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    
    """
    
    # defaults
    step        = kwargs.get("step", 20)
    color       = kwargs.get("color", "black")
    
    # prepare x y coordinates
    if geodata is not None:
        x = np.linspace(geodata['x1']/geodata["xpixelsize"], geodata['x2']/geodata["xpixelsize"], UV.shape[2])
        y = np.linspace(geodata['y1']/geodata["ypixelsize"], geodata['y2']/geodata["ypixelsize"], UV.shape[1])
    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1])
    
    # reduce number of vectors to plot
    UV_ = UV[:, 0:UV.shape[1]:step, 0:UV.shape[2]:step]
    y_ = y[0:UV.shape[1]:step]
    x_ = x[0:UV.shape[2]:step]
    
    plt.quiver(x_, np.flipud(y_), UV_[0,:,:], -UV_[1,:,:], angles='xy',
               color=color)
        
    axes = plt.gca()
    
    if geodata is None:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
   
    return axes
    
def streamplot(UV, geodata=None, **kwargs):
    """Function to plot a motion field as streamlines. 
    
    Parameters 
    ---------- 
    UV : array-like 
        Array of shape (2, m,n) containing the input motion field.
    geodata : dictionary
        Optional dictionary containing geographical information about the field. 
        If geodata is not None, it must contain the following key-value pairs:
        
        x1           x-coordinate of the lower-left corner of the data raster (meters)
        y1           y-coordinate of the lower-left corner of the data raster (meters)
        x2           x-coordinate of the upper-right corner of the data raster (meters)
        y2           y-coordinate of the upper-right corner of the data raster (meters)
        yorigin      a string specifying the location of the first element in
                     the data raster w.r.t. y-axis:
                     'upper' = upper border
                     'lower' = lower border
                     
    Optional kwargs
    ---------------    
    density : float
        Controls the closeness of streamlines.
        Default : 1.5
    color : string
        Optional streamline color. This is a synonym for the PolyCollection 
        facecolor kwarg in matplotlib.collections.
        Default : black
    
    Returns: 
    ----------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    
    """
    
    # defaults
    density     = kwargs.get("density", 1.5)
    color       = kwargs.get("color", "black")
    
    # prepare x y coordinates
    if geodata is not None:
        x = np.linspace(geodata['x1']/geodata["xpixelsize"], geodata['x2']/geodata["xpixelsize"], UV.shape[2])
        y = np.linspace(geodata['y1']/geodata["ypixelsize"], geodata['y2']/geodata["ypixelsize"], UV.shape[1])
    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1],0,-1)
    
    plt.streamplot(x, np.flipud(y), UV[0,:,:], -UV[1,:,:], density=density, 
                   color=color)
    
    axes = plt.gca()
    
    if geodata is None:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
   
    return axes
