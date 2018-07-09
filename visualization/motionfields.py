"""Functions to plot precipitation fields."""

import matplotlib.pylab as plt
import matplotlib.colors as colors

import numpy as np

def plot_motion_field_quiver(UV, geodata=None, step=15):
    """Function to plot a precipitation field witha a colorbar. 
    
    Parameters 
    ---------- 
    UV : array-like 
        Array of shape (2, m,n) containing the input motion field.
    geodata : dictionary
    step : int
    
    Returns: 
    ----------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    
    """
    
    # prepare X Y coordinates
    if geodata is not None:
        X,Y = np.meshgrid(np.arange(geodata['x1']/1000,geodata['x2']/1000),
                          np.arange(geodata['y1']/1000,geodata['y2']/1000))
    else:
        X,Y = np.meshgrid(np.arange(UV.shape[2]),
                          np.arange(UV.shape[1]))
    
    # reduce number of vectors to plot
    UV_ = UV[:, 0:UV.shape[1]:step, 0:UV.shape[2]:step]
    X_ = X[0:UV.shape[1]:step, 0:UV.shape[2]:step]
    Y_ = Y[0:UV.shape[1]:step, 0:UV.shape[2]:step]
    
    plt.quiver(X_, Y_, UV_[0,:,:], -UV_[1,:,:], pivot='tip')
    
    axes = plt.gca()
    
    if geodata is None:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
   
    return axes
    
def plot_motion_field_streamplot(UV, geodata=None):
    """Function to plot a precipitation field witha a colorbar. 
    
    Parameters 
    ---------- 
    UV : array-like 
        Array of shape (2, m,n) containing the input motion field.
    geodata : dictionary
        
    step : int
    
    Returns: 
    ----------
    ax : fig axes
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    
    """
    
    # prepare X Y coordinates
    if geodata is not None:
        x = np.arange(geodata['x1']/1000,geodata['x2']/1000)
        y = np.arange(geodata['y1']/1000,geodata['y2']/1000)
    else:
        x = np.arange(UV.shape[2])
        y = np.arange(UV.shape[1])
        
    plt.streamplot(x, y, UV[0,:,:], -UV[1,:,:], density=2, color='black')
    
    axes = plt.gca()
    
    if geodata is None:
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
   
    return axes