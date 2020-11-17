#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:09:44 2020

@author: mfeldman
"""
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
mpl.use('module://ipykernel.pylab.backend_inline')
import matplotlib.pyplot as plt

def plot_track(track_list, imtitle, savepath, imname, xsize, ysize, poix=None, poiy=None):
    """
    
    Parameters
    ----------
    track_list : list
        List of tracks provided by DATing.
    imtitle : string
        Text written in image title.
    savepath : string
        path to saving location.
    imname : string
        name to save image under, include file ending.
    xsize : int
        size of domain in x-direction.
    ysize : int
        size of domain in y-direction.
    poix : array of int, optional
        X-location of points of interest to plot with tracks (e.g. radar locations).
        The default is None.
    poiy : array of int, optional
        Y-location of points of interest to plot with tracks (e.g. radar locations).
        The default is None.

    Returns
    -------
    None.

    """
    fig=plt.figure(figsize=(14.2,12.8))
    plt.ylim(0, ysize)
    plt.xlim(0, xsize)
    if all(poix != None) and all(poiy != None):
        p1=plt.scatter(poix, poiy,s=None,c='black')
    color=iter(plt.cm.spring(np.linspace(0,1,len(track_list))))
    for track in track_list:
        p2=plt.plot(track.max_x, track.max_y, c=next(color))
    plt.title(imtitle)
    namefig=savepath + imname
    fig.savefig(namefig)
    plt.show()
    plt.close(fig=fig)


def plot_cart_contour(input_image, contours, imtitle, savepath, imname, poix=None, poiy=None):
    """
    Plots input image with identified cell contours. Optionally points of interest added.

    Parameters
    ----------
    input_image : array-like
        2-D array of input array used for cell detection.
    contours : list or dataframe-element
        list of identified cell contours.
    imtitle : string
        caption of image.
    savepath : string
        path to saving location.
    imname : string
        name of saved image.
    poix : array-like, optional
        1-D array of x-locations of points of interest. The default is None.
    poiy : array-like, optional
        1-D array, same length as poix, y-locations of points of interest. The default is None.

    Returns
    -------
    None.

    """
    fig=plt.figure(figsize=(10,7.5))
    cmap=plt.cm.jet
    cmap.set_bad(color='gray')
    p0=plt.pcolormesh(input_image, cmap=cmap)
    plt.colorbar(p0)
    p1=plt.scatter(poix,poiy,s=None,c='black')
    contours=list(contours)
    for contour in contours:
        for c in contour:
            p1=plt.plot(c[:,1], c[:,0], color='black')
        # else: p1=plt.plot(contour[:,1], contour[:,0], color='black')
    plt.title(imtitle)
    namefig=savepath + imname
    plt.show()
    fig.savefig(namefig)
    plt.close(fig=fig)
    