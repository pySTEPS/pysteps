#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pysteps.feature.tstorm
======================

Thunderstorm cell detection module, part of Thunderstorm Detection and Tracking (DATing)

Created on Wed Nov  4 11:09:12 2020

@author: mfeldman

.. autosummary::
    :toctree: ../generated/

    detection
    belonging
    longdistance
    get_profile
"""

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from pysteps.exceptions import MissingOptionalDependency

try:
    import skimage

    SKIMAGE_IMPORTED = True
except ImportError:
    SKIMAGE_IMPORTED = False
if SKIMAGE_IMPORTED:
    import skimage.draw as skid
    import skimage.feature as skif
    import skimage.measure as skime
    import skimage.morphology as skim
    import skimage.segmentation as skis
try:
    import pandas as pd

    PANDAS_IMPORTED = True
except ImportError:
    PANDAS_IMPORTED = False


def detection(
    input_image,
    minref=35,
    maxref=48,
    mindiff=6,
    minsize=50,
    minmax=41,
    mindis=10,
    dyn_thresh=False,
    time="000000000",
):
    """
    This function detects thunderstorms using a multi-threshold approach. It is
    recommended to use a 2-D Cartesian maximum reflectivity composite, however the
    function will process any 2-D array.
    The thunderstorm cell detection requires both scikit-image and pandas.

    Parameters
    ----------
    input_image : array-like
        Array of shape (m,n) containing input image, usually maximum reflectivity in
        dBZ with a resolution of 1 km. Nan values are ignored.
    minref : float, optional
        Lower threshold for object detection. Lower values will be set to NaN.
        The default is 35 dBZ.
    maxref : float, optional
        Upper threshold for object detection. Higher values will be set to this value.
        The default is 48 dBZ.
    mindiff : float, optional
        Minimal difference between two identified maxima within same area to split area
        into two objects. The default is 6 dBZ.
    minsize : float, optional
        Minimal area for possible detected object. The default is 50 pixels.
    minmax : float, optional
        Minimum value of maximum in identified objects. Objects with a maximum lower
        than this will be discarded. The default is 41 dBZ.
    mindis : float, optional
        Minimum distance between two maxima of identified objects. Objects with a
        smaller distance will be merged. The default is 10 km.
    dyn_thresh: binary, optional
        Set to True to activate dynamic lower threshold. Restricts contours to more
        meaningful area. The default is False.
    time : string, optional
        Date and time as string. Used to label time in the resulting dataframe.
        The default is '000000000'.

    Returns
    -------
    cells_id : pandas dataframe
        Pandas dataframe containing all detected cells and their respective properties
        corresponding to the input image.
    labels : array-like
        Array of shape (m,n), grid of labelled cells.
    """
    if not SKIMAGE_IMPORTED:
        raise MissingOptionalDependency(
            "skimage is required for thunderstorm DATing " "but it is not installed"
        )
    if not PANDAS_IMPORTED:
        raise MissingOptionalDependency(
            "pandas is required for thunderstorm DATing " "but it is not installed"
        )
    filt_image = np.zeros(input_image.shape)
    filt_image[input_image > minref] = input_image[input_image > minref]
    filt_image[input_image > maxref] = maxref
    max_image = np.zeros(filt_image.shape)
    max_image[filt_image == maxref] = 1
    labels, n_groups = ndi.label(max_image)
    for n in range(1, n_groups + 1):
        indx, indy = np.where(labels == n)
        if len(indx) > 3:
            max_image[indx[0], indy[0]] = 2
    filt_image[max_image == 2] = maxref + 1
    binary = np.zeros(filt_image.shape)
    binary[filt_image > 0] = 1
    labels, n_groups = ndi.label(binary)
    for n in range(1, n_groups + 1):
        ind = np.where(labels == n)
        size = len(ind[0])
        maxval = np.nanmax(input_image[ind])
        if size < minsize:
            binary[labels == n] = 0
            labels[labels == n] = 0
        if maxval < minmax:
            binary[labels == n] = 0
            labels[labels == n] = 0
    filt_image = filt_image * binary
    if mindis % 2 == 0:
        elem = mindis - 1
    else:
        elem = mindis
    struct = np.ones([elem, elem])
    if np.nanmax(filt_image.flatten()) < minref:
        maxima = np.zeros(filt_image.shape)
    else:
        maxima = skim.h_maxima(filt_image, h=mindiff, selem=struct)
    loc_max = np.where(maxima > 0)

    loc_max = longdistance(loc_max, mindis)
    i_cell = labels[loc_max]
    n_cell = np.unique(labels)[1:]
    for n in n_cell:
        if n not in i_cell:
            binary[labels == n] = 0
            labels[labels == n] = 0

    maxima_dis = np.zeros(maxima.shape)
    maxima_dis[loc_max] = 1

    areas, lines = belonging(input_image, np.nanmin(input_image.flatten()), maxima_dis)

    if dyn_thresh: cells_id, labels = get_profile_dyn(areas, lines, binary, input_image, loc_max, time, minref, mindiff, minsize)
    else: cells_id, labels = get_profile(areas, binary, input_image, loc_max, time, minref)

    return cells_id, labels


def belonging(ref, minval, maxima):
    """
    This function segments the entire 2-D array into areas belonging to each identified
    maximum according to a watershed algorithm.
    """
    ref_t = np.zeros(ref.shape)
    ref_t[:] = minval
    ref_t[ref > minval] = ref[ref > minval]
    markers = ndi.label(maxima)[0]
    areas = skis.watershed(-ref_t, markers=markers)
    lines=skis.watershed(-ref_t, markers=markers, watershed_line=True)

    return areas, lines


def longdistance(loc_max, mindis):
    """
    This function computes the distance between all maxima and rejects maxima that are
    less than a minimum distance apart.
    """
    x_max = loc_max[1]
    y_max = loc_max[0]
    n = 0
    while n < len(y_max):
        disx = x_max[n] - x_max
        disy = y_max[n] - y_max
        dis = np.sqrt(disx * disx + disy * disy)
        close = np.where(dis < mindis)[0]
        close = np.delete(close, np.where(close <= n))
        if len(close) > 0:
            x_max = np.delete(x_max, close)
            y_max = np.delete(y_max, close)
        n += 1

    new_max = y_max, x_max

    return new_max


def get_profile(areas, binary, ref, loc_max, time, minref):
    """
    This function returns the identified cells in a dataframe including their x,y
    locations, location of their maxima, maximum reflectivity and contours.
    """
    cells = areas * binary
    cell_labels = cells[loc_max]
    labels = np.zeros(cells.shape)
    cells_id = pd.DataFrame(
        data=None,
        index=range(len(cell_labels)),
        columns=["ID", "time", "x", "y", "max_x", "max_y", "max_ref", "cont"],
    )
    cells_id.time = time
    for n in range(len(cell_labels)):
        ID = n + 1
        cells_id.ID[n] = ID
        cells_id.x[n] = np.where(cells == cell_labels[n])[1]
        cells_id.y[n] = np.where(cells == cell_labels[n])[0]
        cell_unique = np.zeros(cells.shape)
        cell_unique[cells == cell_labels[n]] = 1
        contours = skime.find_contours(cell_unique, 0.8)
        maxval = cell_unique[loc_max]
        l = np.where(maxval == 1)
        y = loc_max[0][l]
        x = loc_max[1][l]
        maxref = np.nanmax(ref[cells_id.y[n], cells_id.x[n]])
        y, x = np.where(cell_unique * ref == maxref)
        contours = skime.find_contours(cell_unique, 0.8)
        cells_id.cont[n] = contours
        cells_id.max_x[n] = int(np.nanmean(cells_id.x[n]))  # int(x[0])
        cells_id.max_y[n] = int(np.nanmean(cells_id.y[n]))  # int(y[0])
        cells_id.max_ref[n] = maxref
        labels[cells == cell_labels[n]] = ID

    return cells_id, labels

def get_profile_dyn(areas, lines, binary, ref, loc_max, time, minref, dref, min_size):
    """
    This function returns the identified cells in a dataframe including their x,y
    locations, location of their maxima, maximum reflectivity and contours. The lower reflectivity bound is variable
    """
    lines_bin=lines==0
    ref[np.isnan(ref)]=np.nanmin(ref)
    cells = areas * binary
    cell_labels = cells[loc_max]
    labels = np.zeros(cells.shape)
    cells_id = pd.DataFrame(
        data=None,
        index=range(len(cell_labels)),
        columns=["ID", "time", "x", "y", "max_x", "max_y", "max_ref", "cont"],
    )
    cells_id.time = time
    for n in range(len(cell_labels)):
        ID = n + 1
        cells_id.ID[n] = ID
        cell_unique = np.zeros(cells.shape)
        cell_unique[cells == cell_labels[n]] = 1
        max_ref = np.nanmax((ref*cell_unique).flatten())
        cell_edge=skim.binary_dilation(cell_unique, selem=np.ones([2,2]))*lines_bin
        refvec=ref[cell_edge==1]
        if len(refvec)>0:
            min_ref=np.nanmax(refvec)
            if max_ref-min_ref<dref: min_ref=max_ref-dref
        else: min_ref=minref
        ref_unique=cell_unique*ref
        loc=np.where(ref_unique>=min_ref)
        labels1, ngroups = ndi.label(cell_unique)
        c_unique=np.zeros(cells.shape)
        c_unique[loc]=1
        labels1, n_groups = ndi.label(c_unique)
        while (len(loc[0])<min_size or n_groups>ngroups) and min_ref>minref:
            min_ref-=1
            ref_unique=cell_unique*ref
            loc=np.where(ref_unique>=min_ref)
            c_unique=np.zeros(cells.shape)
            c_unique[loc]=1
            labels1, n_groups = ndi.label(c_unique)
        
        cells_id.x[n] = np.where(c_unique == 1)[1]
        cells_id.y[n] = np.where(c_unique == 1)[0]
        contours = skime.find_contours(c_unique, 0.8)
        maxval = c_unique[loc_max]
        l = np.where(maxval == 1)
        y = loc_max[0][l]
        x = loc_max[1][l]
        maxref = np.nanmax(ref_unique.flatten())
        y, x = np.where(c_unique * ref == maxref)
        contours = skime.find_contours(c_unique, 0.8)
        cells_id.cont[n] = contours
        cells_id.max_x[n] = int(np.nanmean(cells_id.x[n]))  # int(x[0])
        cells_id.max_y[n] = int(np.nanmean(cells_id.y[n]))  # int(y[0])
        cells_id.max_ref[n] = maxref
        labels[c_unique == 1] = ID

    return cells_id, labels

