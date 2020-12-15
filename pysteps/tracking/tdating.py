# -*- coding: utf-8 -*-
"""
pysteps.tracking.tdating
========================

Thunderstorm Detection and Tracking (DATing) module
This module was implemented following the procedures used in the TRT Thunderstorms
Radar Tracking algorithm (:cite:`TRT2004`) used operationally at MeteoSwiss.
Modifications include advecting the identified thunderstorms with the optical flow
obtained from pysteps, as well as additional options in the thresholding.

References
...............
:cite:`TRT2004`

@author: mfeldman

.. autosummary::
    :toctree: ../generated/

    dating
    tracking
    advect
    match
    couple_track
"""

import numpy as np

import pysteps.feature.tstorm as tstorm_detect
from pysteps import motion
from pysteps.exceptions import MissingOptionalDependency

try:
    import skimage

    SKIMAGE_IMPORTED = True
except ImportError:
    SKIMAGE_IMPORTED = False
if SKIMAGE_IMPORTED:
    import skimage.measure as skime
try:
    import pandas as pd

    PANDAS_IMPORTED = True
except ImportError:
    PANDAS_IMPORTED = False


def dating(
    input_video,
    timelist,
    mintrack=3,
    cell_list=None,
    label_list=None,
    start=0,
    minref=35,
    maxref=48,
    mindiff=6,
    minsize=50,
    minmax=41,
    mindis=10,
    dyn_thresh=False,
):
    """
    This function performs the thunderstorm detection and tracking DATing.
    It requires a 3-D input array that contains the temporal succession of the 2-D data
    array of each timestep. On each timestep the detection is performed, the identified
    objects are advected with a flow prediction and the advected objects are matched to
    the newly identified objects of the next timestep.
    The last portion re-arranges the data into tracks sorted by ID-number.

    Parameters
    ----------
    input_video: array-like
        Array of shape (t,m,n) containing input image, with t being the temporal
        dimension and m,n the spatial dimensions. Thresholds are tuned to maximum
        reflectivity in dBZ with a spatial resolution of 1 km and a temporal resolution
        of 5 min. Nan values are ignored.
    timelist: list
        List of length t containing string of time and date of each (m,n) field.
    mintrack: int, optional
        minimum duration of cell-track to be counted. The default is 3 time steps.
    cell_list: list or None, optional
        If you wish to expand an existing list of cells, insert previous cell-list here.
        The default is None.
        If not None, requires that label_list has the same length.
    label_list: list or None, optional
        If you wish to expand an existing list of cells, insert previous label-list here.
        The default is None.
        If not None, requires that cell_list has the same length.
    start: int, optional
        If you wish to expand an existing list of cells, the input video must contain 2
        timesteps prior to the merging. The start can then be set to 2, allowing the
        motion vectors to be formed from the first three grids and continuing the cell
        tracking from there. The default is 0, which initiates a new tracking sequence.
    minref: float, optional
        Lower threshold for object detection. Lower values will be set to NaN.
        The default is 35 dBZ.
    maxref: float, optional
        Upper threshold for object detection. Higher values will be set to this value.
        The default is 48 dBZ.
    mindiff: float, optional
        Minimal difference between two identified maxima within same area to split area
        into two objects. The default is 6 dBZ.
    minsize: float, optional
        Minimal area for possible detected object. The default is 50 pixels.
    minmax: float, optional
        Minimum value of maximum in identified objects. Objects with a maximum lower
        than this will be discarded. The default is 41 dBZ.
    mindis: float, optional
        Minimum distance between two maxima of identified objects. Objects with a
        smaller distance will be merged. The default is 10 km.

    Returns
    -------
    track_list: list of dataframes
        Each dataframe contains the track and properties belonging to one cell ID.
        Columns of dataframes: ID - cell ID, time - time stamp, x - array of all
        x-coordinates of cell, y -  array of all y-coordinates of cell, cen_x -
        x-coordinate of cell centroid, cen_y - y-coordinate of cell centroid, max_ref -
        maximum (reflectivity) value of cell, cont - cell contours
    cell_list: list of dataframes
        Each dataframe contains the detected cells and properties belonging to one
        timestep. The IDs are already matched to provide a track.
        Columns of dataframes: ID - cell ID, time - time stamp, x - array of all
        x-coordinates of cell, y -  array of all y-coordinates of cell, cen_x -
        x-coordinate of cell centroid, cen_y - y-coordinate of cell centroid, max_ref -
        maximum (reflectivity) value of cell, cont - cell contours
    label_list: list of arrays
        Each (n,m) array contains the gridded IDs of the cells identified in the
        corresponding timestep. The IDs are already matched to provide a track.

    """
    if not SKIMAGE_IMPORTED:
        raise MissingOptionalDependency(
            "skimage is required for thunderstorm DATing " "but it is not installed"
        )
    if not PANDAS_IMPORTED:
        raise MissingOptionalDependency(
            "pandas is required for thunderstorm DATing " "but it is not installed"
        )

    # Check arguments
    if cell_list is None or label_list is None:
        cell_list = []
        label_list = []
    else:
        if not len(cell_list) == len(label_list):
            raise ValueError("len(cell_list) != len(label_list)")
    if start > len(timelist):
        raise ValueError("start > len(timelist)")

    oflow_method = motion.get_method("LK")
    max_ID = 0
    for t in range(start, len(timelist)):
        cells_id, labels = tstorm_detect.detection(
            input_video[t, :, :],
            minref=minref,
            maxref=maxref,
            mindiff=mindiff,
            minsize=minsize,
            minmax=minmax,
            mindis=mindis,
            time=timelist[t],
        )
        if len(cell_list) < 2:
            cell_list.append(cells_id)
            label_list.append(labels)
            cid = np.unique(labels)
            max_ID = np.nanmax([np.nanmax(cid), max_ID]) + 1
            continue
        if t >= 2:
            flowfield = oflow_method(input_video[t - 2 : t + 1, :, :])
            cells_id, max_ID, newlabels = tracking(
                cells_id, cell_list[-1], labels, flowfield, max_ID
            )
            cid = np.unique(newlabels)
            # max_ID = np.nanmax([np.nanmax(cid), max_ID])
            cell_list.append(cells_id)
            label_list.append(newlabels)

    track_list = couple_track(cell_list[2:], int(max_ID), mintrack)

    return track_list, cell_list, label_list


def tracking(cells_id, cells_id_prev, labels, V1, max_ID):
    """
    This function performs the actual tracking procedure. First the cells are advected,
    then overlapped and finally their IDs are matched. If no match is found, a new ID
    is assigned.
    """
    cells_id_new = cells_id.copy()
    cells_ad = advect(cells_id_prev, labels, V1)
    cells_ov, labels = match(cells_ad, labels)
    newlabels = np.zeros(labels.shape)
    for ID, cell in cells_id_new.iterrows():
        if cell.ID == 0 or np.isnan(cell.ID):
            continue
        new_ID = cells_ov[cells_ov.t_ID == cell.ID].ID.values
        if len(new_ID) > 0:
            xx = cells_ov[cells_ov.t_ID == cell.ID].x
            size = []
            for x in xx:
                size.append(len(x))
            biggest = np.argmax(size)
            new_ID = new_ID[biggest]
            cells_id_new.ID.iloc[ID] = new_ID
        else:
            max_ID += 1
            new_ID = max_ID
            cells_id_new.ID.iloc[ID] = new_ID
        newlabels[labels == ID + 1] = new_ID
        del new_ID
    return cells_id_new, max_ID, newlabels


def advect(cells_id, labels, V1):
    """
    This function advects all identified cells with the estimated flow.
    """
    cells_ad = pd.DataFrame(
        data=None,
        index=range(len(cells_id)),
        columns=[
            "ID",
            "x",
            "y",
            "cen_x",
            "cen_y",
            "max_ref",
            "cont",
            "t_ID",
            "frac",
            "flowx",
            "flowy",
        ],
    )
    for ID, cell in cells_id.iterrows():
        if cell.ID == 0 or np.isnan(cell.ID):
            continue
        ad_x = int(np.nanmean(V1[0, cell.y, cell.x]))
        ad_y = int(np.nanmean(V1[1, cell.y, cell.x]))
        new_x = cell.x + ad_x
        new_y = cell.y + ad_y
        new_x[new_x > labels.shape[1] - 1] = labels.shape[1] - 1
        new_y[new_y > labels.shape[0] - 1] = labels.shape[0] - 1
        new_x[new_x < 0] = 0
        new_y[new_y < 0] = 0
        new_cen_x = cell.cen_x + ad_x
        new_cen_y = cell.cen_y + ad_y
        cells_ad.x[ID] = new_x
        cells_ad.y[ID] = new_y
        cells_ad.flowx[ID] = ad_x
        cells_ad.flowy[ID] = ad_y
        cells_ad.cen_x[ID] = new_cen_x
        cells_ad.cen_y[ID] = new_cen_y
        cells_ad.ID[ID] = cell.ID
        cell_unique = np.zeros(labels.shape)
        cell_unique[new_y, new_x] = 1
        cells_ad.cont[ID] = skime.find_contours(cell_unique, 0.8)

    return cells_ad


def match(cells_ad, labels):
    """
    This function matches the advected cells of the previous timestep to the newly
    identified ones. A minimal overlap of 40% is required. In case of split of merge,
    the larger cell supersedes the smaller one in naming.
    """
    cells_ov = cells_ad.copy()
    for ID_a, cell_a in cells_ov.iterrows():
        if cell_a.ID == 0 or np.isnan(cell_a.ID):
            continue
        ID_vec = labels[cell_a.y, cell_a.x]
        IDs = np.unique(ID_vec)
        n_IDs = len(IDs)
        N = np.zeros(n_IDs)
        for n in range(n_IDs):
            N[n] = len(np.where(ID_vec == IDs[n])[0])
        m = np.argmax(N)
        ID_match = IDs[m]
        ID_coverage = N[m] / len(ID_vec)
        if ID_coverage >= 0.4:
            cells_ov.t_ID[ID_a] = ID_match
        else:
            cells_ov.t_ID[ID_a] = 0
        cells_ov.frac[ID_a] = ID_coverage
    return cells_ov, labels


def couple_track(cell_list, max_ID, mintrack):
    """
    The coupled cell tracks are re-arranged from the list of cells sorted by time, to
    a list of tracks sorted by ID. Tracks shorter than mintrack are rejected.
    """
    track_list = []
    for n in range(1, max_ID):
        cell_track = pd.DataFrame(
            data=None,
            index=None,
            columns=["ID", "time", "x", "y", "cen_x", "cen_y", "max_ref", "cont"],
        )
        for t in range(len(cell_list)):
            mytime = cell_list[t]
            mycell = mytime[mytime.ID == n]
            cell_track = cell_track.append(mycell)

        if len(cell_track) < mintrack:
            continue
        track_list.append(cell_track)
    return track_list
