# -*- coding: utf-8 -*-
"""
pysteps.tracking.tdating
========================

Thunderstorm Detection and Tracking (DATing) module
This module was implemented following the procedures used in the TRT Thunderstorms
Radar Tracking algorithm (:cite:`TRT2004`) used operationally at MeteoSwiss.
Full documentation is published in :cite:`Feldmann2021`.
Modifications include advecting the identified thunderstorms with the optical flow
obtained from pysteps, as well as additional options in the thresholding.

References
...............
:cite:`TRT2004`
:cite:`Feldmann2021`

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
    match_frac=0.4,
    split_frac=0.1,
    merge_frac=0.1,
    output_splits_merges=False,
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
    match_frac: float, optional
        Minimum overlap fraction between two objects to be considered the same object.
        Default is 0.4.
    split_frac: float, optional
        Minimum overlap fraction between two objects for the object at second timestep
        to be considered possibly split from the object at the first timestep.
        Default is 0.1.
    merge_frac: float, optional
        Minimum overlap fraction between two objects for the object at second timestep
        to be considered possibly merged from the object at the first timestep.
        Default is 0.1.
    output_splits_merges: bool, optional
        If True, the output will contain information about splits and merges.
        The provided columns are:

        .. tabularcolumns:: |p{2cm}|L|

        +-------------------+--------------------------------------------------------------+
        | Attribute         | Description                                                  |
        +===================+==============================================================+
        | splitted          | Indicates if the cell is considered split into multiple cells|
        +-------------------+--------------------------------------------------------------+
        | split_IDs         | List of IDs at the next timestep that the cell split into    |
        +-------------------+--------------------------------------------------------------+
        | merged            | Indicates if the cell is considered a merge of multiple cells|
        +-------------------+--------------------------------------------------------------+
        | merged_IDs        | List of IDs from the previous timestep that merged into this |
        |                   | cell                                                         |
        +-------------------+--------------------------------------------------------------+
        | results_from_split| True if the cell is a result of a split (i.e., the ID of the |
        |                   | cell is present in the split_IDs of some cell at the previous|
        |                   | timestep)                                                    |
        +-------------------+--------------------------------------------------------------+
        | will_merge        | True if the cell will merge at the next timestep (i.e., the  |
        |                   | ID of the cell is present in the merge_IDs of some cell at   |
        |                   | the next timestep; empty if the next timestep is not tracked)|
        +-------------------+--------------------------------------------------------------+

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
    if len(label_list) == 0:
        max_ID = 0
    else:
        max_ID = np.nanmax([np.nanmax(np.unique(label_list)), 0])
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
            output_splits_merges=output_splits_merges,
        )
        if len(cell_list) < 2:
            cell_list.append(cells_id)
            label_list.append(labels)
            cid = np.unique(labels)
            max_ID = np.nanmax([np.nanmax(cid), max_ID]) + 1
            continue
        if t >= 2:
            flowfield = oflow_method(input_video[t - 2 : t + 1, :, :])
            cells_id, max_ID, newlabels, splitted_cells = tracking(
                cells_id,
                cell_list[-1],
                labels,
                flowfield,
                max_ID,
                match_frac=match_frac,
                split_frac=split_frac,
                merge_frac=merge_frac,
                output_splits_merges=output_splits_merges,
            )

            if output_splits_merges:
                # Assign splitted parameters for the previous timestep
                for _, split_cell in splitted_cells.iterrows():
                    prev_list_id = cell_list[-1][
                        cell_list[-1].ID == split_cell.ID
                    ].index.item()

                    split_ids = split_cell.split_IDs
                    split_ids_updated = []
                    for sid in split_ids:
                        split_ids_updated.append(newlabels[labels == sid][0])

                    cell_list[-1].at[prev_list_id, "splitted"] = True
                    cell_list[-1].at[prev_list_id, "split_IDs"] = split_ids_updated

                    for sid in split_ids_updated:
                        cur_list_id = cells_id[cells_id.ID == sid].index.item()
                        cells_id.at[cur_list_id, "results_from_split"] = True

                merged_cells = cells_id[cells_id.merged == True]
                for _, cell in merged_cells.iterrows():
                    for merged_id in cell.merged_IDs:
                        prev_list_id = cell_list[-1][
                            cell_list[-1].ID == merged_id
                        ].index.item()
                        cell_list[-1].at[prev_list_id, "will_merge"] = True

            cid = np.unique(newlabels)
            # max_ID = np.nanmax([np.nanmax(cid), max_ID])
            cell_list.append(cells_id)
            label_list.append(newlabels)

    track_list = couple_track(cell_list[2:], int(max_ID), mintrack)

    return track_list, cell_list, label_list


def tracking(
    cells_id,
    cells_id_prev,
    labels,
    V1,
    max_ID,
    match_frac=0.4,
    merge_frac=0.1,
    split_frac=0.1,
    output_splits_merges=False,
):
    """
    This function performs the actual tracking procedure. First the cells are advected,
    then overlapped and finally their IDs are matched. If no match is found, a new ID
    is assigned.
    """
    cells_id_new = cells_id.copy()
    cells_ad = advect(
        cells_id_prev, labels, V1, output_splits_merges=output_splits_merges
    )
    cells_ov, labels, possible_merge_ids = match(
        cells_ad,
        labels,
        output_splits_merges=output_splits_merges,
        split_frac=split_frac,
        match_frac=match_frac,
    )

    splitted_cells = None
    if output_splits_merges:
        splitted_cells = cells_ov[cells_ov.splitted == True]

    newlabels = np.zeros(labels.shape)
    possible_merge_ids_new = {}
    for index, cell in cells_id_new.iterrows():
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
            cells_id_new.loc[index, "ID"] = new_ID
        else:
            max_ID += 1
            new_ID = max_ID
            cells_id_new.loc[index, "ID"] = new_ID
        newlabels[labels == index + 1] = new_ID
        possible_merge_ids_new[new_ID] = possible_merge_ids[cell.ID]
        del new_ID

    if output_splits_merges:
        # Process possible merges
        for target_id, possible_IDs in possible_merge_ids_new.items():
            merge_ids = []
            for p_id in possible_IDs:
                cell_a = cells_ad[cells_ad.ID == p_id]

                ID_vec = newlabels[cell_a.y.item(), cell_a.x.item()]
                overlap = np.sum(ID_vec == target_id) / len(ID_vec)
                if overlap > merge_frac:
                    merge_ids.append(p_id)

            if len(merge_ids) > 1:
                cell_id = cells_id_new[cells_id_new.ID == target_id].index.item()
                # Merge cells
                cells_id_new.at[cell_id, "merged"] = True
                cells_id_new.at[cell_id, "merged_IDs"] = merge_ids

    return cells_id_new, max_ID, newlabels, splitted_cells


def advect(cells_id, labels, V1, output_splits_merges=False):
    """
    This function advects all identified cells with the estimated flow.
    """
    columns = [
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
    ]
    if output_splits_merges:
        columns.extend(["splitted", "split_IDs", "split_fracs"])
    cells_ad = pd.DataFrame(
        data=None,
        index=range(len(cells_id)),
        columns=columns,
    )
    for ID, cell in cells_id.iterrows():
        if cell.ID == 0 or np.isnan(cell.ID):
            continue
        ad_x = np.round(np.nanmean(V1[0, cell.y, cell.x])).astype(int)
        ad_y = np.round(np.nanmean(V1[1, cell.y, cell.x])).astype(int)
        new_x = cell.x + ad_x
        new_y = cell.y + ad_y
        new_x[new_x > labels.shape[1] - 1] = labels.shape[1] - 1
        new_y[new_y > labels.shape[0] - 1] = labels.shape[0] - 1
        new_x[new_x < 0] = 0
        new_y[new_y < 0] = 0
        new_cen_x = cell.cen_x + ad_x
        new_cen_y = cell.cen_y + ad_y

        cells_ad.loc[ID, "x"] = new_x
        cells_ad.loc[ID, "y"] = new_y
        cells_ad.loc[ID, "flowx"] = ad_x
        cells_ad.loc[ID, "flowy"] = ad_y
        cells_ad.loc[ID, "cen_x"] = new_cen_x
        cells_ad.loc[ID, "cen_y"] = new_cen_y
        cells_ad.loc[ID, "ID"] = cell.ID

        cell_unique = np.zeros(labels.shape)
        cell_unique[new_y, new_x] = 1
        cells_ad.loc[ID, "cont"] = skime.find_contours(cell_unique, 0.8)

    return cells_ad


def match(cells_ad, labels, match_frac=0.4, split_frac=0.1, output_splits_merges=False):
    """
    This function matches the advected cells of the previous timestep to the newly
    identified ones. A minimal overlap of 40% is required. In case of split of merge,
    the larger cell supersedes the smaller one in naming.
    """
    cells_ov = cells_ad.copy()
    possible_merge_ids = {i: [] for i in np.unique(labels)}
    for ID_a, cell_a in cells_ov.iterrows():
        if cell_a.ID == 0 or np.isnan(cell_a.ID):
            continue
        ID_vec = labels[cell_a.y, cell_a.x]
        IDs = np.unique(ID_vec)
        n_IDs = len(IDs)
        if n_IDs == 1 and IDs[0] == 0:
            cells_ov.loc[ID_a, "t_ID"] = 0
            continue
        IDs = IDs[IDs != 0]
        n_IDs = len(IDs)

        for i in IDs:
            possible_merge_ids[i].append(cell_a.ID)

        N = np.zeros(n_IDs)
        for n in range(n_IDs):
            N[n] = len(np.where(ID_vec == IDs[n])[0])

        if output_splits_merges:
            # Only consider possible split if overlap is large enough
            valid_split_ids = (N / len(ID_vec)) > split_frac
            # splits here
            if sum(valid_split_ids) > 1:
                # Save split information
                cells_ov.loc[ID_a, "splitted"] = True
                cells_ov.loc[ID_a, "split_IDs"] = IDs[valid_split_ids]
                cells_ov.loc[ID_a, "split_fracs"] = N / len(ID_vec)

        m = np.argmax(N)
        ID_match = IDs[m]
        ID_coverage = N[m] / len(ID_vec)
        if ID_coverage >= match_frac:
            cells_ov.loc[ID_a, "t_ID"] = ID_match
        else:
            cells_ov.loc[ID_a, "t_ID"] = 0
        cells_ov.loc[ID_a, "frac"] = ID_coverage
    return cells_ov, labels, possible_merge_ids


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
        cell_track = []
        for t in range(len(cell_list)):
            mytime = cell_list[t]
            cell_track.append(mytime[mytime.ID == n])
        cell_track = pd.concat(cell_track, axis=0)

        if len(cell_track) < mintrack:
            continue
        track_list.append(cell_track)
    return track_list
