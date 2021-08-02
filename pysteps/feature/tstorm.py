"""
pysteps.feature.tstorm
======================

Thunderstorm cell detection module, part of Thunderstorm Detection and Tracking (DATing)
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

    detection
    breakup
    longdistance
    get_profile
"""

import numpy as np
import scipy.ndimage as ndi

from pysteps.exceptions import MissingOptionalDependency

try:
    import skimage

    SKIMAGE_IMPORTED = True
except ImportError:
    SKIMAGE_IMPORTED = False
if SKIMAGE_IMPORTED:
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
    max_num_features=None,
    minref=35,
    maxref=48,
    mindiff=6,
    minsize=50,
    minmax=41,
    mindis=10,
    output_feat=False,
    time="000000000",
):
    """
    This function detects thunderstorms using a multi-threshold approach. It is
    recommended to use a 2-D Cartesian maximum reflectivity composite, however the
    function will process any 2-D array.
    The thunderstorm cell detection requires both scikit-image and pandas.

    Parameters
    ----------
    input_image: array-like
        Array of shape (m,n) containing input image, usually maximum reflectivity in
        dBZ with a resolution of 1 km. Nan values are ignored.
    max_num_features : int, optional
        The maximum number of cells to detect. Set to None for no restriction.
        If specified, the most significant cells are chosen based on their area.
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
    output_feat: bool, optional
        Set to True to return only the cell coordinates.
    time: string, optional
        Date and time as string. Used to label time in the resulting dataframe.
        The default is '000000000'.

    Returns
    -------
    cells_id: pandas dataframe
        Pandas dataframe containing all detected cells and their respective properties
        corresponding to the input image.
        Columns of dataframe: ID - cell ID, time - time stamp, x - array of all
        x-coordinates of cell, y -  array of all y-coordinates of cell, cen_x -
        x-coordinate of cell centroid, cen_y - y-coordinate of cell centroid, max_ref -
        maximum (reflectivity) value of cell, cont - cell contours
    labels: array-like
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
    filt_image[input_image >= minref] = input_image[input_image >= minref]
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
        if size < minsize:  # removing too small areas
            binary[labels == n] = 0
            labels[labels == n] = 0
        if maxval < minmax:  # removing areas with too low max value
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

    areas, lines = breakup(input_image, np.nanmin(input_image.flatten()), maxima_dis)

    cells_id, labels = get_profile(areas, binary, input_image, loc_max, time, minref)

    if max_num_features is not None:
        idx = np.argsort(cells_id.area.to_numpy())[::-1]

    if not output_feat:
        if max_num_features is None:
            return cells_id, labels
        else:
            for i in idx[max_num_features:]:
                labels[labels == cells_id.ID[i]] = 0
            return cells_id.loc[idx[:max_num_features]], labels
    if output_feat:
        out = np.column_stack([np.array(cells_id.cen_x), np.array(cells_id.cen_y)])
        if max_num_features is not None:
            out = out[idx[:max_num_features], :]

        return out


def breakup(ref, minval, maxima):
    """
    This function segments the entire 2-D array into areas belonging to each identified
    maximum according to a watershed algorithm.
    """
    ref_t = np.zeros(ref.shape)
    ref_t[:] = minval
    ref_t[ref > minval] = ref[ref > minval]
    markers = ndi.label(maxima)[0]
    areas = skis.watershed(-ref_t, markers=markers)
    lines = skis.watershed(-ref_t, markers=markers, watershed_line=True)

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
        columns=["ID", "time", "x", "y", "cen_x", "cen_y", "max_ref", "cont", "area"],
    )
    cells_id.time = time
    for n in range(len(cell_labels)):
        ID = n + 1
        cells_id.ID.iloc[n] = ID
        cells_id.x.iloc[n] = np.where(cells == cell_labels[n])[1]
        cells_id.y.iloc[n] = np.where(cells == cell_labels[n])[0]
        cell_unique = np.zeros(cells.shape)
        cell_unique[cells == cell_labels[n]] = 1
        maxref = np.nanmax(ref[cells_id.y[n], cells_id.x[n]])
        contours = skime.find_contours(cell_unique, 0.8)
        cells_id.cont.iloc[n] = contours
        cells_id.cen_x.iloc[n] = int(np.nanmean(cells_id.x[n]))  # int(x[0])
        cells_id.cen_y.iloc[n] = int(np.nanmean(cells_id.y[n]))  # int(y[0])
        cells_id.max_ref.iloc[n] = maxref
        cells_id.area.iloc[n] = len(cells_id.x.iloc[n])
        labels[cells == cell_labels[n]] = ID

    return cells_id, labels
