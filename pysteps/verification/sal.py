# -- coding: utf-8 --
"""
pysteps.verification.sal
==================================

The Spatial-Amplitude-Location (SAL) score.

.. autosummary::
    :toctree: ../generated/

    sal
    sal_structure
    sal_amplitude
    sal_location
"""
from math import sqrt, hypot

import numpy as np
from scipy.ndimage.measurements import center_of_mass

from pysteps.exceptions import MissingOptionalDependency
from pysteps.feature import tstorm as tstorm_detect

try:
    import pandas as pd

    pandas_imported = True
except ImportError:
    pandas_imported = False

try:
    from skimage.measure import regionprops_table

    skimage_imported = True
except ImportError:
    skimage_imported = False


def sal(
    prediction,
    observation,
    tstorm_kwargs=None,
):
    """
    Compute the Structure-Amplitude-Location (SAL) spatial verification metric.

    Parameters
    ----------
    prediction: array-like
        Array of shape (m,n) with prediction data.
    observation: array-like
        Array of shape (m,n)  with bservation data.
    tstorm_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the tstorm feature
        detection algorithm.
        See the documentation of :py:mod:`pysteps.feature.tstorm`.

    Returns
    -------
    sal: tuple of floats
        A tuple of floats containing the structure, amplitude, location components of SAL.

    References
    ----------
    :cite: Wernli, H., Hofmann, C., & Zimmer, M. (2009).
    :cite: Feldmann, M., Germann, U., Gabella, M., & Berne, A. (2021).

    Notes
    -----
    This implementation uses the thunderstorm detection algorithm by Feldmann et al (2021)
    for the identification of precipitation objects within the considered domain.

    This approach uses multi-threshold algorithm to detect objects, instead of having a
    single threshold (f).

    See also
    --------
    pysteps.feature.tstorm
    """
    if np.nanmax(observation >= 0.1) & np.nanmax(
        prediction >= 0.1
    ):  # to avoid errors of nan values or very low precipitation
        structure = sal_structure(prediction, observation, tstorm_kwargs)
        amplitude = sal_amplitude(prediction, observation)
        location = sal_location(prediction, observation, tstorm_kwargs)
    else:
        structure = np.nan
        amplitude = np.nan
        location = np.nan
    return structure, amplitude, location


def sal_structure(prediction, observation, tstorm_kwargs=None):
    """This function calculates the structure component for SAL based on Wernli et al (2008).

    Parameters
    ----------
    prediction: array-like
        Array of shape (m,n) with prediction data.
    observation: array-like
        Array of shape (m,n)  with bservation data.
    tstorm_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the tstorm feature
        detection algorithm.
        See the documentation of :py:mod:`pysteps.feature.tstorm`.

    Returns
    -------
    structure: float
        The structure component with value between -2 to 2.
    """
    prediction_volume = _sal_scaled_volume(prediction, tstorm_kwargs).scaled_v.sum()
    observation_volume = _sal_scaled_volume(observation, tstorm_kwargs).scaled_v.sum()
    nom = prediction_volume - observation_volume
    denom = prediction_volume + observation_volume
    return nom / (0.5 * denom)


def sal_amplitude(prediction, observation):
    """Calculate the amplitude component for SAL based on Wernli et al (2008).

    This component is the normalized difference of the domain-averaged precipitation
    in observation and forecast.

    Parameters
    ----------
    prediction: array-like
        Array of shape (m,n) with prediction data.
    observation: array-like
        Array of shape (m,n)  with bservation data.

    Returns
    -------
    amplitude: float
        Amplitude parameter with value between -2 to 2 and 0
        denotes perfect forecast in terms of amplitude.
    """
    mean_obs = np.nanmean(observation)
    mean_pred = np.nanmean(prediction)
    return (mean_pred - mean_obs) / (0.5 * (mean_pred + mean_obs))


def sal_location(prediction, observation, tstorm_kwargs=None):
    """Calculate the first parameter of location component for SAL based on
    Wernli et al (2008).

    This parameter indicates the normalized distance between the center of mass in
    observation and forecast.

    Parameters
    ----------
    prediction: array-like
        Array of shape (m,n) with prediction data.
    observation: array-like
        Array of shape (m,n)  with bservation data.
    tstorm_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the tstorm feature
        detection algorithm.
        See the documentation of :py:mod:`pysteps.feature.tstorm`.

    Returns
    -------
    location: float
        The location component with value between 0 to 1  and 0
        denotes perfect forecast in terms of location.
    """
    return sal_l1_param(prediction, observation) + sal_l2_param(
        prediction, observation, tstorm_kwargs
    )


def sal_l1_param(prediction, observation):
    """Calculate the first parameter of location component for SAL based on
    Wernli et al (2008).

    This parameter indicates the normalized distance between the center of mass in
    observation and forecast.

    Parameters
    ----------
    prediction: array-like
        Array of shape (m,n) with prediction data.
    observation: array-like
        Array of shape (m,n)  with bservation data.

    Returns
    -------
    location_1: float
        The first parameter of location component which has a value between 0 to 1.
    """
    maximum_distance = sqrt(
        ((observation.shape[0]) ** 2) + ((observation.shape[1]) ** 2)
    )
    obi = center_of_mass(np.nan_to_num(observation))
    fori = center_of_mass(np.nan_to_num(prediction))
    dist = hypot(fori[1] - obi[1], fori[0] - obi[0])
    return dist / maximum_distance


def sal_l2_param(prediction, observation, tstorm_kwargs=None):
    """Calculate the second parameter of location component for SAL based on Wernli et al (2008).

    Parameters
    ----------
    prediction: array-like
        Array of shape (m,n) with prediction data.
    observation: array-like
        Array of shape (m,n)  with bservation data.
    tstorm_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the tstorm feature
        detection algorithm.
        See the documentation of :py:mod:`pysteps.feature.tstorm`.

    Returns
    -------
    location_2: float
        The secibd parameter of location component with value between 0 to 1.
    """
    maximum_distance = sqrt(
        ((observation.shape[0]) ** 2) + ((observation.shape[1]) ** 2)
    )
    obs_r = (_sal_weighted_distance(observation, tstorm_kwargs)) * (observation.mean())
    forc_r = (_sal_weighted_distance(prediction, tstorm_kwargs)) * (prediction.mean())
    location_2 = 2 * ((abs(obs_r - forc_r)) / maximum_distance)
    return float(location_2)


def _sal_detect_objects(precip, tstorm_kwargs=None):
    """This function detects thunderstorms using a multi-threshold approach (Feldmann et al., 2021).

    Parameters
    ----------
    precip: array-like
        Array of shape (m,n) containing input data. Nan values are ignored.
    tstorm_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the tstorm feature
        detection algorithm.
        See the documentation of :py:mod:`pysteps.feature.tstorm`.

    Returns
    -------
    precip_objects: pd.DataFrame
        Dataframe containing all detected cells and their respective properties corresponding to the input data.
        Columns of dataframe: label, area, centroid, weighted centroid, intensity_max, intensity_mean, image_intensity
    """
    if not pandas_imported:
        raise MissingOptionalDependency(
            "The pandas package is required for the SAL "
            "verification method but it is not installed"
        )
    if not skimage_imported:
        raise MissingOptionalDependency(
            "The scikit-image package is required for the SAL "
            "verification method but it is not installed"
        )
    if tstorm_kwargs is None:
        tstorm_kwargs = dict()
    _, labels = tstorm_detect.detection(precip, **tstorm_kwargs)
    labels = labels.astype(int)
    properties = [
        "label",
        "area",
        "centroid",
        "weighted_centroid",
        "intensity_max",
        "intensity_mean",
        "image_intensity",
    ]
    precip_objects = pd.DataFrame(
        regionprops_table(labels, intensity_image=precip, properties=properties)
    )
    return precip_objects


def _sal_scaled_volume(precip, tstorm_kwargs=None):
    """Calculate the scaled volume parameter based on Wernli et al (2008).

    Parameters
    ----------
    precip: array-like
        Array of shape (m,n).
    tstorm_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the tstorm feature
        detection algorithm.
        See the documentation of :py:mod:`pysteps.feature.tstorm`.
    Returns
    -------
    vol_value: pd.DataFrame
        A dataframe that includes precipitation characteristics (sum, max, number
        of wet cells, and scaled volume) of the input data.
    """
    if not pandas_imported:
        raise MissingOptionalDependency(
            "The pandas package is required for the SAL "
            "verification method but it is not installed"
        )
    ch = []
    ds_with_ob = _sal_detect_objects(precip, tstorm_kwargs)

    for o in ds_with_ob.label - 1:
        tot = ds_with_ob.image_intensity[o].sum()
        mx = ds_with_ob.intensity_max[o]
        n = ds_with_ob.area[o]
        v_ob = tot / mx

        ch.append(
            {
                "obj": o,
                "precip_sum": tot,
                "precip_max": mx,
                "number_of_cells": n,
                "sum": tot.sum(),
                "scaled_v": v_ob,
            }
        )
    return pd.DataFrame(ch)


def _sal_weighted_distance(precip, tstorm_kwargs=None):
    """Compute the weighted averaged distance between the centers of mass of the
    individual objects and the center of mass of the total precipitation field.

    Parameters
    ----------
    precip: array-like
        Array of shape (m,n).
    tstorm_kwargs: dict, optional
        Optional dictionary containing keyword arguments for the tstorm feature
        detection algorithm.
        See the documentation of :py:mod:`pysteps.feature.tstorm`.

    Returns
    -------
    weighted_distance: float
        The weighted averaged distance between the centers of mass of the
        individual objects and the center of mass of the total precipitation field.
    """
    if not pandas_imported:
        raise MissingOptionalDependency(
            "The pandas package is required for the SAL "
            "verification method but it is not installed"
        )
    precip_objects = _sal_detect_objects(precip, tstorm_kwargs)
    centroid_total = center_of_mass(np.nan_to_num(precip))
    r = []
    for i in precip_objects.label - 1:
        xd = (precip_objects["weighted_centroid-1"][i] - centroid_total[1]) ** 2
        yd = (precip_objects["weighted_centroid-0"][i] - centroid_total[0]) ** 2

        dst = sqrt(xd + yd)
        sumr = (precip_objects.image_intensity[i].sum()) * dst

        sump = precip_objects.image_intensity[i].sum()

        r.append({"sum_dist": sumr, "sum_p": sump})
    rr = pd.DataFrame(r)
    return rr.sum_dist.sum() / (rr.sum_p.sum())
