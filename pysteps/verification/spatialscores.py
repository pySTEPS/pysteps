# -- coding: utf-8 --
"""
pysteps.verification.spatialscores
==================================

Skill scores for spatial forecasts.

.. autosummary::
    :toctree: ../generated/

    intensity_scale
    intensity_scale_init
    intensity_scale_accum
    intensity_scale_merge
    intensity_scale_compute
    binary_mse
    binary_mse_init
    binary_mse_accum
    binary_mse_merge
    binary_mse_compute
    fss
    fss_init
    fss_accum
    fss_merge
    fss_compute
"""

import collections
import numpy as np
from pysteps.exceptions import MissingOptionalDependency
from scipy.ndimage.filters import uniform_filter

try:
    import pywt

    pywt_imported = True
except ImportError:
    pywt_imported = False


def intensity_scale(X_f, X_o, name, thrs, scales=None, wavelet="Haar"):
    """Compute an intensity-scale verification score.

    Parameters
    ----------
    X_f: array_like
        Array of shape (m, n) containing the forecast field.
    X_o: array_like
        Array of shape (m, n) containing the verification observation field.
    name: string
        A string indicating the name of the spatial verification score
        to be used:

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  FSS       | Fractions skill score                                  |
        +------------+--------------------------------------------------------+
        |  BMSE      | Binary mean squared error                              |
        +------------+--------------------------------------------------------+

    thrs: float or array_like
        Scalar or 1-D array of intensity thresholds for which to compute the
        verification.
    scales: float or array_like, optional
        Scalar or 1-D array of spatial scales in pixels,
        required if ``name="FSS"``.
    wavelet: str, optional
        The name of the wavelet function to use in the BMSE.
        Defaults to the Haar wavelet, as described in Casati et al. 2004.
        See the documentation of PyWavelets for a list of available options.

    Returns
    -------
    out: array_like
        The two-dimensional array containing the intensity-scale skill scores
        for each spatial scale and intensity threshold.

    References
    ----------
    :cite:`CRS2004`, :cite:`RL2008`, :cite:`EWWM2013`

    See also
    --------
    pysteps.verification.spatialscores.binary_mse,
    pysteps.verification.spatialscores.fss
    """

    intscale = intensity_scale_init(name, thrs, scales, wavelet)
    intensity_scale_accum(intscale, X_f, X_o)
    return intensity_scale_compute(intscale)


def intensity_scale_init(name, thrs, scales=None, wavelet="Haar"):
    """Initialize an intensity-scale verification object.

    Parameters
    ----------
    name: string
        A string indicating the name of the spatial verification score
        to be used:

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  FSS       | Fractions skill score                                  |
        +------------+--------------------------------------------------------+
        |  BMSE      | Binary mean squared error                              |
        +------------+--------------------------------------------------------+

    thrs: float or array_like
        Scalar or 1-D array of intensity thresholds for which to compute the
        verification.
    scales: float or array_like, optional
        Scalar or 1-D array of spatial scales in pixels,
        required if ``name="FSS"``.
    wavelet: str, optional
        The name of the wavelet function, required if ``name="BMSE"``.
        Defaults to the Haar wavelet, as described in Casati et al. 2004.
        See the documentation of PyWavelets for a list of available options.

    Returns
    -------
    out: dict
        The intensity-scale object.
    """

    if name.lower() == "fss" and scales is None:
        message = "an array of spatial scales must be provided for the FSS,"
        message += " but %s was passed" % scales
        raise ValueError(message)

    if name.lower() == "bmse" and wavelet is None:
        message = "the name of a wavelet must be provided for the BMSE,"
        message += " but %s was passed" % wavelet
        raise ValueError(message)

    # catch scalars when passed as arguments
    def get_iterable(x):
        if isinstance(x, collections.abc.Iterable):
            return np.copy(x)
        else:
            return np.copy((x,))

    intscale = {}
    intscale["name"] = name
    intscale["thrs"] = np.sort(get_iterable(thrs))
    if scales is not None:
        intscale["scales"] = np.sort(get_iterable(scales))[::-1]
    else:
        intscale["scales"] = scales
    intscale["wavelet"] = wavelet

    for i, thr in enumerate(intscale["thrs"]):

        if name.lower() == "bmse":
            intscale[thr] = binary_mse_init(thr, intscale["wavelet"])

        elif name.lower() == "fss":
            intscale[thr] = {}

            for j, scale in enumerate(intscale["scales"]):
                intscale[thr][scale] = fss_init(thr, scale)

    if name.lower() == "fss":
        intscale["label"] = "Fractions skill score"
        del intscale["wavelet"]

    elif name.lower() == "bmse":
        intscale["label"] = "Binary MSE skill score"
        intscale["scales"] = None

    else:
        raise ValueError("unknown method %s" % name)

    return intscale


def intensity_scale_accum(intscale, X_f, X_o):
    """Compute and update the intensity-scale verification scores.

    Parameters
    ----------
    intscale: dict
        The intensity-scale object initialized with
        :py:func:`pysteps.verification.spatialscores.intensity_scale_init`.
    X_f: array_like
        Array of shape (m, n) containing the forecast field.
    X_o: array_like
        Array of shape (m, n) containing the verification observation field.
    """

    name = intscale["name"]
    thrs = intscale["thrs"]
    scales = intscale["scales"]

    for i, thr in enumerate(thrs):

        if name.lower() == "bmse":
            binary_mse_accum(intscale[thr], X_f, X_o)

        elif name.lower() == "fss":
            for j, scale in enumerate(scales):
                fss_accum(intscale[thr][scale], X_f, X_o)

    if scales is None:
        intscale["scales"] = intscale[thrs[0]]["scales"]


def intensity_scale_merge(intscale_1, intscale_2):
    """Merge two intensity-scale verification objects.

    Parameters
    ----------
    intscale_1: dict
        Am intensity-scale object initialized with
        :py:func:`pysteps.verification.spatialscores.intensity_scale_init`
        and populated with
        :py:func:`pysteps.verification.spatialscores.intensity_scale_accum`.
    intscale_2: dict
        Another intensity-scale object initialized with
        :py:func:`pysteps.verification.spatialscores.intensity_scale_init`
        and populated with
        :py:func:`pysteps.verification.spatialscores.intensity_scale_accum`.

    Returns
    -------
    out: dict
      The merged intensity-scale object.
    """

    # checks
    if intscale_1["name"] != intscale_2["name"]:
        raise ValueError(
            "cannot merge: the intensity scale methods are not same %s!=%s"
            % (intscale_1["name"], intscale_2["name"])
        )

    intscale = intscale_1.copy()
    name = intscale["name"]
    thrs = intscale["thrs"]
    scales = intscale["scales"]

    for i, thr in enumerate(thrs):

        if name.lower() == "bmse":
            intscale[thr] = binary_mse_merge(intscale[thr], intscale_2[thr])

        elif name.lower() == "fss":
            for j, scale in enumerate(scales):
                intscale[thr][scale] = fss_merge(
                    intscale[thr][scale], intscale_2[thr][scale]
                )

    return intscale


def intensity_scale_compute(intscale):
    """Return the intensity scale matrix.

    Parameters
    ----------
    intscale: dict
        The intensity-scale object initialized with
        :py:func:`pysteps.verification.spatialscores.intensity_scale_init`
        and accumulated with
        :py:func:`pysteps.verification.spatialscores.intensity_scale_accum`.

    Returns
    -------
    out: array_like
        The two-dimensional array of shape (j, k) containing
        the intensity-scale skill scores for **j** spatial scales and
        **k** intensity thresholds.
    """

    name = intscale["name"]
    thrs = intscale["thrs"]
    scales = intscale["scales"]

    SS = np.zeros((scales.size, thrs.size))

    for i, thr in enumerate(thrs):

        if name.lower() == "bmse":
            SS[:, i] = binary_mse_compute(intscale[thr], False)

        elif name.lower() == "fss":
            for j, scale in enumerate(scales):
                SS[j, i] = fss_compute(intscale[thr][scale])

    return SS


def binary_mse(X_f, X_o, thr, wavelet="haar", return_scales=True):
    """Compute the MSE of the binary error as a function of spatial scale.

    This method uses PyWavelets for decomposing the error field between the
    forecasts and observations into multiple spatial scales.

    Parameters
    ----------
    X_f: array_like
        Array of shape (m, n) containing the forecast field.
    X_o: array_like
        Array of shape (m, n) containing the verification observation field.
    thr: sequence
        The intensity threshold for which to compute the verification.
    wavelet: str, optional
        The name of the wavelet function to use. Defaults to the Haar wavelet,
        as described in Casati et al. 2004. See the documentation of PyWavelets
        for a list of available options.
    return_scales: bool, optional
        Whether to return the spatial scales resulting from the wavelet
        decomposition.

    Returns
    -------
    SS: array
        One-dimensional array containing the binary MSE for each spatial scale.
    scales: list, optional
        If ``return_scales=True``, return the spatial scales in pixels resulting
        from the wavelet decomposition.

    References
    ----------
    :cite:`CRS2004`
    """

    bmse = binary_mse_init(thr, wavelet)
    binary_mse_accum(bmse, X_f, X_o)
    return binary_mse_compute(bmse, return_scales)


def binary_mse_init(thr, wavelet="haar"):
    """Initialize a binary MSE (BMSE) verification object.

    Parameters
    ----------
    thr: float
        The intensity threshold.
    wavelet: str, optional
        The name of the wavelet function to use. Defaults to the Haar wavelet,
        as described in Casati et al. 2004. See the documentation of PyWavelets
        for a list of available options.

    Returns
    -------
    bmse: dict
        The initialized BMSE verification object.
    """

    bmse = dict(thr=thr, wavelet=wavelet, scales=None, mse=None, eps=0, n=0)

    return bmse


def binary_mse_accum(bmse, X_f, X_o):
    """Accumulate forecast-observation pairs to an BMSE object.

    Parameters
    -----------
    bmse: dict
        The BMSE object initialized with
        :py:func:`pysteps.verification.spatialscores.binary_mse_init`.
    X_f: array_like
        Array of shape (m, n) containing the forecast field.
    X_o: array_like
        Array of shape (m, n) containing the observation field.
    """
    if not pywt_imported:
        raise MissingOptionalDependency(
            "PyWavelets package is required for the binary MSE spatial "
            "verification method but it is not installed"
        )

    if len(X_f.shape) != 2 or len(X_o.shape) != 2 or X_f.shape != X_o.shape:
        message = "X_f and X_o must be two-dimensional arrays"
        message += " having the same shape"
        raise ValueError(message)

    thr = bmse["thr"]
    wavelet = bmse["wavelet"]

    X_f = X_f.copy()
    X_f[~np.isfinite(X_f)] = thr - 1
    X_o = X_o.copy()
    X_o[~np.isfinite(X_o)] = thr - 1

    w = pywt.Wavelet(wavelet)

    I_f = (X_f >= thr).astype(float)
    I_o = (X_o >= thr).astype(float)

    E_decomp = _wavelet_decomp(I_f - I_o, w)

    n_scales = len(E_decomp)
    if bmse["scales"] is None:
        bmse["scales"] = pow(2, np.arange(n_scales))[::-1]
        bmse["mse"] = np.zeros(n_scales)

    # update eps
    eps = 1.0 * np.sum((X_o >= thr).astype(int)) / X_o.size
    if np.isfinite(eps):
        bmse["eps"] = (bmse["eps"] * bmse["n"] + eps) / (bmse["n"] + 1)

    # update mse
    for j in range(n_scales):
        mse = np.mean(E_decomp[j] ** 2)
        if np.isfinite(mse):
            bmse["mse"][j] = (bmse["mse"][j] * bmse["n"] + mse) / (bmse["n"] + 1)

    bmse["n"] += 1


def binary_mse_merge(bmse_1, bmse_2):
    """Merge two BMSE objects.

    Parameters
    ----------
    bmse_1: dict
      A BMSE object initialized with
      :py:func:`pysteps.verification.spatialscores.binary_mse_init`.
      and populated with
      :py:func:`pysteps.verification.spatialscores.binary_mse_accum`.
    bmse_2: dict
      Another BMSE object initialized with
      :py:func:`pysteps.verification.spatialscores.binary_mse_init`.
      and populated with
      :py:func:`pysteps.verification.spatialscores.binary_mse_accum`.

    Returns
    -------
    out: dict
      The merged BMSE object.
    """

    # checks
    if bmse_1["thr"] != bmse_2["thr"]:
        raise ValueError(
            "cannot merge: the thresholds are not same %s!=%s"
            % (bmse_1["thr"], bmse_2["thr"])
        )
    if bmse_1["wavelet"] != bmse_2["wavelet"]:
        raise ValueError(
            "cannot merge: the wavelets are not same %s!=%s"
            % (bmse_1["wavelet"], bmse_2["wavelet"])
        )
    if list(bmse_1["scales"]) != list(bmse_2["scales"]):
        raise ValueError(
            "cannot merge: the scales are not same %s!=%s"
            % (bmse_1["scales"], bmse_2["scales"])
        )

    # merge the BMSE objects
    bmse = bmse_1.copy()
    bmse["eps"] = (bmse["eps"] * bmse["n"] + bmse_2["eps"] * bmse_2["n"]) / (
        bmse["n"] + bmse_2["n"]
    )
    for j, scale in enumerate(bmse["scales"]):
        bmse["mse"][j] = (
            bmse["mse"][j] * bmse["n"] + bmse_2["mse"][j] * bmse_2["n"]
        ) / (bmse["n"] + bmse_2["n"])
    bmse["n"] += bmse_2["n"]

    return bmse


def binary_mse_compute(bmse, return_scales=True):
    """Compute the BMSE.

    Parameters
    ----------
    bmse: dict
        The BMSE object initialized with
        :py:func:`pysteps.verification.spatialscores.binary_mse_init`
        and accumulated with
        :py:func:`pysteps.verification.spatialscores.binary_mse_accum`.
    return_scales: bool, optional
        Whether to return the spatial scales resulting from the wavelet
        decomposition.

    Returns
    -------
    BMSE: array_like
        One-dimensional array containing the binary MSE for each spatial scale.
    scales: list, optional
        If ``return_scales=True``, return the spatial scales in pixels resulting
        from the wavelet decomposition.
    """

    scales = bmse["scales"]
    n_scales = len(scales)
    eps = bmse["eps"]

    BMSE = np.zeros(n_scales)
    for j in range(n_scales):
        mse = bmse["mse"][j]
        BMSE[j] = 1 - mse / (2 * eps * (1 - eps) / n_scales)

    BMSE[~np.isfinite(BMSE)] = np.nan

    if return_scales:
        return BMSE, scales
    else:
        return BMSE


def fss(X_f, X_o, thr, scale):
    """
    Compute the fractions skill score (FSS) for a deterministic forecast field
    and the corresponding observation field.

    Parameters
    ----------
    X_f: array_like
        Array of shape (m, n) containing the forecast field.
    X_o: array_like
        Array of shape (m, n) containing the observation field.
    thr: float
        The intensity threshold.
    scale: int
        The spatial scale in pixels. In practice, the scale represents the size
        of the moving window that it is used to compute the fraction of pixels
        above the threshold.

    Returns
    -------
    out: float
        The fractions skill score between 0 and 1.

    References
    ----------
    :cite:`RL2008`, :cite:`EWWM2013`
    """

    fss = fss_init(thr, scale)
    fss_accum(fss, X_f, X_o)
    return fss_compute(fss)


def fss_init(thr, scale):
    """Initialize a fractions skill score (FSS) verification object.

    Parameters
    ----------
    thr: float
        The intensity threshold.
    scale: float
        The spatial scale in pixels. In practice, the scale represents the size
        of the moving window that it is used to compute the fraction of pixels
        above the threshold.

    Returns
    -------
    fss: dict
        The initialized FSS verification object.
    """
    fss = dict(thr=thr, scale=scale, sum_fct_sq=0.0, sum_fct_obs=0.0, sum_obs_sq=0.0)

    return fss


def fss_accum(fss, X_f, X_o):
    """Accumulate forecast-observation pairs to an FSS object.

    Parameters
    -----------
    fss: dict
        The FSS object initialized with
        :py:func:`pysteps.verification.spatialscores.fss_init`.
    X_f: array_like
        Array of shape (m, n) containing the forecast field.
    X_o: array_like
        Array of shape (m, n) containing the observation field.
    """
    if len(X_f.shape) != 2 or len(X_o.shape) != 2 or X_f.shape != X_o.shape:
        message = "X_f and X_o must be two-dimensional arrays"
        message += " having the same shape"
        raise ValueError(message)

    X_f = X_f.copy()
    X_f[~np.isfinite(X_f)] = fss["thr"] - 1
    X_o = X_o.copy()
    X_o[~np.isfinite(X_o)] = fss["thr"] - 1

    # Convert to binary fields with the given intensity threshold
    I_f = (X_f >= fss["thr"]).astype(float)
    I_o = (X_o >= fss["thr"]).astype(float)

    # Compute fractions of pixels above the threshold within a square
    # neighboring area by applying a 2D moving average to the binary fields
    if fss["scale"] > 1:
        S_f = uniform_filter(I_f, size=fss["scale"], mode="constant", cval=0.0)
        S_o = uniform_filter(I_o, size=fss["scale"], mode="constant", cval=0.0)
    else:
        S_f = I_f
        S_o = I_o

    fss["sum_obs_sq"] += np.nansum(S_o ** 2)
    fss["sum_fct_obs"] += np.nansum(S_f * S_o)
    fss["sum_fct_sq"] += np.nansum(S_f ** 2)


def fss_merge(fss_1, fss_2):
    """Merge two FSS objects.

    Parameters
    ----------
    fss_1: dict
      A FSS object initialized with
      :py:func:`pysteps.verification.spatialscores.fss_init`.
      and populated with
      :py:func:`pysteps.verification.spatialscores.fss_accum`.
    fss_2: dict
      Another FSS object initialized with
      :py:func:`pysteps.verification.spatialscores.fss_init`.
      and populated with
      :py:func:`pysteps.verification.spatialscores.fss_accum`.

    Returns
    -------
    out: dict
      The merged FSS object.
    """

    # checks
    if fss_1["thr"] != fss_2["thr"]:
        raise ValueError(
            "cannot merge: the thresholds are not same %s!=%s"
            % (fss_1["thr"], fss_2["thr"])
        )
    if fss_1["scale"] != fss_2["scale"]:
        raise ValueError(
            "cannot merge: the scales are not same %s!=%s"
            % (fss_1["scale"], fss_2["scale"])
        )

    # merge the FSS objects
    fss = fss_1.copy()
    fss["sum_obs_sq"] += fss_2["sum_obs_sq"]
    fss["sum_fct_obs"] += fss_2["sum_fct_obs"]
    fss["sum_fct_sq"] += fss_2["sum_fct_sq"]

    return fss


def fss_compute(fss):
    """Compute the FSS.

    Parameters
    ----------
    fss: dict
       An FSS object initialized with
       :py:func:`pysteps.verification.spatialscores.fss_init`
       and accumulated with
       :py:func:`pysteps.verification.spatialscores.fss_accum`.

    Returns
    -------
    out: float
        The computed FSS value.
    """
    numer = fss["sum_fct_sq"] - 2.0 * fss["sum_fct_obs"] + fss["sum_obs_sq"]
    denom = fss["sum_fct_sq"] + fss["sum_obs_sq"]

    return 1.0 - numer / denom


import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians, hypot
from skimage.measure import regionprops_table
from pysteps.feature import tstorm as tstorm_detect
import warnings

warnings.filterwarnings(action="ignore")


def SAL(X_f, X_o,
    minref=0.1,
    maxref=150,
    mindiff=10,
    minsize=10,
    minmax=0.1,
    mindis=10,
):
    """This function calculates the components of Structure Amplitude Location (SAL) approach based on Wernli et al
    (2008). Note that we used the thunderstorm detection algorithm developed by Feldmann et al (2021) to detect precipitation objects.
    This approach uses multi-threshold algorithm to detect objects, instead of having a single threshold (f).

    Parameters
    ----------
    df_obs: 2-d ndarray for the observation data.

    df_forc: 2-d ndarray for the prediction data.

    maximum_distance: maximum distance of the study area.
    If the projection is rectangular (e.g., UTM), this value is the diagonal of the study area.
    If the projection is not rectangular (e.g., lon/lat), 'max_dist' function calculates this value.

    minref: minimum precipitation value for detecting object(s), If r star is lower than this threshold.
    The default is 0.1 mm.

    Returns
    -------
    sal:
    A dataframe with all three components of SAL.

    References
    ----------
    :cite: Wernli, H., Hofmann, C., & Zimmer, M. (2009).
    :cite: Feldmann, M., Germann, U., Gabella, M., & Berne, A. (2021).

    See also
    --------
    pysteps.feature.tstorm
    """

    if np.nanmax(X_o >= 0.1) & np.nanmax(
        X_f >= 0.1
    ):  # to avoid errors of nan values or very low precipitation
        s = s_param(X_o, X_f, minref, maxref, mindiff, minsize, minmax, mindis)
        a = Amplitude(X_o, X_f)
        l = l1_param(X_o, X_f) + l2_param(
            X_o, X_f, minref, maxref, mindiff, minsize, minmax, mindis
        )
    else:
        s = np.nan
        a = np.nan
        l = np.nan
    dic = {"S": s, "A": a, "L": l}
    sal = pd.DataFrame(dic, index=[1])
    sal.index.name = "step"
    return sal


def detect_objects_tstorm(df, minref, maxref, mindiff, minsize, minmax, mindis):
    """This function detects thunderstorms using a multi-threshold approach (Feldmann et al., 2021).
    Parameters
    df: array-like
    Array of shape (m,n) containing input data. Nan values are ignored.

    minref: float, optional
    Lower threshold for object detection. Lower values will be set to NaN. The default is 0.1.

    maxref: float, optional
    Upper threshold for object detection. Higher values will be set to this value. The default is 150.

    mindiff: float, optional
    Minimal difference between two identified maxima within same area to split area into two objects. The default is 3.

    minsize: float, optional
    Minimal area for possible detected object. The default is 4 pixels.

    minmax: float, optional
    Minimum value of maximum in identified objects. Objects with a maximum lower than this will be discarded. The default is 0.1.

    mindis: float, optional
    Minimum distance between two maxima of identified objects. Objects with a smaller distance will be merged. The default is 5 km.


    Returns
    table: pandas dataframe
    Pandas dataframe containing all detected cells and their respective properties corresponding to the input data.
    Columns of dataframe: label, area, centroid, weighted centroid, intensity_max, intensity_mean, image_intensity
    """
    _, labels = tstorm_detect.detection(
        df,
        minref=minref,
        maxref=maxref,
        mindiff=mindiff,
        minsize=minsize,
        minmax=minmax,
        mindis=mindis,
    )
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
    table = pd.DataFrame(
        regionprops_table(labels, intensity_image=df, properties=properties)
    )
    return table


def vol(ds, minref, maxref, mindiff, minsize, minmax, mindis):
    """This function calculates the scaled volume parameter based on Wernli et al (2008).

    Parameters
    ----------
    ds: 2-d ndarray data.

    minref: float, optional
    Lower threshold for object detection. Lower values will be set to NaN. The default is 0.1.

    maxref: float, optional
    Upper threshold for object detection. Higher values will be set to this value. The default is 150.

    mindiff: float, optional
    Minimal difference between two identified maxima within same area to split area into two objects. The default is 3.

    minsize: float, optional
    Minimal area for possible detected object. The default is 4 pixels.

    minmax: float, optional
    Minimum value of maximum in identified objects. Objects with a maximum lower than this will be discarded. The default is 0.1.

    mindis: float, optional
    Minimum distance between two maxima of identified objects. Objects with a smaller distance will be merged. The default is 5 km.

    Returns
    -------
    vol_value:
    A dataframe that includes precipitation characteristics (sum, max, number of wet cells, and scaled volume)
    of the input data.
    """
    ch = []
    ds_with_ob = detect_objects_tstorm(
        ds, minref, maxref, mindiff, minsize, minmax, mindis
    )

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
    vol_value = pd.DataFrame(ch)
    return vol_value


def c_m(df):
    """This function calculates the center of total (precipitation) mass in one time step.
    All nan values are replaced with 0 to calculate centroid.

    Parameters
    ----------
    ds: 2-d xarray for a time step containing input precipitation data.

    Returns
    -------
    cent:
    The coordinates of center of mass.
    """
    from scipy import ndimage

    cent = ndimage.measurements.center_of_mass(np.nan_to_num(df))
    return cent


def Amplitude(ob, pre):
    """This function calculates the amplitude component for SAL based on Wernli et al (2008).
    This component is the normalized difference of the domain-averaged precipitation in observation and forecast.
    Parameters
    ----------
    ob: 2-d ndarray for the observation data.
    pre: 2-d ndarray for the prediction data.
    max_distance: Maximum distance of the study domain in kilometers

    Returns
    -------
    a:
    Amplitude parameter which has a value between -2 to 2.
    """
    R_obs = np.nanmean(ob)
    R_model = np.nanmean(pre)
    a = (R_model - R_obs) / (0.5 * (R_model + R_obs))
    return a


def l1_param(ob, pre):
    """This function calculates the first parameter of location component for SAL based on Wernli et al (2008).
    This parameter indicates the normalized distance between the center of mass in observation and forecast.
    Parameters
    ----------
    ob: 2-d ndarray for the observation data.
    pre: 2-d ndarray for the prediction data.
    max_distance: Maximum distance of the study domain in kilometers.
    Returns
    -------
    l1:
    The first parameter of location component which has a value between 0 to 1.
    """
    maximum_distance = sqrt(((ob.shape[0]) ** 2) + ((ob.shape[1]) ** 2))
    obi = c_m(ob)
    fori = c_m(pre)
    dist = hypot(fori[1] - obi[1], fori[0] - obi[0])

    l1 = dist / maximum_distance
    return l1


def weighted_r(df, minref, maxref, mindiff, minsize, minmax, mindis):
    """This function is to calculated The weighted averaged distance between the centers of mass of the
    individual objects and the center of mass of the total precipitation field (Wernli et al, 2008).

    Parameters
    ----------
    df: 2-d xarray for a time step containing input precipitation data.


    Returns
    -------
    w_r:
    weighted averaged distance between the centers of mass of the
    individual objects and the center of mass of the total precipitation field.
    """
    df_obj = detect_objects_tstorm(df, minref, maxref, mindiff, minsize, minmax, mindis)
    centroid_total = c_m(df)
    r = []
    for i in df_obj.label - 1:
        xd = (df_obj["weighted_centroid-1"][i] - centroid_total[1]) ** 2
        yd = (df_obj["weighted_centroid-0"][i] - centroid_total[0]) ** 2

        dst = sqrt(xd + yd)
        sumr = (df_obj.image_intensity[i].sum()) * dst

        sump = df_obj.image_intensity[i].sum()

        r.append({"sum_dist": sumr, "sum_p": sump})
    rr = pd.DataFrame(r)
    w_r = rr.sum_dist.sum() / (rr.sum_p.sum())
    return w_r


def l2_param(df_obs, df_forc, minref, maxref, mindiff, minsize, minmax, mindis):
    """This function calculates the second parameter of location component for SAL based on Wernli et al (2008).

    Parameters
    ----------
    df_obs: 2-d ndarray for the observation data.

    df_pre: 2-d ndarray for the prediction data.

    minref: minimum precipitation value for detecting object(s), If r star is lower than this threshold.
    The default is 0.1 mm.

    Returns
    -------
    l2:
    The first parameter of location component which has a value between 0 to 1.
    """
    maximum_distance = sqrt(((df_obs.shape[0]) ** 2) + ((df_obs.shape[1]) ** 2))
    obs_r = (weighted_r(df_obs, minref, maxref, mindiff, minsize, minmax, mindis)) * (
        df_obs.mean()
    )
    forc_r = (weighted_r(df_forc, minref, maxref, mindiff, minsize, minmax, mindis)) * (
        df_forc.mean()
    )
    l2 = 2 * ((abs(obs_r - forc_r)) / maximum_distance)
    return float(l2)


def s_param(df_obs, df_pre, minref, maxref, mindiff, minsize, minmax, mindis):
    """This function calculates the structure component for SAL based on Wernli et al (2008).

    Parameters
    ----------
    df_obs: 2-d ndarray for the observation data.

    df_pre: 2-d ndarray for the prediction data.

    minref: minimum precipitation value for detecting object(s), If r star is lower than this threshold.
    The default is 0.1 mm.

    Returns
    -------
    S:
    The structure component which has a value between -2 to 2.
    """
    nom = (
        vol(df_pre, minref, maxref, mindiff, minsize, minmax, mindis).scaled_v.sum()
        - vol(df_obs, minref, maxref, mindiff, minsize, minmax, mindis).scaled_v.sum()
    )
    denom = (
        vol(df_pre, minref, maxref, mindiff, minsize, minmax, mindis).scaled_v.sum()
        + vol(df_obs, minref, maxref, mindiff, minsize, minmax, mindis).scaled_v.sum()
    )
    S = nom / (0.5 * (denom))
    return S


def max_dist(min_lon, min_lat, max_lon, max_lat):
    """This function calculates the maximum distance of the study area based on lon/lat coordinates.

    Parameters
    ----------
    min_lon: minimum longitude of the study area
    min_lat: minimum latitude of the study area
    max_lon: maximum longitude of the study area
    max_lat: maximum latitude of the study area

    Returns
    -------
    distance:
    Maximum distance of the study domain in kilometers.
    """
    R = 6373.0
    minx = min_lon
    miny = min_lat
    maxx = max_lon
    maxy = max_lat
    lat1 = radians(maxy)
    lon1 = radians(minx)
    lat2 = radians(miny)
    lon2 = radians(maxx)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def sal_init():
    ...

def sal_accum():
    ...

def sal_compute():
    ...


def _wavelet_decomp(X, w):
    c = pywt.wavedec2(X, w)

    X_out = []
    for k in range(len(c)):
        c_ = c[:]
        for k_ in set(range(len(c))).difference([k]):
            c_[k_] = tuple([np.zeros_like(v) for v in c[k_]])
        X_k = pywt.waverec2(c_, w)
        X_out.append(X_k)

    return X_out



