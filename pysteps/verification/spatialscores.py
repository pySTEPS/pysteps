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
from scipy.ndimage import uniform_filter

from pysteps.exceptions import MissingOptionalDependency
from pysteps.verification.salscores import sal  # make SAL accessible from this module

try:
    import pywt

    pywt_imported = True
except ImportError:
    pywt_imported = False


def intensity_scale(X_f, X_o, name, thrs, scales=None, wavelet="Haar"):
    """
    Compute an intensity-scale verification score.

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
    """
    Initialize an intensity-scale verification object.

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
    """
    Compute and update the intensity-scale verification scores.

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
    """
    Merge two intensity-scale verification objects.

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
    """
    Return the intensity scale matrix.

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
    """
    Compute the MSE of the binary error as a function of spatial scale.

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
    """
    Initialize a binary MSE (BMSE) verification object.

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
    """
    Merge two BMSE objects.

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
    """
    Compute the BMSE.

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
    """
    Initialize a fractions skill score (FSS) verification object.

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

    fss["sum_obs_sq"] += np.nansum(S_o**2)
    fss["sum_fct_obs"] += np.nansum(S_f * S_o)
    fss["sum_fct_sq"] += np.nansum(S_f**2)


def fss_merge(fss_1, fss_2):
    """
    Merge two FSS objects.

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
    """
    Compute the FSS.

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


def fss_ens(fcst, obs, thr, scales, version='fss'):
    """
    Compute Fractions Skill Score (FSS) for ensemble or deterministic forecasts
    across multiple thresholds and spatial scales.

    Parameters
    ----------
    fcst : np.ndarray
        Forecast field. Shape can be:
        - (n_members, n_leadtimes, y, x)
        - (n_leadtimes, y, x)
        - (n_members, y, x)
        - (y, x)
    obs : np.ndarray
        Observation field. Shape can be:
        - (n_leadtimes, y, x)
        - (y, x)
    thr : float or list of float
        Threshold(s) used to binarize forecast and observation fields.
    scales : list or np.ndarray
        List of spatial scales (window sizes) to apply smoothing.
    version : str
        Type of FSS computation:
        - 'fss'   : member-wise FSS
        - 'avfss' : average of member-wise FSS
        - 'pfss'  : FSS of ensemble mean probability
        - 'agfss' : FSS of aggregated ensemble fields

    Returns
    -------
    fss_scores : np.ndarray
        FSS scores with shape:
        - (n_thresholds, n_scales, n_members, n_leadtimes) for 'fss'
        - (n_thresholds, n_scales, 1, n_leadtimes) for ensemble versions
    """
    version = version.lower()
    scales = np.atleast_1d(scales)
    thr = np.atleast_1d(thr)

    # Normalize input shapes
    fcst, obs = normalize_forecast_obs(fcst, obs)
    n_members, n_leadtimes, y, x = fcst.shape
    n_scales = len(scales)
    n_thresholds = len(thr)

    # Initialize output array
    if version == 'fss':
        fss_scores = np.empty((n_thresholds, n_scales, n_members, n_leadtimes))
    else:
        fss_scores = np.empty((n_thresholds, n_scales, 1, n_leadtimes))

    # Loop over thresholds and lead times
    for i, thr_val in enumerate(thr):
        for t in range(n_leadtimes):
            obs_bin = (obs[t] >= thr_val).astype(float)

            if version in ['fss', 'avfss']:
                for m in range(n_members):
                    fcst_bin = (fcst[m, t] >= thr_val).astype(float)
                    for s, scale in enumerate(scales):
                        num, denom = _compute_fss_components(fcst_bin, obs_bin, scale)
                        fss_scores[i, s, m, t] = _compute_fss_score(num, denom)

                if version == 'avfss':
                    for s in range(n_scales):
                        fss_scores[i, s, 0, t] = fss_scores[i, s, :, t].mean()

            elif version == 'pfss':
                fcst_bin = (fcst[:, t] >= thr_val).astype(float)
                fcst_mean = np.mean(fcst_bin, axis=0)
                for s, scale in enumerate(scales):
                    num, denom = _compute_fss_components(fcst_mean, obs_bin, scale)
                    fss_scores[i, s, 0, t] = _compute_fss_score(num, denom)

            elif version == 'agfss':
                for s, scale in enumerate(scales):
                    num, denom = 0.0, 0.0
                    for m in range(n_members):
                        fcst_bin = (fcst[m, t] >= thr_val).astype(float)
                        n, d = _compute_fss_components(fcst_bin, obs_bin, scale)
                        num += n
                        denom += d
                    fss_scores[i, s, 0, t] = _compute_fss_score(num, denom)

            else:
                raise ValueError("Invalid version. Choose from 'fss', 'pfss', 'agfss', 'avfss'.")

    return fss_scores


def normalize_forecast_obs(pred, obs):
    """
    Normalize forecast and observation arrays to shape:
    - pred: (n_members, n_leadtimes, y, x)
    - obs : (n_leadtimes, y, x)

    Parameters
    ----------
    pred : np.ndarray
        Forecast array with shape:
        - (y, x)
        - (n_members, y, x)
        - (n_leadtimes, y, x)
        - (n_members, n_leadtimes, y, x)
    obs : np.ndarray
        Observation array with shape:
        - (y, x)
        - (n_leadtimes, y, x)

    Returns
    -------
    pred : np.ndarray
        Normalized forecast array of shape (n_members, n_leadtimes, y, x)
    obs : np.ndarray
        Normalized observation array of shape (n_leadtimes, y, x)
    """
    pred = np.asarray(pred)
    obs = np.asarray(obs)

    # Normalize observation shape
    if obs.ndim == 2:  # (y, x)
        obs = obs[np.newaxis, ...]
    elif obs.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported observation shape: {obs.shape}")

    n_leadtimes = obs.shape[0]

    # Normalize forecast shape
    if pred.ndim == 2:  # (y, x)
        pred = pred[np.newaxis, np.newaxis, ...]
    elif pred.ndim == 3:
        if pred.shape[0] == n_leadtimes:  # (n_leadtimes, y, x)
            pred = pred[np.newaxis, ...]
        else:  # (n_members, y, x)
            pred = pred[:, np.newaxis, ...]
    elif pred.ndim == 4:
        if pred.shape[1] != n_leadtimes:
            raise ValueError(f"Forecast lead time dimension {pred.shape[1]} does not match obs {n_leadtimes}")
    else:
        raise ValueError(f"Unsupported forecast shape: {pred.shape}")

    return pred, obs


def smooth_fields(fcst_bin, obs_bin, scale):
    """
    Apply uniform smoothing to forecast and observation binary fields.

    Parameters
    ----------
    fcst_bin : np.ndarray
        Binary forecast field.
    obs_bin : np.ndarray
        Binary observation field.
    scale : int
        Size of the smoothing window.

    Returns
    -------
    S_f : np.ndarray
        Smoothed forecast field.
    S_o : np.ndarray
        Smoothed observation field.
    """
    if scale > 1:
        S_f = uniform_filter(fcst_bin, size=scale, mode="constant", cval=0.0)
        S_o = uniform_filter(obs_bin, size=scale, mode="constant", cval=0.0)
    else:
        S_f = fcst_bin
        S_o = obs_bin
    return S_f, S_o


def _compute_fss_terms(S_f, S_o):
    """
    Compute numerator and denominator for FSS from smoothed fields.

    Parameters
    ----------
    S_f : np.ndarray
        Smoothed forecast field.
    S_o : np.ndarray
        Smoothed observation field.

    Returns
    -------
    num : float
        Sum of squared differences.
    denom : float
        Sum of squared values.
    """
    num = np.sum((S_f - S_o) ** 2)
    denom = np.sum(S_f ** 2 + S_o ** 2)
    return num, denom


def _compute_fss_components(fcst_bin, obs_bin, scale):
    """
    Compute numerator and denominator for the Fractions Skill Score (FSS)
    at a given spatial scale using smoothed binary fields.

    Parameters
    ----------
    fcst_bin : np.ndarray
        Binary forecast field.
    obs_bin : np.ndarray
        Binary observation field.
    scale : int
        Size of the smoothing window.

    Returns
    -------
    num : float
        Sum of squared differences.
    denom : float
        Sum of squared values.
    """
    S_f, S_o = smooth_fields(fcst_bin, obs_bin, scale)
    return _compute_fss_terms(S_f, S_o)


def _compute_fss_score(num, denom, eps=1e-10):
    """
    Compute the Fractions Skill Score (FSS) from numerator and denominator.

    Parameters
    ----------
    num : float
        Numerator of the FSS formula (error term).
    denom : float
        Denominator of the FSS formula (normalization term).
    eps : float, optional
        Small constant to avoid division by zero (default is 1e-10).

    Returns
    -------
    fss : float
        FSS score ranging from 0 (no skill) to 1 (perfect match).
    """
    return 1.0 - num / (denom + eps)


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
