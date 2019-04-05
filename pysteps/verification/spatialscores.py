"""
pysteps.verification.spatialscores
==================================

Skill scores for spatial forecasts.

.. autosummary::
    :toctree: ../generated/

    intensity_scale
    intensity_scale_init
    intensity_scale_accum
    intensity_scale_compute
    binary_mse
    fss
"""

import numpy as np
from scipy.ndimage.filters import uniform_filter

try:
    import pywt

    pywt_imported = True
except ImportError:
    pywt_imported = False
from .. import cascade
from .. import utils
from pysteps.noise.fftgenerators import build_2D_tapering_function


def intensity_scale(X_f, X_o, name, thrs, scales=None, wavelet="Haar"):
    """Compute an intensity-scale verification score.

    Parameters
    ----------
    X_f : array_like
        Array of shape (n,m) containing the forecast field.
    X_o : array_like
        Array of shape (n,m) containing the verification observation field.
    name : string
        A string indicating the name of the spatial verification score
        to be used:

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  FSS       | Fractions skill score                                  |
        +------------+--------------------------------------------------------+
        |  BMSE      | Binary mean squared error                              |
        +------------+--------------------------------------------------------+

    thrs : sequence
        A sequence of intensity thresholds for which to compute the
        verification.
    scales : sequence, optional
        A sequence of spatial scales in pixels to be used in the FSS.
    wavelet : str, optional
        The name of the wavelet function to use in the BMSE.
        Defaults to the Haar wavelet, as described in Casati et al. 2004.
        See the documentation of PyWavelets for a list of available options.

    Returns
    -------
    out : array_like
        The two-dimensional array containing the intensity-scale skill scores
        for each spatial scale and intensity threshold.
    """

    X_f = X_f.copy()
    X_o = X_o.copy()
    intscale = intensity_scale_init(name, thrs, scales, wavelet)
    intensity_scale_accum(intscale, X_f, X_o)
    return intensity_scale_compute(intscale)


def intensity_scale_init(name, thrs, scales=None, wavelet="Haar"):
    """Initialize an intensty-scale verification object.

    Parameters
    ----------
    name : string
        A string indicating the name of the spatial verification score
        to be used:

        +------------+--------------------------------------------------------+
        | Name       | Description                                            |
        +============+========================================================+
        |  FSS       | Fractions skill score                                  |
        +------------+--------------------------------------------------------+
        |  BMSE      | Binary mean squared error                              |
        +------------+--------------------------------------------------------+

    thrs : sequence
        A sequence of intensity thresholds for which to compute the
        verification.
    scales : sequence, optional
        A sequence of spatial scales in pixels to be used in the FSS.
    wavelet : str, optional
        The name of the wavelet function to use in the BMSE.
        Defaults to the Haar wavelet, as described in
        Casati et al. 2004. See the documentation of
        PyWavelets for a list of available options.

    Returns
    -------
    out : dict
        The intensity-scale object.

    """
    intscale = {}
    intscale["name"] = name
    intscale["SS"] = None
    intscale["thrs"] = thrs[:]
    intscale["scales"] = scales
    intscale["wavelet"] = wavelet
    intscale["n"] = None
    intscale["shape"] = None
    if name.lower() == "fss":
        intscale["label"] = "Fractions skill score"
        del intscale["wavelet"]
    if name.lower() == "bmse":
        intscale["label"] = "Binary MSE skill score"
        intscale["scales"] = None
    if name.lower() == "fss" and scales is None:
        message = "a sequence of scales must be provided for the FSS,"
        message += " but %s was passed" % scales
        raise ValueError(message)
    return intscale


def intensity_scale_accum(intscale, X_f, X_o):
    """Compute and update the intensity-scale verification scores.

    Parameters
    ----------
    intscale : dict
        The intensity-scale object.
    X_f : array_like
        Array of shape (n,m) containing the forecast field.
    X_o : array_like
        Array of shape (n,m) containing the verification observation field.
    """
    if len(X_f.shape) != 2 or len(X_o.shape) != 2 or X_f.shape != X_o.shape:
        message = "X_f and X_o must be two-dimensional arrays"
        message += " having the same shape, but "
        message += "X_f = %s and X_o = %s" % (str(X_f.shape), str(X_o.shape))
        raise ValueError(message)

    if intscale["shape"] is not None and X_f.shape != intscale["shape"]:
        message = "X_f and X_o shapes do not match the shape"
        message += " of the intensity-scale object"
        raise ValueError(message)

    if intscale["shape"] is None:
        intscale["shape"] = X_f.shape

    thrs = intscale["thrs"]
    thr_min = np.min(thrs)
    n_thrs = len(thrs)

    X_f = X_f.copy()
    X_f[~np.isfinite(X_f)] = thr_min - 1
    X_o = X_o.copy()
    X_o[~np.isfinite(X_o)] = thr_min - 1

    if intscale["name"].lower() == "bmse":
        SS = None
        n_thrs = len(thrs)
        for i in range(n_thrs):
            SS_, scales = binary_mse(X_f, X_o, thrs[i], intscale["wavelet"])
            if SS is None:
                SS = np.empty((SS_.size, n_thrs))
            SS[:, i] = SS_
        if intscale["scales"] is None:
            intscale["scales"] = scales
    elif intscale["name"].lower() == "fss":
        scales = intscale["scales"]
        n_scales = len(scales)
        SS = np.empty((n_scales, n_thrs))
        for i in range(n_thrs):
            for j in range(n_scales):
                SS[j, i] = fss(X_f, X_o, thrs[i], scales[j])
    else:
        raise ValueError("unknown method %s" % intscale["name"])

    # update scores
    if intscale["n"] is None:
        intscale["n"] = np.ones(SS.shape, dtype=int)
    intscale["n"] += (~np.isnan(SS)).astype(int)

    if intscale["SS"] is None:
        intscale["SS"] = SS
    else:
        intscale["SS"] += np.nansum((SS, -1 * intscale["SS"]), axis=0) / intscale["n"]


def intensity_scale_compute(intscale):
    """Return the intensity scale matrix.

    Parameters
    ----------
    intscale : dict
        The intensity-scale object.

    Returns
    -------
    out : array_like
        The two-dimensional array containing the intensity-scale skill scores
        for each given spatial scale and intensity threshold.

    """
    return intscale["SS"]


def binary_mse(X_f, X_o, thr, wavelet="haar"):
    """Compute an intensity-scale verification as the MSE of the binary error.

    This method uses PyWavelets for decomposing the error field between the
    forecasts and observations into multiple spatial scales.

    Parameters
    ----------
    X_f : array_like
        Array of shape (n,m) containing the forecast field.
    X_o : array_like
        Array of shape (n,m) containing the verification observation field.
    thr : sequence
        The intensity threshold for which to compute the verification.
    wavelet : str, optional
        The name of the wavelet function to use. Defaults to the Haar wavelet,
        as described in Casati et al. 2004. See the documentation of PyWavelets
        for a list of available options.

    Returns
    -------
    SS : array
        One-dimensional array containing the binary MSE for each spatial scale.
    spatial_scale : list

    References
    ----------
    :cite:`CRS2004`

    """
    if len(X_f.shape) != 2 or len(X_o.shape) != 2 or X_f.shape != X_o.shape:
        message = "X_f and X_o must be two-dimensional arrays"
        message += " having the same shape"
        raise ValueError(message)

    X_f = X_f.copy()
    X_f[~np.isfinite(X_f)] = thr - 1
    X_o = X_o.copy()
    X_o[~np.isfinite(X_o)] = thr - 1

    w = pywt.Wavelet(wavelet)

    SS = None

    I_f = (X_f >= thr).astype(float)
    I_o = (X_o >= thr).astype(float)

    E_decomp = _wavelet_decomp(I_f - I_o, w)
    n_scales = len(E_decomp)

    eps = 1.0 * np.sum((X_o >= thr).astype(int)) / np.size(X_o)

    SS = np.empty((n_scales))
    for j in range(n_scales):
        mse = np.mean(E_decomp[j] ** 2)
        SS[j] = 1 - mse / (2 * eps * (1 - eps) / n_scales)
    SS[~np.isfinite(SS)] = np.nan

    scales = pow(2, np.arange(SS.size))[::-1]

    return SS, scales


def fss(X_f, X_o, thr, scale):
    """
    Compute the fractions skill score (FSS) for a deterministic forecast
    field and the corresponding observation.

    Parameters
    ----------
    X_f : array_like
        Array of shape (n,m) containing the forecast field.
    X_o : array_like
        Array of shape (n,m) containing the reference field (observation).
    thr : float
        Intensity threshold.
    scale : int
        The spatial scale  in px. In practice they represent the size of the
        moving window that it is used to compute the fraction of pixels above
        the threshold.

    Returns
    -------
    out : float
        The fractions skill score between 0 and 1.

    References
    ----------
    :cite:`RL2008`, :cite:`EWWM2013`

    """
    if len(X_f.shape) != 2 or len(X_o.shape) != 2 or X_f.shape != X_o.shape:
        message = "X_f and X_o must be two-dimensional arrays"
        message += " having the same shape"
        raise ValueError(message)

    X_f = X_f.copy()
    X_f[~np.isfinite(X_f)] = thr - 1
    X_o = X_o.copy()
    X_o[~np.isfinite(X_o)] = thr - 1
    X_f.size

    # Convert to binary fields with the intensity threshold
    I_f = (X_f >= thr).astype(float)
    I_o = (X_o >= thr).astype(float)

    # Compute fractions of pixels above the threshold within a square
    # neighboring area by applying a 2D moving average to the binary fields
    S_f = uniform_filter(I_f, size=int(scale), mode="constant", cval=0.0)
    S_o = uniform_filter(I_o, size=int(scale), mode="constant", cval=0.0)

    # Compute the numerator
    n = X_f.size
    N = 1.0 * np.sum((S_o - S_f) ** 2) / n
    # Compute the denominator
    D = 1.0 * (np.sum(S_o ** 2) + np.nansum(S_f ** 2)) / n

    return 1 - N / D


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
