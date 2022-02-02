# -*- coding: utf-8 -*-
"""
pysteps.utils.transformation
============================

Methods for transforming data values.

.. autosummary::
    :toctree: ../generated/

    boxcox_transform
    dB_transform
    NQ_transform
    sqrt_transform
"""

import numpy as np
import scipy.stats as scipy_stats
import warnings
from scipy.interpolate import interp1d

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # To deactivate warnings for comparison operators with NaNs


def boxcox_transform(
    R, metadata=None, Lambda=None, threshold=None, zerovalue=None, inverse=False
):
    """
    The one-parameter Box-Cox transformation.

    The Box-Cox transform is a well-known power transformation introduced by
    Box and Cox (1964). In its one-parameter version, the Box-Cox transform
    takes the form T(x) = ln(x) for Lambda = 0,
    or T(x) = (x**Lambda - 1)/Lambda otherwise.

    Default parameters will produce a log transform (i.e. Lambda=0).

    Parameters
    ----------
    R: array-like
        Array of any shape to be transformed.
    metadata: dict, optional
        Metadata dictionary containing the transform, zerovalue and threshold
        attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    Lambda: float, optional
        Parameter Lambda of the Box-Cox transformation.
        It is 0 by default, which produces the log transformation.

        Choose Lambda < 1 for positively skewed data, Lambda > 1 for negatively
        skewed data.
    threshold: float, optional
        The value that is used for thresholding with the same units as R.
        If None, the threshold contained in metadata is used.
        If no threshold is found in the metadata,
        a value of 0.1 is used as default.
    zerovalue: float, optional
        The value to be assigned to no rain pixels as defined by the threshold.
        It is equal to the threshold - 1 by default.
    inverse: bool, optional
        If set to True, it performs the inverse transform. False by default.

    Returns
    -------
    R: array-like
        Array of any shape containing the (back-)transformed units.
    metadata: dict
        The metadata with updated attributes.

    References
    ----------
    Box, G. E. and Cox, D. R. (1964), An Analysis of Transformations. Journal
    of the Royal Statistical Society: Series B (Methodological), 26: 211-243.
    doi:10.1111/j.2517-6161.1964.tb00553.x
    """

    R = R.copy()

    if metadata is None:
        if inverse:
            metadata = {"transform": "BoxCox"}
        else:
            metadata = {"transform": None}

    else:
        metadata = metadata.copy()

    if not inverse:

        if metadata["transform"] == "BoxCox":
            return R, metadata

        if Lambda is None:
            Lambda = metadata.get("BoxCox_lambda", 0.0)

        if threshold is None:
            threshold = metadata.get("threshold", 0.1)

        zeros = R < threshold

        # Apply Box-Cox transform
        if Lambda == 0.0:
            R[~zeros] = np.log(R[~zeros])
            threshold = np.log(threshold)

        else:
            R[~zeros] = (R[~zeros] ** Lambda - 1) / Lambda
            threshold = (threshold**Lambda - 1) / Lambda

        # Set value for zeros
        if zerovalue is None:
            zerovalue = threshold - 1  # TODO: set to a more meaningful value
        R[zeros] = zerovalue

        metadata["transform"] = "BoxCox"
        metadata["BoxCox_lambda"] = Lambda
        metadata["zerovalue"] = zerovalue
        metadata["threshold"] = threshold

    elif inverse:

        if metadata["transform"] not in ["BoxCox", "log"]:
            return R, metadata

        if Lambda is None:
            Lambda = metadata.pop("BoxCox_lambda", 0.0)
        if threshold is None:
            threshold = metadata.get("threshold", -10.0)
        if zerovalue is None:
            zerovalue = 0.0

        # Apply inverse Box-Cox transform
        if Lambda == 0.0:
            R = np.exp(R)
            threshold = np.exp(threshold)

        else:
            R = np.exp(np.log(Lambda * R + 1) / Lambda)
            threshold = np.exp(np.log(Lambda * threshold + 1) / Lambda)

        R[R < threshold] = zerovalue

        metadata["transform"] = None
        metadata["zerovalue"] = zerovalue
        metadata["threshold"] = threshold

    return R, metadata


def dB_transform(R, metadata=None, threshold=None, zerovalue=None, inverse=False):
    """Methods to transform precipitation intensities to/from dB units.

    Parameters
    ----------
    R: array-like
        Array of any shape to be (back-)transformed.
    metadata: dict, optional
        Metadata dictionary containing the transform, zerovalue and threshold
        attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    threshold: float, optional
        Optional value that is used for thresholding with the same units as R.
        If None, the threshold contained in metadata is used.
        If no threshold is found in the metadata,
        a value of 0.1 is used as default.
    zerovalue: float, optional
        The value to be assigned to no rain pixels as defined by the threshold.
        It is equal to the threshold - 1 by default.
    inverse: bool, optional
        If set to True, it performs the inverse transform. False by default.

    Returns
    -------
    R: array-like
        Array of any shape containing the (back-)transformed units.
    metadata: dict
        The metadata with updated attributes.
    """

    R = R.copy()

    if metadata is None:
        if inverse:
            metadata = {"transform": "dB"}
        else:
            metadata = {"transform": None}

    else:
        metadata = metadata.copy()

    # to dB units
    if not inverse:

        if metadata["transform"] == "dB":
            return R, metadata

        if threshold is None:
            threshold = metadata.get("threshold", 0.1)

        zeros = R < threshold

        # Convert to dB
        R[~zeros] = 10.0 * np.log10(R[~zeros])
        threshold = 10.0 * np.log10(threshold)

        # Set value for zeros
        if zerovalue is None:
            zerovalue = threshold - 5  # TODO: set to a more meaningful value
        R[zeros] = zerovalue

        metadata["transform"] = "dB"
        metadata["zerovalue"] = zerovalue
        metadata["threshold"] = threshold

        return R, metadata

    # from dB units
    elif inverse:

        if metadata["transform"] != "dB":
            return R, metadata

        if threshold is None:
            threshold = metadata.get("threshold", -10.0)
        if zerovalue is None:
            zerovalue = 0.0

        R = 10.0 ** (R / 10.0)
        threshold = 10.0 ** (threshold / 10.0)
        R[R < threshold] = zerovalue

        metadata["transform"] = None
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

        return R, metadata


def NQ_transform(R, metadata=None, inverse=False, **kwargs):
    """
    The normal quantile transformation as in Bogner et al (2012).
    Zero rain vales are set to zero in norm space.

    Parameters
    ----------
    R: array-like
        Array of any shape to be transformed.
    metadata: dict, optional
        Metadata dictionary containing the transform, zerovalue and threshold
        attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    inverse: bool, optional
        If set to True, it performs the inverse transform. False by default.

    Other Parameters
    ----------------
    a: float, optional
        The offset fraction to be used for plotting positions;
        typically in (0,1).
        The default is 0., that is, it spaces the points evenly in the uniform
        distribution.

    Returns
    -------
    R: array-like
        Array of any shape containing the (back-)transformed units.
    metadata: dict
        The metadata with updated attributes.

    References
    ----------
    Bogner, K., Pappenberger, F., and Cloke, H. L.: Technical Note: The normal
    quantile transformation and its application in a flood forecasting system,
    Hydrol. Earth Syst. Sci., 16, 1085-1094,
    https://doi.org/10.5194/hess-16-1085-2012, 2012.
    """

    # defaults
    a = kwargs.get("a", 0.0)

    R = R.copy()
    shape0 = R.shape
    R = R.ravel().astype(float)
    idxNan = np.isnan(R)
    R_ = R[~idxNan]

    if metadata is None:
        if inverse:
            metadata = {"transform": "NQT"}
        else:
            metadata = {"transform": None}
        metadata["zerovalue"] = np.min(R_)

    else:
        metadata = metadata.copy()

    if not inverse:
        # Plotting positions
        # https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Plotting_position
        n = R_.size
        Rpp = ((np.arange(n) + 1 - a) / (n + 1 - 2 * a)).reshape(R_.shape)

        # NQ transform
        Rqn = scipy_stats.norm.ppf(Rpp)
        R__ = np.interp(R_, R_[np.argsort(R_)], Rqn)

        # set zero rain to 0 in norm space
        R__[R[~idxNan] == metadata["zerovalue"]] = 0

        # build inverse transform
        metadata["inqt"] = interp1d(
            Rqn, R_[np.argsort(R_)], bounds_error=False, fill_value=(R_.min(), R_.max())
        )

        metadata["transform"] = "NQT"
        metadata["zerovalue"] = 0
        metadata["threshold"] = R__[R__ > 0].min()

    else:
        f = metadata.pop("inqt")
        R__ = f(R_)
        metadata["transform"] = None
        metadata["zerovalue"] = R__.min()
        metadata["threshold"] = R__[R__ > R__.min()].min()

    R[~idxNan] = R__

    return R.reshape(shape0), metadata


def sqrt_transform(R, metadata=None, inverse=False, **kwargs):
    """
    Square-root transform.

    Parameters
    ----------
    R: array-like
        Array of any shape to be transformed.
    metadata: dict, optional
        Metadata dictionary containing the transform, zerovalue and threshold
        attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.
    inverse: bool, optional
        If set to True, it performs the inverse transform. False by default.

    Returns
    -------
    R: array-like
        Array of any shape containing the (back-)transformed units.
    metadata: dict
        The metadata with updated attributes.

    """

    R = R.copy()

    if metadata is None:
        if inverse:
            metadata = {"transform": "sqrt"}
        else:
            metadata = {"transform": None}
        metadata["zerovalue"] = np.nan
        metadata["threshold"] = np.nan
    else:
        metadata = metadata.copy()

    if not inverse:
        # sqrt transform
        R = np.sqrt(R)

        metadata["transform"] = "sqrt"
        metadata["zerovalue"] = np.sqrt(metadata["zerovalue"])
        metadata["threshold"] = np.sqrt(metadata["threshold"])
    else:
        # inverse sqrt transform
        R = R**2

        metadata["transform"] = None
        metadata["zerovalue"] = metadata["zerovalue"] ** 2
        metadata["threshold"] = metadata["threshold"] ** 2

    return R, metadata
