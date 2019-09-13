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
from scipy.interpolate import interp1d
import warnings

try:
    import xarray as xr

    XARRAY_IMPORTED = True
except ImportError:
    XARRAY_IMPORTED = False

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # To deactivate warnings for comparison operators with NaNs


def boxcox_transform(
    R,
    metadata=None,
    Lambda=None,
    threshold=None,
    zerovalue=None,
    inverse=False,
):
    """The one-parameter Box-Cox transformation.

    The Box-Cox transform is a well-known power transformation introduced by
    Box and Cox (1964). In its one-parameter version, the Box-Cox transform
    takes the form T(x) = ln(x) for Lambda = 0,
    or T(x) = (x**Lambda - 1)/Lambda otherwise.

    Default parameters will produce a log transform (i.e. Lambda=0).

    Parameters
    ----------

    R : array-like or xarray.DataArray
        Array of any shape to be transformed.

    metadata : dict, optional
        Metadata dictionary containing the transform, zerovalue and threshold
        attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.

    Lambda : float, optional
        Parameter Lambda of the Box-Cox transformation.
        It is 0 by default, which produces the log transformation.

        Choose Lambda < 1 for positively skewed data, Lambda > 1 for negatively
        skewed data.

    threshold : float, optional
        The value that is used for thresholding with the same units as R.
        If None, the threshold contained in metadata is used.
        If no threshold is found in the metadata,
        a value of 0.1 is used as default.

    zerovalue : float, optional
        The value to be assigned to no rain pixels as defined by the threshold.
        It is equal to the threshold - 1 by default.

    inverse : bool, optional
        If set to True, it performs the inverse transform. False by default.

    Returns
    -------

    R : array-like or xarray.DataArray
        Array of any shape containing the (back-)transformed units.

    metadata : dict
        The metadata with updated attributes.

    References
    ----------
    Box, G. E. and Cox, D. R. (1964), An Analysis of Transformations. Journal
    of the Royal Statistical Society: Series B (Methodological), 26: 211-243.
    doi:10.1111/j.2517-6161.1964.tb00553.x
    """

    if XARRAY_IMPORTED and isinstance(R, xr.DataArray):
        R = R.copy()
        metadata = R.attrs
        isxarray = True

    else:
        R = np.copy(R)
        isxarray = False

        if metadata is None:
            if inverse:
                metadata = {"transform": "BoxCox"}
            else:
                metadata = {"transform": None}

        else:
            metadata = metadata.copy()

    if not inverse:

        if metadata["transform"] == "BoxCox":
            if isxarray:
                return R
            else:
                return R, metadata

        if Lambda is None:
            Lambda = metadata.get("BoxCox_lambda", 0.0)

        if threshold is None:
            threshold = metadata.get("threshold", 0.1)

        zeros = R < threshold

        # Apply Box-Cox transform
        if Lambda == 0.0:
            if isxarray:
                R = np.log(R.where(~zeros))
            else:
                R[~zeros] = np.log(R[~zeros])
            threshold = np.log(threshold)

        else:
            if isxarray:
                R = (R.where(~zeros) ** Lambda - 1) / Lambda
            else:
                R[~zeros] = (R[~zeros] ** Lambda - 1) / Lambda
            threshold = (threshold ** Lambda - 1) / Lambda

        # Set value for zeros
        if zerovalue is None:
            if isxarray:
                dr = (
                    float(float(R.where(R > threshold).min().load()))
                    - threshold
                )
            else:
                dr = R[R > threshold].min() - threshold
            zerovalue = threshold - dr  # TODO: set to a more meaningful value

        if isxarray:
            R = xr.where(zeros, zerovalue, R)
        else:
            R[zeros] = zerovalue

        metadata["transform"] = "BoxCox"
        metadata["BoxCox_lambda"] = Lambda
        metadata["zerovalue"] = zerovalue
        metadata["threshold"] = threshold

    elif inverse:

        if metadata["transform"] not in ["BoxCox", "log"]:
            if isxarray:
                return R
            else:
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

        if isxarray:
            R = xr.where(R < threshold, zerovalue, R)
        else:
            R[R < threshold] = zerovalue

        metadata["transform"] = None
        metadata["zerovalue"] = zerovalue
        metadata["threshold"] = threshold

    if isxarray:
        R.attrs.update(metadata)
        return R

    return R, metadata


def dB_transform(
    R, metadata=None, threshold=None, zerovalue=None, inverse=False
):
    """Methods to transform precipitation intensities to/from dB units.

    Parameters
    ----------

    R : array-like or xarray.DataArray
        Array of any shape to be (back-)transformed.

    metadata : dict, optional
        Metadata dictionary containing the transform, zerovalue and threshold
        attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.

    threshold : float, optional
        Optional value that is used for thresholding with the same units as R.
        If None, the threshold contained in metadata is used.
        If no threshold is found in the metadata,
        a value of 0.1 is used as default.

    zerovalue : float, optional
        The value to be assigned to no rain pixels as defined by the threshold.
        It is equal to the threshold - 1 by default.

    inverse : bool, optional
        If set to True, it performs the inverse transform. False by default.

    Returns
    -------

    R : array-like or xarray.DataArray
        Array of any shape containing the (back-)transformed units.

    metadata : dict
        The metadata with updated attributes.
    """

    if XARRAY_IMPORTED and isinstance(R, xr.DataArray):
        R = R.copy()
        metadata = R.attrs
        isxarray = True

    else:
        R = np.copy(R)
        isxarray = False

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
            if isxarray:
                return R
            else:
                return R, metadata

        if threshold is None:
            threshold = metadata.get("threshold", 0.1)

        zeros = R < threshold

        # Convert to dB
        if isxarray:
            R = 10.0 * np.log10(R.where(~zeros))
        else:
            R[~zeros] = 10.0 * np.log10(R[~zeros])
        threshold = 10.0 * np.log10(threshold)

        # Set value for zeros
        if zerovalue is None:
            if isxarray:
                dr = (
                    float(float(R.where(R > threshold).min().load()))
                    - threshold
                )
            else:
                dr = R[R > threshold].min() - threshold
            zerovalue = threshold - dr  # TODO: set to a more meaningful value

        if isxarray:
            R = xr.where(zeros, zerovalue, R)
        else:
            R[zeros] = zerovalue

        metadata["transform"] = "dB"
        metadata["zerovalue"] = zerovalue
        metadata["threshold"] = threshold

    # from dB units
    elif inverse:

        if metadata["transform"] != "dB":
            if isxarray:
                return R
            else:
                return R, metadata

        if threshold is None:
            threshold = metadata.get("threshold", -10.0)

        if zerovalue is None:
            zerovalue = 0.0

        R = 10.0 ** (R / 10.0)
        threshold = 10.0 ** (threshold / 10.0)
        if isxarray:
            R = xr.where(R < threshold, zerovalue, R)
        else:
            R[R < threshold] = zerovalue

        metadata["transform"] = None
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    if isxarray:
        R.attrs.update(metadata)
        return R

    return R, metadata


def NQ_transform(R, metadata=None, inverse=False, **kwargs):
    """The normal quantile transformation as in Bogner et al (2012).
    Zero rain vales are set to zero in norm space.

    Parameters
    ----------

    R : array-like or xarray.DataArray
        Array of any shape to be transformed.

    metadata : dict, optional
        Metadata dictionary containing the transform, zerovalue and threshold
        attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.

    inverse : bool, optional
        If set to True, it performs the inverse transform. False by default.

    Other Parameters
    ----------------

    a : float, optional
        The offset fraction to be used for plotting positions;
        typically in (0,1).
        The default is 0., that is, it spaces the points evenly in the uniform
        distribution.

    Returns
    -------

    R : array-like or xarray.DataArray
        Array of any shape containing the (back-)transformed units.

    metadata : dict
        The metadata with updated attributes.

    References
    ----------
    Bogner, K., Pappenberger, field., and Cloke, H. L.: Technical Note: The normal
    quantile transformation and its application in a flood forecasting system,
    Hydrol. Earth Syst. Sci., 16, 1085-1094,
    https://doi.org/10.5194/hess-16-1085-2012, 2012.
    """

    if XARRAY_IMPORTED and isinstance(R, xr.DataArray):
        R = R.copy()
        metadata = R.attrs
        array = R.load().data

        isxarray = True

    else:
        array = np.copy(R)

        isxarray = False

        if metadata is None:
            if inverse:
                metadata = {"transform": "NQT"}
            else:
                metadata = {"transform": None}

        else:
            metadata = metadata.copy()

    shape0 = array.shape
    array = array.ravel().astype(float)
    isfinite = np.isfinite(array)
    wet = array >= metadata["threshold"]
    idx = np.logical_and(isfinite, wet)
    array_wet = array[idx]

    if not inverse:

        # Plotting positions
        # https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Plotting_position

        # Default offset fraction
        a = kwargs.get("a", 0.0)

        n = array_wet.size
        pp = ((np.arange(n) + 1 - a) / (n + 1 - 2 * a)).reshape(
            array_wet.shape
        )

        # NQ transform
        qn = scipy_stats.norm.ppf(pp)
        narray_wet = np.interp(array_wet, array_wet[np.argsort(array_wet)], qn)

        # build inverse transform
        metadata["inqt"] = interp1d(
            qn,
            array_wet[np.argsort(array_wet)],
            bounds_error=False,
            fill_value=(array_wet.min(), array_wet.max()),
        )

        threshold = narray_wet.min()
        dr = narray_wet[narray_wet > threshold].min() - threshold
        zerovalue = threshold - dr  # TODO: set to a more meaningful value

        metadata["transform"] = "NQT"
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    else:

        f = metadata.pop("inqt")
        narray_wet = f(array_wet)
        metadata["transform"] = None
        metadata["zerovalue"] = 0
        metadata["threshold"] = narray_wet.min()

    array[idx] = narray_wet
    array[~wet] = narray_wet.min()
    array[~isfinite] = np.nan

    if isxarray:
        R.data = array.reshape(shape0)
        R.attrs.update(metadata)
        return R

    return R.reshape(shape0), metadata


def sqrt_transform(R, metadata=None, inverse=False, **kwargs):
    """Square-root transform.

    Parameters
    ----------

    R : array-like  or xarray.Dataset
        Array of any shape to be transformed.

    metadata : dict, optional
        Metadata dictionary containing the transform, zerovalue and threshold
        attributes as described in the documentation of
        :py:mod:`pysteps.io.importers`.

    inverse : bool, optional
        If set to True, it performs the inverse transform. False by default.

    Returns
    -------

    R : array-like or xarray.DataArray
        Array of any shape containing the (back-)transformed units.

    metadata : dict
        The metadata with updated attributes.
    """

    if XARRAY_IMPORTED and isinstance(R, xr.DataArray):
        R = R.copy()
        metadata = R.attrs
        isxarray = True

    else:
        R = np.copy(R)
        isxarray = False

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
        R = R ** 2

        metadata["transform"] = None
        metadata["zerovalue"] = metadata["zerovalue"] ** 2
        metadata["threshold"] = metadata["threshold"] ** 2

    if isxarray:
        R.attrs.update(metadata)
        return R

    return R, metadata
