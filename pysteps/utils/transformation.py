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

import warnings

import numpy as np
import scipy.stats as scipy_stats
import xarray as xr
from scipy.interpolate import interp1d

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # To deactivate warnings for comparison operators with NaNs


def boxcox_transform(
    dataset: xr.Dataset, Lambda=None, threshold=None, zerovalue=None, inverse=False
) -> xr.Dataset:
    """
    The one-parameter Box-Cox transformation.

    The Box-Cox transform is a well-known power transformation introduced by
    Box and Cox (1964). In its one-parameter version, the Box-Cox transform
    takes the form T(x) = ln(x) for Lambda = 0,
    or T(x) = (x**Lambda - 1)/Lambda otherwise.

    Default parameters will produce a log transform (i.e. Lambda=0).

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset to be transformed.
    Lambda: float, optional
        Parameter Lambda of the Box-Cox transformation.
        It is 0 by default, which produces the log transformation.

        Choose Lambda < 1 for positively skewed data, Lambda > 1 for negatively
        skewed data.
    threshold: float, optional
        The value that is used for thresholding with the same units as in the dataset.
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
    dataset: xarray.Dataset
        Dataset containing the (back-)transformed units.

    References
    ----------
    Box, G. E. and Cox, D. R. (1964), An Analysis of Transformations. Journal
    of the Royal Statistical Society: Series B (Methodological), 26: 211-243.
    doi:10.1111/j.2517-6161.1964.tb00553.x
    """

    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs
    precip_data = dataset[precip_var].values

    if not inverse:
        if metadata["transform"] == "BoxCox":
            return dataset

        if Lambda is None:
            Lambda = metadata.get("BoxCox_lambda", 0.0)

        if threshold is None:
            threshold = metadata.get("threshold", 0.1)

        zeros = precip_data < threshold

        # Apply Box-Cox transform
        if Lambda == 0.0:
            precip_data[~zeros] = np.log(precip_data[~zeros])
            threshold = np.log(threshold)

        else:
            precip_data[~zeros] = (precip_data[~zeros] ** Lambda - 1) / Lambda
            threshold = (threshold**Lambda - 1) / Lambda

        # Set value for zeros
        if zerovalue is None:
            zerovalue = threshold - 1  # TODO: set to a more meaningful value
        precip_data[zeros] = zerovalue

        metadata["transform"] = "BoxCox"
        metadata["BoxCox_lambda"] = Lambda
        metadata["zerovalue"] = zerovalue
        metadata["threshold"] = threshold

    elif inverse:
        if metadata["transform"] not in ["BoxCox", "log"]:
            return precip_data, metadata

        if Lambda is None:
            Lambda = metadata.pop("BoxCox_lambda", 0.0)
        if threshold is None:
            threshold = metadata.get("threshold", -10.0)
        if zerovalue is None:
            zerovalue = 0.0

        # Apply inverse Box-Cox transform
        if Lambda == 0.0:
            precip_data = np.exp(precip_data)
            threshold = np.exp(threshold)

        else:
            precip_data = np.exp(np.log(Lambda * precip_data + 1) / Lambda)
            threshold = np.exp(np.log(Lambda * threshold + 1) / Lambda)

        precip_data[precip_data < threshold] = zerovalue

        metadata["transform"] = None
        metadata["zerovalue"] = zerovalue
        metadata["threshold"] = threshold

    dataset[precip_var].data[:] = precip_data

    return dataset


def dB_transform(
    dataset: xr.Dataset, threshold=None, zerovalue=None, inverse=False
) -> xr.Dataset:
    """Methods to transform precipitation intensities to/from dB units.

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset to be (back-)transformed.
    threshold: float, optional
        Optional value that is used for thresholding with the same units as in the dataset.
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
    dataset: xarray.Dataset
        Dataset containing the (back-)transformed units.
    """

    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs
    precip_data = dataset[precip_var].values

    # to dB units
    if not inverse:
        if metadata["transform"] == "dB":
            return dataset

        if threshold is None:
            threshold = metadata.get("threshold", 0.1)

        zeros = precip_data < threshold

        # Convert to dB
        precip_data[~zeros] = 10.0 * np.log10(precip_data[~zeros])
        threshold = 10.0 * np.log10(threshold)

        # Set value for zeros
        if zerovalue is None:
            zerovalue = threshold - 5  # TODO: set to a more meaningful value
        precip_data[zeros] = zerovalue

        metadata["transform"] = "dB"
        metadata["zerovalue"] = zerovalue
        metadata["threshold"] = threshold

    # from dB units
    elif inverse:
        if metadata["transform"] != "dB":
            return dataset

        if threshold is None:
            threshold = metadata.get("threshold", -10.0)
        if zerovalue is None:
            zerovalue = 0.0

        precip_data = 10.0 ** (precip_data / 10.0)
        threshold = 10.0 ** (threshold / 10.0)
        precip_data[precip_data < threshold] = zerovalue

        metadata["transform"] = None
        metadata["threshold"] = threshold
        metadata["zerovalue"] = zerovalue

    dataset[precip_var].data[:] = precip_data

    return dataset


def NQ_transform(dataset: xr.Dataset, inverse: bool = False, **kwargs) -> xr.Dataset:
    """
    The normal quantile transformation as in Bogner et al (2012).
    Zero rain vales are set to zero in norm space.

    Parameters
    ----------
    dataset: xarray.Dataset
       Dataset to be transformed.
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
    dataset: xarray.Dataset
        Dataset containing the (back-)transformed units.

    References
    ----------
    Bogner, K., Pappenberger, F., and Cloke, H. L.: Technical Note: The normal
    quantile transformation and its application in a flood forecasting system,
    Hydrol. Earth Syst. Sci., 16, 1085-1094,
    https://doi.org/10.5194/hess-16-1085-2012, 2012.
    """

    # defaults
    a = kwargs.get("a", 0.0)

    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs
    precip_data = dataset[precip_var].values

    shape0 = precip_data.shape
    precip_data = precip_data.ravel().astype(float)
    idxNan = np.isnan(precip_data)
    precip_data_ = precip_data[~idxNan]

    if not inverse:
        # Plotting positions
        # https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Plotting_position
        n = precip_data_.size
        Rpp = ((np.arange(n) + 1 - a) / (n + 1 - 2 * a)).reshape(precip_data_.shape)

        # NQ transform
        Rqn = scipy_stats.norm.ppf(Rpp)
        precip_data__ = np.interp(
            precip_data_, precip_data_[np.argsort(precip_data_)], Rqn
        )

        # set zero rain to 0 in norm space
        precip_data__[precip_data[~idxNan] == metadata["zerovalue"]] = 0

        # build inverse transform
        metadata["inqt"] = interp1d(
            Rqn,
            precip_data_[np.argsort(precip_data_)],
            bounds_error=False,
            fill_value=(precip_data_.min(), precip_data_.max()),
        )

        metadata["transform"] = "NQT"
        metadata["zerovalue"] = 0
        metadata["threshold"] = precip_data__[precip_data__ > 0].min()

    else:
        f = metadata.pop("inqt")
        precip_data__ = f(precip_data_)
        metadata["transform"] = None
        metadata["zerovalue"] = precip_data__.min()
        metadata["threshold"] = precip_data__[precip_data__ > precip_data__.min()].min()

    precip_data[~idxNan] = precip_data__

    dataset[precip_var].data[:] = precip_data.reshape(shape0)

    return dataset


def sqrt_transform(dataset: xr.Dataset, inverse: bool = False, **kwargs) -> xr.Dataset:
    """
    Square-root transform.

    Parameters
    ----------
    dataset: xarray.Dataset
        Dataset to be transformed.
    inverse: bool, optional
        If set to True, it performs the inverse transform. False by default.

    Returns
    -------
    dataset: xarray.Dataset
        Dataset containing the (back-)transformed units.

    """

    dataset = dataset.copy(deep=True)
    precip_var = dataset.attrs["precip_var"]
    metadata = dataset[precip_var].attrs
    precip_data = dataset[precip_var].values

    if not inverse:
        # sqrt transform
        precip_data = np.sqrt(precip_data)

        metadata["transform"] = "sqrt"
        metadata["zerovalue"] = np.sqrt(metadata["zerovalue"])
        metadata["threshold"] = np.sqrt(metadata["threshold"])
    else:
        # inverse sqrt transform
        precip_data = precip_data**2

        metadata["transform"] = None
        metadata["zerovalue"] = metadata["zerovalue"] ** 2
        metadata["threshold"] = metadata["threshold"] ** 2

    dataset[precip_var].data[:] = precip_data

    return dataset
