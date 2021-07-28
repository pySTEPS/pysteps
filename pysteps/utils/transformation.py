# -*- coding: utf-8 -*-
"""
pysteps.utils.transformation
============================

Methods for transforming data values.

.. autosummary::
    :toctree: ../generated/

    boxcox_transform
    db_transform
    nq_transform
    sqrt_transform
"""
from functools import partial

import numpy as np
import warnings
import xarray as xr
from scipy.stats import norm

from .decorators import dataarray_utils


warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # To deactivate warnings for comparison operators with NaNs


@dataarray_utils
def boxcox_transform(data_array, boxcox_lambda=0.0, offset=0.01, inverse=False):
    """The one-parameter Box-Cox transformation.

    The Box-Cox transform is a well-known power transformation introduced by
    Box and Cox (1964). In its one-parameter version, the Box-Cox transform
    takes the form T(x) = ln(x) for boxcox_lambda = 0,
    or T(x) = (x ** boxcox_lambda - 1) / boxcox_lambda otherwise.

    Default parameters will produce a log transform (i.e. boxcox_lambda=0).

    Parameters
    ----------
    data_array: xr.DataArray
        Array of any shape to be (back-)transformed.

    boxcox_lambda: float
        Parameter lambda of the Box-Cox transformation.
        It is 0 by default, which produces the log transformation.

        Choose boxcox_lambda < 1 for positively skewed data, boxcox_lambda > 1
        for negatively skewed data.

    inverse: bool
        If set to True, it performs the inverse transform. False by default.

    offset: float
        A small constant offset which prevents the division by zero or the
        computation of log(0).

    Returns
    -------
    data_array: xr.DataArray
        DataArray containing the (back-)transformed units.

    References
    ----------
    Box, G. E. and Cox, D. R. (1964), An Analysis of Transformations. Journal
    of the Royal Statistical Society: Series B (Methodological), 26: 211-243.
    doi:10.1111/j.2517-6161.1964.tb00553.x
    """
    data_array = data_array.copy()
    attrs = data_array.attrs

    if not inverse:

        data_array = _back_transform(data_array)
        data_array += offset
        if boxcox_lambda == 0.0:
            data_array = np.log(data_array)
        else:
            data_array = (data_array ** boxcox_lambda - 1) / boxcox_lambda

        attrs.update(
            {
                "transform": "BoxCox",
                "boxcox_lambda": boxcox_lambda,
                "offset": offset,
            }
        )
        data_array.attrs = attrs

    else:

        if attrs["transform"].lower() not in ["boxcox", "log"]:
            raise ValueError(f"cannot back-transform from {attrs['transform']}")

        boxcox_lambda = attrs.pop("boxcox_lambda", boxcox_lambda)

        if boxcox_lambda == 0.0:
            data_array = np.exp(data_array)
        else:
            data_array = np.log(boxcox_lambda * data_array + 1) / boxcox_lambda
            data_array = np.exp(data_array)

        data_array -= attrs.pop("offset", offset)
        attrs.update({"transform": None})
        data_array.attrs = attrs

    return data_array


@dataarray_utils
def db_transform(data_array, offset=0.01, inverse=False):
    """Transform to/from decibel (dB).

    Parameters
    ----------
    data_array: xr.DataArray
        Array of any shape to be (back-)transformed.

    offset: float
        A small constant offset which prevents the division by zero or the
        computation of log(0).

    Returns
    -------
    data_array: xr.DataArray
        DataArray containing the (back-)transformed units.
    """
    data_array = data_array.copy()
    attrs = data_array.attrs

    if not inverse:
        data_array = _back_transform(data_array)
        data_array += offset
        data_array = 10.0 * np.log10(data_array)

        attrs.update(
            {
                "transform": "dB",
                "offset": offset,
            }
        )
        data_array.attrs = attrs

    else:

        if not attrs["transform"].lower() == "db":
            raise ValueError(f"cannot back-transform from {attrs['transform']}")

        data_array = 10.0 ** (data_array / 10.0)
        data_array -= attrs.pop("offset", offset)
        attrs.update({"transform": None})
        data_array.attrs = attrs

    return data_array


@dataarray_utils
def nq_transform(data_array, nq_a=0.0, template=None, inverse=False):
    """The normal quantile transformation as in Bogner et al (2012).
    Zero rain vales are set to zero in norm space.

    Parameters
    ----------
    data_array: xr.DataArray
        Array of any shape to be (back-)transformed.

    nq_a: float
        The offset fraction to be used for plotting positions; typically in (0, 1).
        The default is 0, that is, it spaces the points evenly in the uniform
        distribution.

    inverse: bool
        If set to True, it performs the inverse transform. False by default.

    template: array_like, optional
        Array of any shape containing the samples to build the target empirical
        distribution.
        Required if inverse is True.

    Returns
    -------
    data_array: xr.DataArray
        DataArray containing the (back-)transformed units.

    References
    ----------
    Bogner, K., Pappenberger, F., and Cloke, H. L.: Technical Note: The normal
    quantile transformation and its application in a flood forecasting system,
    Hydrol. Earth Syst. Sci., 16, 1085-1094,
    https://doi.org/10.5194/hess-16-1085-2012, 2012.
    """
    data_array = data_array.copy()
    attrs = data_array.attrs

    if not inverse:

        data_array = _back_transform(data_array)

        # Resolve ties at random
        data_array += np.random.random(data_array.shape) / 1e10

        # Compute quantiles
        data_array = data_array.stack(dummy=(data_array.dims))
        n = int(data_array.notnull().sum())
        data_quantiles = (data_array.rank("dummy") - nq_a) / (n + 1 - 2 * nq_a)
        data_quantiles = data_quantiles.unstack("dummy")

        # Normal quantile transform
        data_array = xr.apply_ufunc(norm.ppf, data_quantiles)

        attrs.update(
            {
                "transform": "NQ",
                "unit": None,
            }
        )
        data_array.attrs = attrs

    else:

        if not attrs["transform"].lower() == "nq":
            raise ValueError(f"cannot back-transform from {attrs['transform']}")

        if template is None:
            raise ValueError("inverse NQ transform needs a template")

        try:
            template_unit = template.attrs.get("unit", None)
        except AttributeError:
            template_unit = None

        # Flatten and sort template
        template = np.array(template).flatten()
        template.sort()
        n = int(np.isfinite(template).sum())
        template = np.concatenate((template, np.array([np.nan])))

        # Inverse of ECDF
        data_array = data_array.stack(dummy=(data_array.dims))
        data_quantiles = xr.apply_ufunc(norm.cdf, data_array)
        index = np.floor((n - 1) * data_quantiles)
        index = index.fillna(n).astype("uint32")
        data_array = xr.DataArray(
            data=template[index], coords=data_array.coords, dims=data_array.dims
        ).unstack("dummy")

        attrs.update({"transform": None, "unit": template_unit})
        data_array.attrs = attrs

    return data_array


@dataarray_utils
def sqrt_transform(data_array, inverse=False):
    """Square-root transform.

    Parameters
    ----------
    data_array: xr.DataArray
        Array of any shape to be (back-)transformed.

    inverse: bool, optional
        If set to True, it performs the inverse transform. False by default.

    Returns
    -------
    data_array: xr.DataArray
        DataArray containing the (back-)transformed units.
    """
    data_array = data_array.copy()
    attrs = data_array.attrs

    if not inverse:
        data_array = _back_transform(data_array)
        data_array = np.sqrt(data_array)

        attrs.update(
            {
                "transform": "sqrt",
            }
        )
        data_array.attrs = attrs

    else:
        if not attrs["transform"].lower() == "sqrt":
            raise ValueError(f"cannot back-transform from {attrs['transform']}")

        data_array = data_array ** 2.0
        attrs.update({"transform": None})
        data_array.attrs = attrs

    return data_array


def _back_transform(da):
    """Remove any existing transformation."""
    inverse_methods = {
        None: lambda x: x.copy(),
        "dB": partial(db_transform, inverse=True),
        "boxcox": partial(boxcox_transform, inverse=True),
        "log": partial(boxcox_transform, inverse=True, boxcox_lambda=0.0),
        "nq": partial(nq_transform, inverse=True),
        "sqrt": partial(sqrt_transform, inverse=True),
    }
    transform = da.attrs.get("transform")
    inverse_method = inverse_methods.get(transform)
    if inverse_method is None:
        raise ValueError(f"unknown transformation {transform}.")
    return inverse_method(da)
