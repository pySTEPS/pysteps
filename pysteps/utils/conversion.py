# -*- coding: utf-8 -*-
"""
pysteps.utils.conversion
========================

Methods for converting physical units.

.. autosummary::
    :toctree: ../generated/

    to_rainrate
    to_raindepth
    to_reflectivity
"""
from typing import Any

import xarray as xr

from .decorators import dataarray_utils


@dataarray_utils
def to_rainrate(data_array: xr.DataArray, **kwargs: Any) -> xr.DataArray:
    """Convert to rain rates [mm/h].

    Parameters
    ----------
    data_array : xr.DataArray
        DataArray to be converted.

    Other parameters
    ----------------
    accutime: float
        The accumulation time in minutes, used if converting from rain depth.
        If not passed as an argument, the accutime specified in the
        attributes of the input xr.DataArray is used.

    zr_a, zr_b: float
        The a and b coefficients of the Z-R relationship (Z = a*R^b), used if
        converting from Z units.
        If not passed as arguments nor included as attributes in the input
        xr.DataArray, the default Marshall-Palmer is used (a=200, b=1.6).

    Returns
    -------
    data_array : xr.DataArray
        DataArray with units converted to mm/h.
    """
    data_array = data_array.pysteps.back_transform()
    attrs = data_array.attrs

    units = attrs["unit"]
    accutime = kwargs.get("accutime", data_array.attrs["accutime"])
    zr_a = kwargs.get("zr_a", data_array.attrs.get("zr_a", 200.0))
    zr_b = kwargs.get("zr_b", data_array.attrs.get("zr_b", 1.6))

    if units == "mm/h":
        pass

    elif units == "mm":
        data_array = data_array / float(accutime) * 60.0

    elif units == "Z":
        data_array = (data_array / zr_a) ** (1.0 / zr_b)

    else:
        raise ValueError(f"Unsupported unit conversion from {units}.")

    attrs.update({"unit": "mm/h"})
    data_array.attrs = attrs

    return data_array


@dataarray_utils
def to_raindepth(data_array: xr.DataArray, **kwargs: Any) -> xr.DataArray:
    """Convert to rain depth [mm].

    Parameters
    ----------
    data_array : xr.DataArray
        DataArray to be converted.

    Other parameters
    ----------------
    accutime: float
        The accumulation time in minutes.
        If not passed as an argument, the accutime specified in the
        attributes of the input xr.DataArray is used.

    zr_a, zr_b: float
        The a and b coefficients of the Z-R relationship (Z = a*R^b), used if
        converting from dBZ.
        If not passed as arguments nor include as attributes in the input
        xr.DataArray, the default Marshall-Palmer is used (a=200, b=1.6).

    Returns
    -------
    data_array : xr.DataArray
        DataArray with units converted to mm.
    """
    data_array = data_array.pysteps.to_rainrate(**kwargs)
    attrs = data_array.attrs

    accutime = kwargs.get("accutime", attrs["accutime"])
    data_array = data_array / 60.0 * accutime

    attrs.update({"unit": "mm"})
    data_array.attrs = attrs

    return data_array


@dataarray_utils
def to_reflectivity(
    data_array: xr.DataArray, to_decibels: bool = True, offset: float = 0.01, **kwargs: Any
) -> xr.DataArray:
    """Convert to linear reflectivity [Z] or dBZ units.

    Parameters
    ----------
    data_array : xr.DataArray
        DataArray to be converted.

    to_decibels: bool
        If true, transform to decibels [dB].

    offset: float
        A small constant offset which prevents the division by zero or the
        computation of log(0). Used if to_decibels is True.

    Other parameters
    ----------------
    accutime: float
        The accumulation time in minutes, used if converting from rain depth.
        If not passed as an argument, the accutime specified in the
        attributes of the input xr.DataArray is used.

    zr_a, zr_b: float
        The a and b coefficients of the Z-R relationship `Z = a * R ** b`.
        If not passed as arguments nor include as attributes in the input
        xr.DataArray, the default Marshall-Palmer is used (a=200, b=1.6).

    Returns
    -------
    data_array : xr.DataArray
        DataArray with units converted to Z or dBZ units.
    """
    data_array = data_array.pysteps.to_rainrate(**kwargs)
    attrs = data_array.attrs

    zr_a = kwargs.get("zr_a", attrs.get("zr_a", 200.0))
    zr_b = kwargs.get("zr_b", attrs.get("zr_b", 1.6))
    data_array = zr_a * data_array ** zr_b

    attrs.update({"unit": "Z"})
    data_array.attrs = attrs

    if to_decibels:
        return data_array.pysteps.db_transform(offset=offset)
    else:
        return data_array
