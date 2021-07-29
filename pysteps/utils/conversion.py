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
import warnings

from .decorators import dataarray_utils


# TODO: This should not be done. Instead fix the code so that it doesn't
# produce the warnings.
# to deactivate warnings for comparison operators with NaNs
warnings.filterwarnings("ignore", category=RuntimeWarning)



@dataarray_utils
def to_rainrate(data_array, zr_a=None, zr_b=None):
    """Convert to rain rates [mm/h].

    Parameters
    ----------
    data_array : xr.DataArray
        DataArray to be converted.

    zr_a, zr_b: float, optional
        The a and b coefficients of the Z-R relationship (Z = a*R^b).
        Used only if converting from dBZ.
        It defaults to Marshall-Palmer if the coefficients are not specified in
        the attributes of the variable being converted.

    Returns
    -------
    data_array : xr.DataArray
        DataArray with units converted to mm/h.
    """
    data_array = data_array.pysteps.back_transform()
    attrs = data_array.attrs

    units = attrs["unit"]
    accutime = attrs["accutime"]
    zr_a = zr_a if zr_a is not None else attrs.get("zr_a", 200.0)
    zr_b = zr_b if zr_b is not None else attrs.get("zr_b", 1.6)

    if units == "mm/h":
        pass

    elif units == "mm":
        data_array = data_array / float(accutime) * 60.0

    elif units == "dBZ":
        data_array = (data_array / zr_a) ** (1.0 / zr_b)

    else:
        raise ValueError(f"Unsupported unit conversion from {units}.")

    attrs.update({"unit": "mm/h"})
    data_array.attrs = attrs

    return data_array


@dataarray_utils
def to_raindepth(data_array, zr_a=None, zr_b=None):
    """Convert to rain depth [mm].

    Parameters
    ----------
    data_array : xr.DataArray
        DataArray to be converted.

    zr_a, zr_b: float, optional
        The a and b coefficients of the Z-R relationship (Z = a*R^b).
        Used only if converting from dBZ.
        It defaults to Marshall-Palmer if the coefficients are not specified in
        the attributes of the variable being converted.

    Returns
    -------
    data_array : xr.DataArray
        DataArray with units converted to mm.
    """
    data_array = data_array.pysteps.to_rainrate(zr_a, zr_b)
    attrs = data_array.attrs

    data_array = data_array / 60.0 * attrs["accutime"]

    attrs.update({"unit": "mm"})
    data_array.attrs = attrs

    return data_array


@dataarray_utils
def to_reflectivity(data_array, zr_a=None, zr_b=None, offset=0.01):
    """Convert to reflectivity [dBZ].

    Parameters
    ----------
    data_array : xr.DataArray
        DataArray to be converted.

    zr_a, zr_b: float, optional
        The a and b coefficients of the Z-R relationship (Z = a*R^b).
        Used only if converting from dBZ.
        It defaults to Marshall-Palmer if the coefficients are not specified in
        the attributes of the variable being converted.

    offset: float
        A small constant offset which prevents the division by zero or the
        computation of log(0).

    Returns
    -------
    data_array : xr.DataArray
        DataArray with units converted to dBZ.
    """
    data_array = data_array.pysteps.to_rainrate(zr_a, zr_b)
    attrs = data_array.attrs

    zr_a = zr_a if zr_a is not None else data_array.attrs.get("zr_a", 200.0)
    zr_b = zr_b if zr_b is not None else data_array.attrs.get("zr_b", 1.6)
    data_array = zr_a * data_array ** zr_b

    attrs.update({"unit": "dBZ"})
    data_array.attrs = attrs

    return data_array.pysteps.db_transform(offset=offset)
