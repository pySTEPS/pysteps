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

from . import dataarray_accessor


# TODO: This should not be done. Instead fix the code so that it doesn't
# produce the warnings.
# to deactivate warnings for comparison operators with NaNs
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataarray_accessor
def to_rainrate(da, zr_a=None, zr_b=None):
    """Convert to rain rates [mm/h].

    Parameters
    ----------
    da : xr.DataArray
        DataArray to be converted.
    zr_a, zr_b: float, optional
        The a and b coefficients of the Z-R relationship (Z = a*R^b).
        Used only if converting from dBZ.
        It defaults to Marshall-Palmer if the coefficients are not specified in
        the attributes of the variable being converted.

    Returns
    -------
    da : xr.DataArray
        DataArray with units converted to mm/h.
    """
    da = _back_transform(da)

    units = da.attrs["unit"]
    threshold = da.attrs["threshold"]
    zerovalue = da.attrs["zerovalue"]
    accutime = da.attrs["accutime"]
    zr_a = zr_a if zr_a is not None else da.attrs.get("zr_a", 200.0)
    zr_b = zr_b if zr_b is not None else da.attrs.get("zr_b", 1.6)

    if units == "mm/h":
        pass

    elif units == "mm":
        da = da / float(accutime) * 60.0
        threshold = threshold / float(accutime) * 60.0
        zerovalue = zerovalue / float(accutime) * 60.0

    elif units == "dBZ":
        da = (da / zr_a) ** (1.0 / zr_b)
        threshold = (threshold / zr_a) ** (1.0 / zr_b)
        zerovalue = (zerovalue / zr_a) ** (1.0 / zr_b)

    else:
        raise ValueError(f"Unsupported unit conversion from {units}.")

    da.attrs.update(
        {
            "unit": "mm/h",
            "threshold": threshold,
            "zerovalue": zerovalue,
        }
    )

    return da


@dataarray_accessor
def to_raindepth(da, zr_a=None, zr_b=None):
    """Convert to rain depth [mm].

    Parameters
    ----------
    da : xr.DataArray
        DataArray to be converted.
    zr_a, zr_b: float, optional
        The a and b coefficients of the Z-R relationship (Z = a*R^b).
        Used only if converting from dBZ.
        It defaults to Marshall-Palmer if the coefficients are not specified in
        the attributes of the variable being converted.

    Returns
    -------
    da : xr.DataArray
        DataArray with units converted to mm.
    """
    da = da.to_rainrate(zr_a, zr_b)

    threshold = da.attrs["threshold"]
    zerovalue = da.attrs["zerovalue"]
    accutime = da.attrs["accutime"]

    da = da / 60.0 * accutime
    threshold = threshold / 60.0 * accutime
    zerovalue = zerovalue / 60.0 * accutime

    da.attrs.update(
        {
            "unit": "mm",
            "threshold": threshold,
            "zerovalue": zerovalue,
        }
    )

    return da


@dataarray_accessor
def to_reflectivity(da, zr_a=None, zr_b=None):
    """Convert to reflectivity [dBZ].

    Parameters
    ----------
    da : xr.DataArray
        DataArray to be converted.
    zr_a, zr_b: float, optional
        The a and b coefficients of the Z-R relationship (Z = a*R^b).
        Used only if converting from dBZ.
        It defaults to Marshall-Palmer if the coefficients are not specified in
        the attributes of the variable being converted.

    Returns
    -------
    da : xr.DataArray
        DataArray with units converted to dBZ.
    """
    da = da.to_rainrate(zr_a, zr_b)

    threshold = da.attrs["threshold"]
    zerovalue = da.attrs["zerovalue"]
    accutime = da.attrs["accutime"]
    zr_a = zr_a if zr_a is not None else da.attrs.get("zr_a", 200.0)
    zr_b = zr_b if zr_b is not None else da.attrs.get("zr_b", 1.6)

    da = da / 60.0 * accutime
    threshold = threshold / 60.0 * accutime
    zerovalue = zerovalue / 60.0 * accutime

    da = zr_a * da ** zr_b
    threshold = zr_a * threshold ** zr_b
    zerovalue = zr_a * zerovalue ** zr_b

    da.attrs.update(
        {
            "unit": "mm",
            "threshold": threshold,
            "zerovalue": zerovalue,
        }
    )

    return da
