# -*- coding: utf-8 -*-
"""
"""
from functools import partial, wraps

import xarray as xr


def _make_accessor(func, accessor_class):
    """
    Turn functions for xarray objects into accessors.
    """

    @wraps(func)
    def method(accessor, *args, **kwargs):
        return func(accessor._data, *args, **kwargs)

    setattr(accessor_class, func.__name__, method)
    return func


@xr.register_dataarray_accessor("pysteps")
class DataArrayUtils:
    """
    Xarray accessor class for pysteps utilities on DataArrays.
    """

    def __init__(self, da: xr.DataArray):
        self._data = da


dataarray_utils = partial(_make_accessor, accessor_class=DataArrayUtils)
