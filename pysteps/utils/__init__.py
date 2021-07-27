"""Miscellaneous utility functions."""
from functools import partial, wraps

import xarray as xr

from .arrays import *
from .cleansing import *
from .conversion import *
from .dimension import *
from .images import *
from .interface import get_method
from .interpolate import *
from .fft import *
from .spectral import *
from .tapering import *
from .transformation import *


def _make_accessor(func, accessor_class):
    """
    Turn functions for xarray objects into accessors.
    """
    @wraps(func)
    def method(accessor, *args, **kwargs):
        return func(accessor._data, *args, **kwargs)

    setattr(accessor_class, func.__name__, method)
    return func

@xr.register_dataset_accessor("pysteps")
class DatasetUtils:
    """
    Xarray accessor class for pysteps utilities on Datasets.
    """
    def __init__(self, ds: xr.Dataset):
        self._data = ds


dataset_utils = partial(_make_accessor, accessor_class=DatasetUtils)


@xr.register_dataarray_accessor("pysteps")
class DataArrayUtils:
    """
    Xarray accessor class for pysteps utilities on DataArrays.
    """
    def __init__(self, da: xr.DataArray):
        self._data = da


dataarray_utils = partial(_make_accessor, accessor_class=DatasetUtils)
