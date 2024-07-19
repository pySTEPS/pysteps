"""
pysteps.io.postprocessors
====================

Methods for applying postprocessing.

The methods in this module implement the following interface::

    postprocess_xxx(optional arguments)

where **xxx** is the name of the postprocess to be applied.

Postprocessor standardizations can be specified here if there is a desired input and output format that all should adhere to.

Available Postprocessors
------------------------

.. autosummary::
    :toctree: ../generated/

"""

import gzip
import os
from functools import partial

import numpy as np

from matplotlib.pyplot import imread

from pysteps.decorators import postprocess_import
from pysteps.exceptions import DataModelError
from pysteps.exceptions import MissingOptionalDependency
from pysteps.utils import aggregate_fields

try:
    from osgeo import gdal, gdalconst, osr

    GDAL_IMPORTED = True
except ImportError:
    GDAL_IMPORTED = False

try:
    import h5py

    H5PY_IMPORTED = True
except ImportError:
    H5PY_IMPORTED = False

try:
    import metranet

    METRANET_IMPORTED = True
except ImportError:
    METRANET_IMPORTED = False

try:
    import netCDF4

    NETCDF4_IMPORTED = True
except ImportError:
    NETCDF4_IMPORTED = False

try:
    from PIL import Image

    PIL_IMPORTED = True
except ImportError:
    PIL_IMPORTED = False

try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False

try:
    import pygrib

    PYGRIB_IMPORTED = True
except ImportError:
    PYGRIB_IMPORTED = False