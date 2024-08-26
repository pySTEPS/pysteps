"""
pysteps.postprocessing.diagnostics
====================

Methods for applying diagnostics postprocessing.

The methods in this module implement the following interface::

    diagnostics_xxx(optional arguments)

where **xxx** is the name of the diagnostic to be applied.

Available Diagnostics Postprocessors
------------------------

.. autosummary::
    :toctree: ../generated/

"""

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


def postprocessors_diagnostics_example1(filename, **kwargs):
    return "Hello, I am an example diagnostics postprocessor."


def postprocessors_diagnostics_example2(filename, **kwargs):
    return [[42, 42], [42, 42]]
