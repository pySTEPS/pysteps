# -*- coding: utf-8 -*-
"""
One line description of this module. E.g. "Awesome Bureau of Composites importers."

Here you can write a more extensive description (optional). For example, describe the
readers that are implemented (e.g. Grib2 and Netcdf) and any other thing that you
consider relevant.
"""

# Here import your libraries
import numpy as np

### Uncomment the next lines if pyproj is needed for the importer.
# try:
#     import pyproj
#
#     PYPROJ_IMPORTED = True
# except ImportError:
#     PYPROJ_IMPORTED = False

from pysteps.decorators import postprocess_import
from pysteps.exceptions import MissingOptionalDependency


# Function import_abc_grib to import Grib files from the Awesome Bureau of Composites
# ===================================================================================

# IMPORTANT: The name of the importer should follow the "importer_institution_format"
# naming convention, where "institution" is the acronym or short-name of the
# institution. The "importer_" prefix to the importer name is MANDATORY since it is
# used by the pysteps interface.
#
# Check the pysteps documentation for examples of importers names that follow this
# convention:
# https://pysteps.readthedocs.io/en/latest/pysteps_reference/io.html#available-importers
#
# The definition of the new importer functions should have the following form:
#
#  @postprocess_import()
#  def import_institution_format(filename, keyword1="some_keyword", keyword2=10, **kwargs):
#
# The "filename" positional argument and the "**kwargs" argument are mandatory.
# However, the keyword1="some_keyword", keyword2=10 are optional shown in the above
# example are optional. In the case that the new importer needs optional keywords to
# fine control its behavior, it is strongly encouraged to define them explicitly
# instead of relying on the "**kwargs" argument.
#
#
# The @postprocess_import operator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `pysteps.io.importers` module uses the `postprocess_import` decorator to easily
# define the default data type and default value used for the missing/invalid data.
# The allowed postprocessing operations are
#   - Data type casting using the `dtype` keyword.
#   - Set invalid or missing data to a predefined value using the `fillna` keyword.
# The @postprocess_import decorator should always be added immediately above the
# importer definition to maintain full compatibility with the pysteps library.
# Adding the decorator @add_postprocess_keywords() without any keywords will ensure that
# the precipitation data returned by the importer has double precision, and the
# invalid/missing data is set to `np.nan`.
# For more information on the postprocessing decorator, see:
# https://pysteps.readthedocs.io/en/latest/generated/pysteps.decorators.postprocess_import.html
#
# Function arguments
# ~~~~~~~~~~~~~~~~~~
#
# The function arguments should have the following form:
# (filename, keyword1="some_keyword", keyword2=10,...,keywordN="something", **kwargs)
# The `filename` and `**kwargs` arguments are mandatory to comply with the pysteps
# interface. To fine-control the behavior of the importer, additional keywords can be
# added to the function.
# For example: keyword1="some_keyword", keyword2=10, ..., keywordN="something"
# It is recommended to declare the keywords explicitly in the function to improve the
# readability.
#
#
# Return arguments
# ~~~~~~~~~~~~~~~~
#
# The importer should always return the following fields:
#
# precipitation : 2D array (ndarray or MaskedArray)
#     Precipitation field in mm/h. The dimensions are [latitude, longitude].
# quality : 2D array or None
#     If no quality information is available, set to None.
# metadata : dict
#     Associated metadata (pixel sizes, map projections, etc.).
#
#
# Now, let's implement the importer.


@postprocess_import()
def import_abc_grib(filename, keyword1="some_keyword", keyword2=10, **kwargs):
    """
    Here you need to add the documentation for the importer. A minimal documentation is
    strictly needed since the pysteps importers interface expect docstrings.

    For example, a documentation may look like this:

    Import a precipitation field from the Awesome Bureau of Composites stored in
    Grib format.

    Parameters
    ----------
    filename : str
        Name of the file to import.

    keyword1 : str
        Some keyword used to fine control the importer behavior.

    keyword2 : int
        Another keyword used to fine control the importer behavior.

    {extra_kwargs_doc}

    ####################################################################################
    # The {extra_kwargs_doc} above is needed to add default keywords added to this     #
    # importer by the pysteps.decorator.postprocess_import decorator.                  #
    # IMPORTANT: Remove this box in the final version of this function                 #
    ####################################################################################

    Returns
    -------
    precipitation : 2D array, float32
        Precipitation field in mm/h. The dimensions are [latitude, longitude].
    quality : 2D array or None
        If no quality information is available, set to None.
    metadata : dict
        Associated metadata (pixel sizes, map projections, etc.).
    """

    ### Uncomment the next lines if pyproj is needed for the importer
    # if not PYPROJ_IMPORTED:
    #     raise MissingOptionalDependency(
    #         "pyproj package is required by import_abc_grib
    #         "but it is not installed"
    #     )

    ####################################################################################
    # Add the code to read the precipitation data here. Note that only cartesian grids
    # are supported by pysteps!

    # In this example, we are going create a precipitation fields of only zeros.
    precip = np.zeros((100, 100), dtype="double")
    # The "double" precision is used in this example to indicate that the imaginary
    # grib file stores the data using double precision.

    # Quality field, should have the same dimensions of the precipitation field.
    # Use None if not information is available.
    quality = None

    # Adjust the metadata fields according to the file format specifications.
    # For additional information on the metadata fields, see:
    # https://pysteps.readthedocs.io/en/latest/pysteps_reference/io.html#pysteps-io-importers

    # The projection definition is a string with a PROJ.4-compatible projection
    # definition of the cartographic projection used for the data
    # More info at: https://proj.org/usage/projections.html

    # For example:
    projection_definition = (
        "+proj=stere +lon_0=25E +lat_0=90N +lat_ts=60 +a=6371288 "
        "+x_0=380886.310 +y_0=3395677.920 +no_defs",
    )

    metadata = dict(
        xpixelsize=1,
        ypixelsize=1,
        cartesian_unit="km",
        unit="mm/h",
        transform=None,
        zerovalue=0,
        institution="The institution that created the file",
        projection=projection_definition,
        yorigin="upper",
        threshold=0.03,
        x1=0,
        x2=100,
        y1=0,
        y2=100,
    )

    # IMPORTANT! The importers should always return the following fields:
    return precip, quality, metadata


# Function import_abc_netcdf to import NETCDF files from the Awesome Bureau of Composites
# =======================================================================================


@postprocess_import()
def import_abc_netcdf(filename, keyword1="some_keyword", keyword2=10, **kwargs):
    """Import a precipitation field from the Awesome Bureau of Composites stored in
    Netcdf format.

    Parameters
    ----------
    filename : str
        Name of the file to import.

    keyword1 : str
        Some keyword used to fine control the importer behavior.

    keyword2 : int
        Another keyword used to fine control the importer behavior.

    {extra_kwargs_doc}

    ####################################################################################
    # The {extra_kwargs_doc} above is needed to add default keywords added to this     #
    # importer by the pysteps.decorator.postprocess_import decorator.                  #
    # IMPORTANT: Remove these box in the final version of this function                #
    ####################################################################################

    Returns
    -------
    precipitation : 2D array, float32
        Precipitation field in mm/h. The dimensions are [latitude, longitude].
    quality : 2D array or None
        If no quality information is available, set to None.
    metadata : dict
        Associated metadata (pixel sizes, map projections, etc.).
    """

    ### Uncomment the next lines if pyproj is needed for the importer
    # if not PYPROJ_IMPORTED:
    #     raise MissingOptionalDependency(
    #         "pyproj package is required by import_abc_grib
    #         "but it is not installed"
    #     )

    # Add the code to read the precipitation data (cartesian grid!) here.
    precip = np.zeros((100, 100))
    quality = np.ones((100, 100))

    projection_definition = (
        "+proj=stere +lon_0=25E +lat_0=90N +lat_ts=60 +a=6371288 "
        "+x_0=380886.310 +y_0=3395677.920 +no_defs",
    )

    metadata = dict(
        xpixelsize=1,
        ypixelsize=1,
        cartesian_unit="km",
        unit="mm/h",
        transform=None,
        zerovalue=0,
        institution="The institution that created the file",
        projection=projection_definition,
        yorigin="upper",
        threshold=0.03,
        x1=0,
        x2=100,
        y1=0,
        y2=100,
    )
    return precip, quality, metadata
