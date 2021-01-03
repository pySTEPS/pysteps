"""
pysteps.visualization.utils
===========================

Miscellaneous utility functions for the visualization module.

.. autosummary::
    :toctree: ../generated/

    parse_proj4_string
    proj4_to_cartopy
    reproject_geodata
    get_geogrid
    get_basemap_axis
"""
import warnings

import numpy as np
from cartopy.mpl.geoaxes import GeoAxesSubplot

from pysteps.exceptions import MissingOptionalDependency
from pysteps.exceptions import UnsupportedSomercProjection
import matplotlib.pylab as plt

from pysteps.visualization import basemaps

try:
    import cartopy.crs as ccrs

    CARTOPY_IMPORTED = True
except ImportError:
    CARTOPY_IMPORTED = False
try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False

CARTOPY_PROJ_KWRDS_TO_PYPROJ = {
    "lon_0": "central_longitude",
    "lat_0": "central_latitude",
    "lat_ts": "true_scale_latitude",
    "x_0": "false_easting",
    "y_0": "false_northing",
    "k": "scale_factor",
    "zone": "zone",
}

# Invert the CARTOPY_PROJ_KWRDS_TO_PYPROJ dict
PYPROJ_PROJ_KWRDS_TO_CARTOPY = {v: k for k, v in CARTOPY_PROJ_KWRDS_TO_PYPROJ.items()}

PYPROJ_GLOB_KWRDS_TO_CARTOPY = {
    "a": "semimajor_axis",
    "b": "semiminor_axis",
    "datum": "datum",
    "ellps": "ellipse",
    "f": "flattening",
    "rf": "inverse_flattening",
}

PYPROJ_PROJECTION_TO_CARTOPY = dict(
    tmerc=ccrs.TransverseMercator,
    laea=ccrs.LambertAzimuthalEqualArea,
    lcc=ccrs.LambertConformal,
    merc=ccrs.Mercator,
    utm=ccrs.UTM,
    stere=ccrs.Stereographic,
    aea=ccrs.AlbersEqualArea,
    aeqd=ccrs.AzimuthalEquidistant,
    # Note: ccrs.epsg(2056) doesn't work because the projection
    # limits are too strict.
    # We'll use the Stereographic projection as an alternative.
    somerc=ccrs.Stereographic,
    geos=ccrs.Geostationary,
)


def parse_proj4_string(proj4str):
    """Construct a dictionary from a PROJ.4 projection string.

    Parameters
    ----------
    proj4str: str
      A PROJ.4-compatible projection string.

    Returns
    -------
    out: dict
      Dictionary, where keys and values are parsed from the projection
      parameter tokens beginning with '+'.
    """

    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required for parse_proj4_string function utility "
            "but it is not installed"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # Ignore the warning raised by to_dict() about losing information.
        proj_dict = pyproj.Proj(proj4str).crs.to_dict()

    return proj_dict


def proj4_to_cartopy(proj4str):
    """Convert a PROJ.4 projection string into a Cartopy coordinate reference
    system (crs) object.

    Parameters
    ----------
    proj4str: str
        A PROJ.4-compatible projection string.

    Returns
    -------
    out: object
        Instance of a crs class defined in cartopy.crs.
    """
    if not CARTOPY_IMPORTED:
        raise MissingOptionalDependency(
            "cartopy package is required for proj4_to_cartopy function "
            "utility but it is not installed"
        )

    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required for proj4_to_cartopy function utility "
            "but it is not installed"
        )

    proj = pyproj.Proj(proj4str)

    try:
        # pyproj >= 2.2.0
        is_geographic = proj.crs.is_geographic
    except AttributeError:
        # pyproj < 2.2.0
        is_geographic = proj.is_latlong()

    if is_geographic:
        return ccrs.PlateCarree()

    proj_dict = parse_proj4_string(proj4str)

    cartopy_crs_kwargs = dict()
    globe_kwargs = dict()
    cartopy_crs = None
    globe = None

    for key, value in proj_dict.items():
        if key == "proj":
            if value in PYPROJ_PROJECTION_TO_CARTOPY:
                cartopy_crs = PYPROJ_PROJECTION_TO_CARTOPY[value]
            else:
                raise ValueError(f"Unsupported projection: {value}")

        if key in PYPROJ_PROJ_KWRDS_TO_CARTOPY:
            cartopy_crs_kwargs[PYPROJ_PROJ_KWRDS_TO_CARTOPY[key]] = value

        if key in PYPROJ_GLOB_KWRDS_TO_CARTOPY:
            globe_kwargs[PYPROJ_GLOB_KWRDS_TO_CARTOPY[key]] = value

    # issubset: <=
    if {"lat_1", "lat_2"} <= proj_dict.keys():
        cartopy_crs_kwargs["standard_parallels"] = (
            proj_dict["lat_1"],
            proj_dict["lat_2"],
        )

    if globe_kwargs:
        globe = ccrs.Globe(**globe_kwargs)

    if isinstance(cartopy_crs, ccrs.Mercator):
        cartopy_crs_kwargs.pop("false_easting", None)
        cartopy_crs_kwargs.pop("false_northing", None)

    return cartopy_crs(globe=globe, **cartopy_crs_kwargs)


def reproject_geodata(geodata, t_proj4str, return_grid=None):
    """
    Reproject geodata and optionally create a grid in a new projection.

    Parameters
    ----------
    geodata: dictionary
        Dictionary containing geographical information about the field.
        It must contain the attributes projection, x1, x2, y1, y2, xpixelsize,
        ypixelsize, as defined in the documentation of pysteps.io.importers.
    t_proj4str: str
        The target PROJ.4-compatible projection string (fallback).
    return_grid: {None, 'coords', 'quadmesh'}, optional
        Whether to return the coordinates of the projected grid.
        The default return_grid=None does not compute the grid,
        return_grid='coords' returns the centers of projected grid points,
        return_grid='quadmesh' returns the coordinates of the quadrilaterals
        (e.g. to be used by pcolormesh).

    Returns
    -------
    geodata: dictionary
        Dictionary containing the reprojected geographical information
        and optionally the required X_grid and Y_grid. \n
        It also includes a fixed boolean attribute
        regular_grid=False to indicate
        that the reprojected grid has no regular spacing.
    """
    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required for reproject_geodata function utility"
            " but it is not installed"
        )

    geodata = geodata.copy()
    s_proj4str = geodata["projection"]
    extent = (geodata["x1"], geodata["x2"], geodata["y1"], geodata["y2"])
    shape = (
        int((geodata["y2"] - geodata["y1"]) / geodata["ypixelsize"]),
        int((geodata["x2"] - geodata["x1"]) / geodata["xpixelsize"]),
    )

    s_srs = pyproj.Proj(s_proj4str)
    t_srs = pyproj.Proj(t_proj4str)

    x1 = extent[0]
    x2 = extent[1]
    y1 = extent[2]
    y2 = extent[3]

    # Reproject grid on fall-back projection
    if return_grid is not None:
        if return_grid == "coords":
            y_coord = np.linspace(y1, y2, shape[0]) + geodata["ypixelsize"] / 2.0
            x_coord = np.linspace(x1, x2, shape[1]) + geodata["xpixelsize"] / 2.0
        elif return_grid == "quadmesh":
            y_coord = np.linspace(y1, y2, shape[0] + 1)
            x_coord = np.linspace(x1, x2, shape[1] + 1)
        else:
            raise ValueError("unknown return_grid value %s" % return_grid)

        x_grid, y_grid = np.meshgrid(x_coord, y_coord)

        x_grid, y_grid = pyproj.transform(
            s_srs, t_srs, x_grid.flatten(), y_grid.flatten()
        )
        x_grid = x_grid.reshape((y_coord.size, x_coord.size))
        y_grid = y_grid.reshape((y_coord.size, x_coord.size))
        geodata["X_grid"] = x_grid
        geodata["Y_grid"] = y_grid

    # Reproject extent on fall-back projection
    x1, y1 = pyproj.transform(s_srs, t_srs, x1, y1)
    x2, y2 = pyproj.transform(s_srs, t_srs, x2, y2)

    # update geodata
    geodata["projection"] = t_proj4str
    geodata["x1"] = x1
    geodata["x2"] = x2
    geodata["y1"] = y1
    geodata["y2"] = y2
    geodata["regular_grid"] = False
    geodata["xpixelsize"] = None
    geodata["ypixelsize"] = None

    return geodata


def get_geogrid(nlat, nlon, geodata=None):
    """
    Get the geogrid data.
    If geodata is None, a regular grid is returned. In this case, it is assumed that
    the origin of the 2D input data is the upper left corner ("upper").

    However, the origin of the x and y grids corresponds to the bottom left of the
    domain. That is, x and y are sorted in ascending order.

    Parameters
    ----------
    nlat: int
        Number of grid points along the latitude axis
    nlon: int
        Number of grid points along the longitude axis
    geodata:
        geodata: dictionary or None
        Optional dictionary containing geographical information about
        the field.

        If geodata is not None, it must contain the following key-value pairs:

        .. tabularcolumns:: |p{1.5cm}|L|

        +----------------+----------------------------------------------------+
        |        Key     |                  Value                             |
        +================+====================================================+
        |   projection   | PROJ.4-compatible projection definition            |
        +----------------+----------------------------------------------------+
        |    x1          | x-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y1          | y-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    x2          | x-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y2          | y-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    yorigin     | a string specifying the location of the first      |
        |                | element in the data raster w.r.t. y-axis:          |
        |                | 'upper' = upper border, 'lower' = lower border     |
        +----------------+----------------------------------------------------+

    Returns
    -------
    x_grid: 2D array
    y_grid: 2D array
    extent: tuple
        Four-element tuple specifying the extent of the domain according to
        (lower left x, upper right x, lower left y, upper right y).
    regular_grid: bool
        True is the grid is regular. False otherwise.
    origin: str
        Place the [0, 0] index of the array to plot in the upper left or lower left
        corner of the axes. Note that the vertical axes points upward for 'lower' but
        downward for 'upper'.
    """
    if geodata is not None:
        regular_grid = geodata.get("regular_grid", True)
        xmin = min((geodata["x1"], geodata["x2"]))
        xmax = max((geodata["x1"], geodata["x2"]))
        x = np.linspace(xmin, xmax, nlon)
        xpixelsize = np.abs(x[1] - x[0])
        x += xpixelsize / 2.0

        ymin = min((geodata["y1"], geodata["y2"]))
        ymax = max((geodata["y1"], geodata["y2"]))
        y = np.linspace(ymin, ymax, nlat)
        ypixelsize = np.abs(y[1] - y[0])
        y += ypixelsize / 2.0

        extent = (geodata["x1"], geodata["x2"], geodata["y1"], geodata["y2"])
        origin = geodata["yorigin"]
        try:
            proj4_to_cartopy(geodata["projection"])
            x_grid, y_grid = np.meshgrid(x, y)
        except UnsupportedSomercProjection:
            # Define fall-back projection for Swiss data(EPSG:3035)
            # This will work reasonably well for Europe only.
            t_proj4str = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs"
            geodata = reproject_geodata(geodata, t_proj4str, return_grid="coords")
            extent = (
                geodata["x1"],
                geodata["x2"],
                geodata["y1"],
                geodata["y2"],
            )
            x_grid, y_grid = geodata["X_grid"], geodata["Y_grid"]
    else:
        x_grid, y_grid = np.meshgrid(np.arange(nlon), np.arange(nlat))
        extent = (0, nlon - 1, 0, nlat - 1)
        regular_grid = True
        origin = "upper"

    return x_grid, y_grid, extent, regular_grid, origin


def get_basemap_axis(extent, geodata=None, ax=None, map_kwargs=None):
    """
    Safely get a basemap axis. If ax is None, the current axis is returned.

    If geodata is not None and ax is not a cartopy axis already, it creates a basemap
    axis and return it.

    Parameters
    ----------
    extent: tuple
        Four-element tuple specifying the extent of the domain according to
        (lower left x, upper right x, lower left y, upper right y).
    geodata:
        geodata: dictionary or None
        Optional dictionary containing geographical information about
        the field.

        If geodata is not None, it must contain the following key-value pairs:

        .. tabularcolumns:: |p{1.5cm}|L|

        +----------------+----------------------------------------------------+
        |        Key     |                  Value                             |
        +================+====================================================+
        |   projection   | PROJ.4-compatible projection definition            |
        +----------------+----------------------------------------------------+
        |    x1          | x-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y1          | y-coordinate of the lower-left corner of the data  |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    x2          | x-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    y2          | y-coordinate of the upper-right corner of the data |
        |                | raster                                             |
        +----------------+----------------------------------------------------+
        |    yorigin     | a string specifying the location of the first      |
        |                | element in the data raster w.r.t. y-axis:          |
        |                | 'upper' = upper border, 'lower' = lower border     |
        +----------------+----------------------------------------------------+

    ax: axis object
        Optional axis object to use for plotting.
    map_kwargs: dict
        Optional parameters that need to be passed to
        :py:func:`pysteps.visualization.basemaps.plot_geography`.

    Returns
    -------
    ax: axis object
    """

    if map_kwargs is None:
        map_kwargs = dict()

    if ax is None:
        # If no axes is passed, use the current axis.
        ax = plt.gca()

    if (geodata is not None) and (not isinstance(ax, GeoAxesSubplot)):
        # Check `ax` is not a GeoAxesSubplot axis to avoid overwriting the map.
        ax = basemaps.plot_geography(geodata["projection"], extent, **map_kwargs)

    return ax
