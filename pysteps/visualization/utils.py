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
    tokens = proj4str.split("+")

    result = {}
    for t in tokens[1:]:
        if "=" in t:
            k, v = t.split("=")
            result[k] = v.strip()

    return result


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

    km_proj = {
        "lon_0": "central_longitude",
        "lat_0": "central_latitude",
        "lat_ts": "true_scale_latitude",
        "x_0": "false_easting",
        "y_0": "false_northing",
        "k": "scale_factor",
        "zone": "zone",
    }
    km_globe = {"a": "semimajor_axis", "b": "semiminor_axis"}
    km_std = {"lat_1": "lat_1", "lat_2": "lat_2"}

    kw_proj = {}
    kw_globe = {}
    kw_std = {}

    for s in proj.srs.split("+"):
        s = s.split("=")
        if len(s) != 2:
            continue
        k = s[0].strip()
        v = s[1].strip()
        try:
            v = float(v)
        except Exception:
            pass

        if k == "proj":
            if v == "tmerc":
                cl = ccrs.TransverseMercator
            elif v == "laea":
                cl = ccrs.LambertAzimuthalEqualArea
            elif v == "lcc":
                cl = ccrs.LambertConformal
            elif v == "merc":
                cl = ccrs.Mercator
            elif v == "utm":
                cl = ccrs.UTM
            elif v == "stere":
                cl = ccrs.Stereographic
            elif v == "aea":
                cl = ccrs.AlbersEqualArea
            elif v == "aeqd":
                cl = ccrs.AzimuthalEquidistant
            elif v == "somerc":
                # Note: ccrs.epsg(2056) doesn't work because the projection
                # limits are too strict.
                # We'll use the Stereographic projection as an alternative.
                cl = ccrs.Stereographic
            elif v == "geos":
                cl = ccrs.Geostationary
            else:
                raise ValueError("unsupported projection: %s" % v)
        elif k in km_proj:
            kw_proj[km_proj[k]] = v
        elif k in km_globe:
            kw_globe[km_globe[k]] = v
        elif k in km_std:
            kw_std[km_std[k]] = v

    globe = None
    if kw_globe:
        globe = ccrs.Globe(**kw_globe)
    if kw_std:
        kw_proj["standard_parallels"] = (kw_std["lat_1"], kw_std["lat_2"])

    if cl.__name__ == "Mercator":
        kw_proj.pop("false_easting", None)
        kw_proj.pop("false_northing", None)

    return cl(globe=globe, **kw_proj)


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

        X, Y = np.meshgrid(x_coord, y_coord)

        X, Y = pyproj.transform(s_srs, t_srs, X.flatten(), Y.flatten())
        X = X.reshape((y_coord.size, x_coord.size))
        Y = Y.reshape((y_coord.size, x_coord.size))

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
    geodata["X_grid"] = X
    geodata["Y_grid"] = Y

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
