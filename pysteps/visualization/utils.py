"""
pysteps.visualization.utils
===========================

Miscellaneous utility functions for the visualization module.

.. autosummary::
    :toctree: ../generated/

    parse_proj4_string
    proj4_to_basemap
    proj4_to_cartopy
    reproject_geodata
"""
import numpy as np
from pysteps.exceptions import MissingOptionalDependency, UnsupportedSomercProjection

try:
    import cartopy.crs as ccrs
    cartopy_imported = True
except ImportError:
    cartopy_imported = False
try:
    import pyproj
    pyproj_imported = True
except ImportError:
    pyproj_imported = False

def parse_proj4_string(proj4str):
    """Construct a dictionary from a PROJ.4 projection string.

    Parameters
    ----------
    proj4str : str
      A PROJ.4-compatible projection string.

    Returns
    -------
    out : dict
      Dictionary, where keys and values are parsed from the projection parameter
      tokens beginning with '+'.

    """
    tokens = proj4str.split('+')

    result = {}
    for t in tokens[1:]:
        if '=' in t:
            k,v = t.split('=')
            result[k] = v.strip()

    return result

def proj4_to_basemap(proj4str):
    """Convert a PROJ.4 projection string into a dictionary that can be expanded
    as keyword arguments to mpl_toolkits.basemap.Basemap.__init__.

    Parameters
    ----------
    proj4str : str
        A PROJ.4-compatible projection string.

    Returns
    -------
    out : dict
        The output dictionary.

    """
    pdict = parse_proj4_string(proj4str)
    odict = {}

    for k,v in list(pdict.items()):
        if k == "proj":
            # TODO: Make sure that the proj.4 projection type is in all cases
            # mapped to the corresponding (or closest matching) Basemap projection.
            if v not in ["latlon", "latlong", "lonlat", "longlat"]:
                odict["projection"] = v
            else:
                odict["projection"] = "cyl"
        elif k == "lon_0" or k == "lat_0" or k == "lat_ts":
            # TODO: Check that east/west and north/south hemispheres are
            # handled correctly.
            if v[-1] in ["E", "N", "S", "W"]:
                v = v[:-1]
            odict[k] = float(v)
        elif k == "ellps":
            odict[k] = v
        elif k == "R":
            odict["rsphere"] = float(v)
        elif k in ["k", "k0"]:
            odict["k_0"] = float(v)

    return odict

def proj4_to_cartopy(proj4str):
    """Convert a PROJ.4 projection string into a Cartopy coordinate reference
    system (crs) object.

    Parameters
    ----------
    proj4str : str
        A PROJ.4-compatible projection string.

    Returns
    -------
    out : object
        Instance of a crs class defined in cartopy.crs.

    """
    if not cartopy_imported:
        raise MissingOptionalDependency(
            "cartopy package is required for proj4_to_cartopy function utility "
            "but it is not installed")

    if not pyproj_imported:
        raise MissingOptionalDependency(
            "pyproj package is required for proj4_to_cartopy function utility "
            "but it is not installed")

    proj = pyproj.Proj(proj4str)

    if proj.is_latlong():
        return ccrs.PlateCarree()

    km_proj = {"lon_0": "central_longitude",
               "lat_0": "central_latitude",
               "lat_ts": "true_scale_latitude",
               "x_0": "false_easting",
               "y_0": "false_northing",
               "k": "scale_factor",
               "zone": "zone"}
    km_globe = {'a': "semimajor_axis",
                'b': "semiminor_axis"}
    km_std = {"lat_1": "lat_1",
              "lat_2": "lat_2"}

    kw_proj  = {}
    kw_globe = {}
    kw_std   = {}

    for s in proj.srs.split('+'):
        s = s.split('=')
        if len(s) != 2:
            continue
        k = s[0].strip()
        v = s[1].strip()
        try:
            v = float(v)
        except:
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
            elif v == "somerc":
                raise UnsupportedSomercProjection("unsupported projection: somerc")
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
        kw_proj.pop("false_easting",  None)
        kw_proj.pop("false_northing", None)

    return cl(globe=globe, **kw_proj)

def reproject_geodata(geodata, t_proj4str, return_grid=None):
    """
    Reproject geodata and optionally create a grid in a new projection.
    
    Parameters
    ----------
    geodata : dictionary
        Dictionary containing geographical information about the field.
        It must contain the attributes projection, x1, x2, y1, y2, xpixelsize, 
        ypixelsize, as defined in the documentation of pysteps.io.importers.
    t_proj4str: str
        The target PROJ.4-compatible projection string (fallback).
    return_grid : {None, 'coords', 'quadmesh'}, optional
        Whether to return the coordinates of the projected grid.
        The default return_grid=None does not compute the grid,
        return_grid='coords' returns the centers of projected grid points,
        return_grid='quadmesh' returns the coordinates of the quadrilaterals
        (e.g. to be used by pcolormesh).
    
    Returns
    -------
    geodata : dictionary
        Dictionary containing the reprojected geographical information 
        and optionally the required X_grid and Y_grid. \n
        It also includes a fixed boolean attribute regular_grid=False to indicate
        that the reprojected grid has no regular spacing.
    """
    if not pyproj_imported:
        raise MissingOptionalDependency(
            "pyproj package is required for reproject_geodata function utility "
            "but it is not installed")
    
    geodata = geodata.copy()
    s_proj4str = geodata["projection"]
    extent = (geodata["x1"], geodata["x2"], geodata["y1"], geodata["y2"])
    shape = (int((geodata["y2"]-geodata["y1"])/geodata["ypixelsize"]),
            int((geodata["x2"]-geodata["x1"])/geodata["xpixelsize"]))
    
    s_srs = pyproj.Proj(s_proj4str)
    t_srs = pyproj.Proj(t_proj4str)
    
    x1 = extent[0]
    x2 = extent[1]
    y1 = extent[2]
    y2 = extent[3]
    
    # Reproject grid on fall-back projection
    if return_grid is not None:
        if return_grid == "coords":
            y_coord = np.linspace(y1, y2, shape[0]) + geodata["ypixelsize"]/2.0
            x_coord = np.linspace(x1, x2, shape[1]) + geodata["xpixelsize"]/2.0
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