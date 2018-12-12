"""Miscellaneous utility functions."""
from pysteps.exceptions import MissingOptionalDependency

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
