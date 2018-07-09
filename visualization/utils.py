"""Miscellaneous utility functions."""

def parse_proj4_string(proj4str, parse_type="default"):
    """Construct a dictionary from a proj 4 string.
    
    Parameters
    ----------
    proj4str : str
      A proj.4-compatible projection string.
    parse_type : str
      The valid options are 'default'=take each token (beginning with '+') in 
      proj4str as is, 'basemap'=convert the keys to be compatible with Basemap.
    
    Returns
    -------
    out : dict
      Dictionary, where keys and values are parsed from the projection parameter 
      tokens beginning with '+'.
    """
    if parse_type not in ["default", "basemap"]:
        raise ValueError("invalid parse type: must be 'default' or 'basemap'")
    
    tokens = proj4str.split('+')
    result = {}
    for t in tokens[1:]:
        if '=' in t:
            k,v = t.split('=')
            v = v.strip()
            if parse_type == "basemap":
                if k == "proj":
                    # TODO: Make sure that the proj.4 projection type is in all cases 
                    # mapped to the corresponding (or closest matching) Basemap projection.
                    if v not in ["latlon", "latlong", "lonlat", "longlat"]:
                        result["projection"] = v
                    else:
                        result["projection"] = "cyl"
                elif k == "lon_0" or k == "lat_0" or k == "lat_ts":
                    # TODO: Check that east/west and north/south hemispheres are 
                    # handled correctly.
                    if v[-1] in ["E", "N", "S", "W"]:
                        v = v[:-1]
                    result[k] = float(v)
                elif k == "ellps":
                    result[k] = v
                elif k == "LL_lat":
                    result["llcrnrlat"] = float(v)
                elif k == "LL_lon":
                    result["llcrnrlon"] = float(v)
                elif k == "UR_lat":
                    result["urcrnrlat"] = float(v)
                elif k == "UR_lon":
                    result["urcrnrlon"] = float(v)
                elif k == "R":
                    result["rsphere"] = float(v)
                elif k in ["k", "k0"]:
                    result["k_0"] = float(v)
            else:
                result[k] = v
    
    return result
