"""
pysteps.visualization.basemaps
==============================

Methods for plotting basemaps using Cartopy or Basemap.

.. autosummary::
    :toctree: ../generated/

    plot_map_basemap
    plot_map_cartopy
"""

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np

try:
    from mpl_toolkits.basemap import Basemap
    basemap_imported = True
except ImportError:
    basemap_imported = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    cartopy_imported = True
except ImportError:
    cartopy_imported = False
    
def plot_map_basemap(bm_params, drawlonlatlines=False, coastlinecolor=(0.3,0.3,0.3),
                    countrycolor=(0.3,0.3,0.3), continentcolor=(0.95,0.95,0.85),
                    lakecolor=(0.65,0.75,0.9), rivercolor=(0.65,0.75,0.9),
                    mapboundarycolor=(0.65,0.75,0.9), lw=0.5):
    """
    Plot coastlines, countries, rivers and meridians/parallels using Basemap.
    
    Parameters
    ----------
    bm_params : optional
        Optional arguments for the Basemap class constructor:
        https://basemaptutorial.readthedocs.io/en/latest/basemap.html
    drawlonlatlines : bool
        Whether to plot longitudes and latitudes.
    coastlinecolor : scalars (r, g, b)
        Coastline color.
    countrycolor : scalars (r, g, b)
        Countrycolor color.    
    continentcolor : scalars (r, g, b)
        Continentcolor color.
    lakecolor : scalars (r, g, b)
        Lakecolor color.
    rivercolor : scalars (r, g, b)
        Rivercolor color.
    mapboundarycolor : scalars (r, g, b)
        Mapboundarycolor color.
    lw : float 
        Line width.
    
    Returns
    -------
    ax : axes
        Basemap axes.
    """
    if not basemap_imported:
        raise MissingOptionalDependency(
            "map='basemap' option passed to plot_map_basemap function"
            "but the basemap package is not installed")
            
    ax = Basemap(**bm_params)
    
    if coastlinecolor is not None:
        ax.drawcoastlines(color=coastlinecolor, linewidth=lw, zorder=0.1)
    if countrycolor is not None:
        ax.drawcountries(color=countrycolor, linewidth=lw, zorder=0.2)
    if rivercolor is not None:
        ax.drawrivers(zorder=0.2, color=rivercolor)
    if continentcolor is not None:
        ax.fillcontinents(color=continentcolor, lake_color=lakecolor, zorder=0)
    if mapboundarycolor is not None:
        ax.drawmapboundary(fill_color=mapboundarycolor, zorder=-1)
    if drawlonlatlines:
        ax.drawmeridians(np.linspace(bm.llcrnrlon, bm.urcrnrlon, 10),
                         color=(0.5,0.5,0.5), linewidth=0.25, labels=[1,0,0,1],
                         fmt="%.1f", fontsize=6)
        ax.drawparallels(np.linspace(bm.llcrnrlat, bm.urcrnrlat, 10),
                         color=(0.5,0.5,0.5), linewidth=0.25, labels=[1,0,0,1],
                         fmt="%.1f", fontsize=6)

    return ax

def plot_map_cartopy(crs, extent, scale, drawlonlatlines=False,
                    lw=0.5, subplot=(1,1,1)):
    """
    Plot coastlines, countries, rivers and meridians/parallels using Cartopy.
    
    Parameters
    ----------
    crs : object
        Instance of a crs class defined in cartopy.crs.
        It can be created using utils.proj4_to_cartopy.
    extent : scalars (left, right, bottom, top)
        The coordinates of the bounding box.
    drawlonlatlines : bool
        Whether to plot longitudes and latitudes.
    scale : {'10m', '50m'}
        Scale of the shapefile (generalization level).
    lw : float 
        Line width.
    subplot : scalars (nrows, ncols, index)
        Subplot dimensions (n_ros, n_cols) and subplot number (index).
    
    Returns
    -------
    ax : axes
        Cartopy axes. Compatible with matplotlib.
    """
    if not cartopy_imported:
        raise MissingOptionalDependency(
            "map='cartopy' option passed to plot_map_cartopy function"
            "but the cartopy package is not installed")
            
    if isinstance(subplot, gridspec.SubplotSpec):
        ax = plt.subplot(subplot, projection=crs)
    else:
        ax = plt.subplot(subplot[0], subplot[1], subplot[2], projection=crs)

    ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", scale = "50m" if scale is "10m" else scale,
        edgecolor="none", facecolor=np.array([0.59375, 0.71484375, 0.8828125])), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land",
       scale=scale, edgecolor="none", facecolor=np.array([0.9375, 0.9375, 0.859375])), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", scale=scale,
        edgecolor="black", facecolor="none", linewidth=lw), zorder=2)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", scale=scale,
        edgecolor="none", facecolor=np.array([0.59375, 0.71484375, 0.8828125])), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines",
        scale=scale, edgecolor=np.array([ 0.59375, 0.71484375, 0.8828125]),
        facecolor="none"), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land",
        scale=scale, edgecolor="black", facecolor="none", linewidth=lw), zorder=2)

    if drawlonlatlines:
        ax.gridlines(crs=ccrs.PlateCarree())

    ax.set_extent(extent, crs)

    return ax