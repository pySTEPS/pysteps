# -*- coding: utf-8 -*-
"""
xarray and pysteps, just good friends
=====================================

This tutorial shows how the xarray data model could be used to simplify
reading/processing/plotting in pysteps.

Credits
-------
This tutorial is an adaption of the original gist by kmuehlbauer:
https://gist.github.com/kmuehlbauer/645e42a53b30752230c08c20a9c964f9
"""

import xarray as xr
import pysteps as sp

import datetime as dt
import matplotlib.pyplot as pl
import numpy as np
import os

###############################################################################
# How to use xarray with pysteps
# ------------------------------
#
# First, we will show how a netcdf file is currenty imported in the pysteps
# workflow and how this can be done using xarray.

# The path to an example BOM netcdf file
root_path = sp.rcparams.data_sources["bom"]["root_path"]
filename = os.path.join(
    root_path, "prcp-cscn/2/2018/06/16/2_20180616_150000.prcp-cscn.nc"
)

###############################################################################
# pysteps original
# ~~~~~~~~~~~~~~~~

# Read data
R, __, metadata = sp.io.import_bom_rf3(filename)
metadata

# Visualization
sp.visualization.plot_precip_field(
    R,
    map=None,# "cartopy",
    type="depth",
    units="mm",
    geodata=metadata,
    drawlonlatlines=True,
)

###############################################################################
# pysteps using xarray
# ~~~~~~~~~~~~~~~~~~~~

# Read data
bom = xr.open_mfdataset(filename)
print(bom)

# get the pysteps colormap, norm and clevels
cmap, norm, clevs, clevsStr = sp.visualization.precipfields.get_colormap(
    "depth", "mm", "pysteps"
)

# xarray powered plotting
fig = pl.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

cbar_kwargs = {
    "ticks": clevs,
    "norm": norm,
    "extend": "max",
    "fraction": 0.040,
}

bom.precipitation.plot(
    ax=ax,
    cmap=cmap,
    norm=norm,
    add_colorbar=True,
    cbar_kwargs=cbar_kwargs,
)
pl.show()

###############################################################################
# Easily read multiple consecutive datasets into one xarray Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The reading of consecutive datasets is really simple using xarray, which simplifies
# the workflow, particularly for netcdf source data.

root_path = sp.rcparams.data_sources["bom"]["root_path"]
filename = os.path.join(
    root_path, "prcp-cscn/2/2018/06/16/2_20180616_1*0000.prcp-cscn.nc"
)

bom = xr.open_mfdataset(filename, concat_dim="valid_time")
print(bom.precipitation)

###############################################################################
# With other data formats
# -----------------------
#
# In principle the same can be also applied to other data formats, although
# some tweaking is needed.

###############################################################################
# Reading GIF
# ~~~~~~~~~~~
#
# MeteoSwiss uses a GIF format to store its radar composite data.
# First, we will use the pysteps methods to retrieve filenames.

date = dt.datetime.strptime("201505151630", "%Y%m%d%H%M")
data_source = sp.rcparams.data_sources["mch"]

root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]
timestep = data_source["timestep"]

# Find the input files from the archive
fns = sp.io.archive.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_prev_files=9
)

importer = sp.io.get_method(importer_name, "importer")

###############################################################################
# Import GIF file into xarray
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_mch(filename, timestep, **importer_kwargs):
    R, quality, meta = importer(filename, **importer_kwargs)
    x1 = meta["x1"]
    y1 = meta["y1"]
    xsize = meta["xpixelsize"]
    ysize = meta["ypixelsize"]

    ds = xr.Dataset(
        {"precipitation": (["y", "x"], R[::-1, :])},
        coords={
            "x": (
                ["x"],
                np.arange(x1 + xsize // 2, x1 + xsize * R.shape[1], xsize),
            ),
            "y": (
                ["y"],
                np.arange(y1 + ysize // 2, y1 + ysize * R.shape[0], ysize),
            ),
            "time": (["time"], [timestep]),
        },
    )
    root = [
        "projection",
        "x1",
        "x2",
        "y1",
        "y2",
        "xpixelsize",
        "ypixelsize",
        "yorigin",
    ]
    prod = [
        "accutime",
        "unit",
        "transform",
        "threshold",
        "zerovalue",
        "institution",
        "product",
    ]
    for key in root:
        ds.attrs.update({key: meta[key]})
    for key in prod:
        ds.precipitation.attrs.update({key: meta[key]})
    return ds


###############################################################################
# Create timeseries
# ~~~~~~~~~~~~~~~~~

def create_timeseries(fns, **importer_kwargs):
    ds = []
    for fname, tstep in zip(*fns):
        ds.append(load_mch(fname, tstep, **importer_kwargs))

    ds = xr.concat(ds, dim="time")
    return ds


ds = create_timeseries(fns, **importer_kwargs)
print(ds)

###############################################################################
# Convert and transform the data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ds["RR"] = sp.utils.to_rainrate(ds.precipitation)
ds["dBZ"] = sp.utils.dB_transform(ds.precipitation)
print(ds)
