#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:29:23 2020

@author: mfeldman
"""
#%%
"""
Thunderstorm Detection and Tracking - DATing
============================================

This example shows how to use the thunderstorm DATing module. The example is based on
MeteoSwiss radar data and uses the Cartesian composite of maximum reflectivity on a
1 km grid. All default values are tuned to this grid, but can be modified.
The first section demonstrates thunderstorm cell detection and how to plot contours.
The second section demonstrates detection and tracking in combination,
as well as how to plot the resulting tracks.
"""
#%% IMPORT ALL REQUIRED FUNCTIONS

from datetime import datetime
import numpy as np
import sys
import os

from pysteps.visualization import tstorm as tstorm_plot
from pysteps.feature import tstorm as tstorm_detect
from pysteps.tracking import tdating as tstorm_dating
import pysteps

from pysteps import io, rcparams
from pysteps.utils import to_reflectivity


#%% LOAD PYSTEPS EXAMPLE DATA

date = datetime.strptime("201607112100", "%Y%m%d%H%M")
data_source = rcparams.data_sources["mch"]

root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]
timestep = data_source["timestep"]

#%% FIND AND READ FILES
fns = io.archive.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_next_files=20
)

importer = io.get_method(importer_name, "importer")
R, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)
Z, metadata = to_reflectivity(R, metadata)
timelist = metadata["timestamps"]
#%% IDENTIFICATION OF THUNDERSTORMS IN SINGLE TIMESTEP
input_image = Z[5, :, :]
time = timelist[5]
cells_id, labels = tstorm_detect.detection(
    input_image,
    minref=35,
    maxref=48,
    mindiff=6,
    minsize=50,
    minmax=41,
    mindis=10,
    time=time,
)
#%% PLOTTING IDENTIFIED CONTOURS ON COMPOSITE
input_image[input_image < 0] = np.nan
poix = np.array([426.201, 242.057, 452.957, 348.687, 524.7])
poiy = np.array([397.604, 302.408, 259.762, 295.476, 349.79])
tstorm_plot.plot_cart_contour(
    input_image,
    cells_id.cont,
    "contours",
    "",
    "contours.png",
    poix=poix,
    poiy=poiy,
)
#%% COMPUTATION OF THUNDERSTORM TRACKS OVER ENTIRE TIMELINE
track_list, cell_list, label_list = tstorm_dating.dating(
    input_video=Z, timelist=timelist, mintrack=3, cell_list=[], label_list=[], start=0
)
#%% PLOTTING EXAMPLE TRACKS WITH RADAR LOCATIONS
poix = np.array([426.201, 242.057, 452.957, 348.687, 524.7])
poiy = np.array([397.604, 302.408, 259.762, 295.476, 349.79])
tstorm_plot.plot_track(
    track_list,
    "tracks",
    "",
    "tracks_c.png",
    710,
    640,
    poix=poix,
    poiy=poiy,
)
