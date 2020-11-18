#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thunderstorm Detection and Tracking - DATing
============================================

This example shows how to use the thunderstorm DATing module. The example is based on
MeteoSwiss radar data and uses the Cartesian composite of maximum reflectivity on a
1 km grid. All default values are tuned to this grid, but can be modified.
The first section demonstrates thunderstorm cell detection and how to plot contours.
The second section demonstrates detection and tracking in combination,
as well as how to plot the resulting tracks.

Created on Thu Nov  5 10:29:23 2020

@author: mfeldman
"""
#%% IMPORT ALL REQUIRED FUNCTIONS

import os
import sys
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from pysteps import io, rcparams
from pysteps.feature import tstorm as tstorm_detect
from pysteps.tracking import tdating as tstorm_dating
from pysteps.utils import to_reflectivity
from pysteps.visualization import plot_precip_field, plot_track, plot_cart_contour

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

pprint(metadata)

#%% IDENTIFICATION OF THUNDERSTORMS IN SINGLE TIMESTEP
input_image = Z[5, :, :].copy()
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

#%% COMPUTATION OF THUNDERSTORM TRACKS OVER ENTIRE TIMELINE
track_list, cell_list, label_list = tstorm_dating.dating(
    input_video=Z, timelist=timelist, mintrack=3, cell_list=[], label_list=[], start=0
)

#%% PLOTTING

# Plot precipitation field
plot_precip_field(Z[5, :, :], units=metadata["unit"])

# Plot the identified cells
plot_cart_contour(cells_id.cont)

# Plot tracks
plot_track(
    track_list,
    710,
    640,
)
plt.show()
