#!/bin/env python
"""
Plot precipitation using custom colormap
=============

This tutorial shows how to plot data using a custom colormap with a specific
range of precipitation values.

"""

import os
from datetime import datetime
import matplotlib.pyplot as plt

import pysteps
from pysteps import io, rcparams
from pysteps.utils import conversion
from pysteps.visualization import plot_precip_field
from pysteps.datasets import download_pysteps_data, create_default_pystepsrc


###############################################################################
# Download the data if it is not available
# ----------------------------------------
#
# The following code block downloads datasets from the pysteps-data repository
# if it is not available on the disk. The dataset is used to demonstrate the
# plotting of precipitation data using a custom colormap.

# Check if the pysteps-data repository is available (it would be pysteps-data in pysteps)
# Implies that you are running this script from the `pysteps/examples` folder

if not os.path.exists(rcparams.data_sources["mrms"]["root_path"]):
    download_pysteps_data("pysteps_data")
    config_file_path = create_default_pystepsrc("pysteps_data")
    print(f"Configuration file has been created at {config_file_path}")


###############################################################################
# Read precipitation field
# ------------------------
#
# First thing, load a frame from Multi-Radar Multi-Sensor dataset and convert it
# to precipitation rate in mm/h.

# Define the dataset and the date for which you want to load data
data_source = pysteps.rcparams.data_sources["mrms"]
date = datetime(2019, 6, 10, 0, 2, 0)  # Example date

# Extract the parameters from the data source
root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]
timestep = data_source["timestep"]

# Find the frame in the archive for the specified date
fns = io.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=1
)

# Read the frame from the archive
importer = io.get_method(importer_name, "importer")
R, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)

# Convert the reflectivity data to rain rate
R, metadata = conversion.to_rainrate(R, metadata)

# Plot the first rainfall field from the loaded data
plt.figure(figsize=(10, 5), dpi=300)
plt.axis("off")
plot_precip_field(R[0, :, :], geodata=metadata, axis="off")

plt.tight_layout()
plt.show()

###############################################################################
# Define the custom colormap
# --------------------------
#
# Assume that the default colormap does not represent the precipitation values
# in the desired range. In this case, you can define a custom colormap that will
# be used to plot the precipitation data and pass the class instance to the
# `plot_precip_field` function.
#
# It essential for the custom colormap to have the following attributes:
#
# - `cmap`: The colormap object.
# - `norm`: The normalization object.
# - `clevs`: The color levels for the colormap.
#
# `plot_precip_field` can handle each of the classes defined in the `matplotlib.colors`
# https://matplotlib.org/stable/api/colors_api.html#colormaps
# There must be as many colors in the colormap as there are levels in the color levels.


# Define the custom colormap

from matplotlib import colors


class ColormapConfig:
    def __init__(self):
        self.cmap = None
        self.norm = None
        self.clevs = None

        self.build_colormap()

    def build_colormap(self):
        # Define the colormap boundaries and colors
        # color_list = ['lightgrey', 'lightskyblue', 'blue', 'yellow', 'orange', 'red', 'darkred']
        color_list = ["blue", "navy", "yellow", "orange", "green", "brown", "red"]

        self.clevs = [0.1, 0.5, 1.5, 2.5, 4, 6, 10]  # mm/hr

        # Create a ListedColormap object with the defined colors
        self.cmap = colors.ListedColormap(color_list)
        self.cmap.name = "Custom Colormap"

        # Set the color for values above the maximum level
        self.cmap.set_over("darkmagenta")
        # Set the color for values below the minimum level
        self.cmap.set_under("none")
        # Set the color for missing values
        self.cmap.set_bad("gray", alpha=0.5)

        # Create a BoundaryNorm object to normalize the data values to the colormap boundaries
        self.norm = colors.BoundaryNorm(self.clevs, self.cmap.N)


# Create an instance of the ColormapConfig class
config = ColormapConfig()

# Plot the precipitation field using the custom colormap
plt.figure(figsize=(10, 5), dpi=300)
plt.axis("off")
plot_precip_field(R[0, :, :], geodata=metadata, axis="off", colormap_config=config)

plt.tight_layout()
plt.show()
