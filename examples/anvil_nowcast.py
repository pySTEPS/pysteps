# coding: utf-8

"""
ANVIL nowcast
=============

This example demonstrates how the ANVIL method can predict growth and decay of
precipitation.

Load the libraries.
"""
from datetime import datetime, timedelta
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import numpy as np
from pysteps import motion, io, rcparams, utils
from pysteps.nowcasts import anvil, extrapolation
from pysteps.visualization import plot_precip_field

################################################################################
# Read the input data
# -------------------
#
# The ANVIL method was originally developed to use vertically integrated liquid
# (VIL) as the input data, but the model allows using any two-dimensional input
# fields. Here we use a composite of rain rates.

date = datetime.strptime("201505151620", "%Y%m%d%H%M")

# Read the data source information from rcparams
data_source = rcparams.data_sources["mch"]

root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]

# Find the input files in the archive, use history length of 5 timesteps
filenames = io.archive.find_by_date(date, root_path, path_fmt, fn_pattern,
                                    fn_ext, timestep=5, num_prev_files=5)

# Read the input time series
importer = io.get_method(importer_name, "importer")
rainrate_field, quality, metadata = io.read_timeseries(filenames, importer,
                                                       **importer_kwargs)

# Convert to rain rate (mm/h)
rainrate_field, metadata = utils.to_rainrate(rainrate_field, metadata)

################################################################################
# Compute the advection field
#
# ---------------------------
# Apply the Lucas-Kanade method with the parameters given in Pulkkinen et al.
# (2020) to compute the advection field.

fd_kwargs = {}
fd_kwargs["max_corners"] = 1000
fd_kwargs["quality_level"] = 0.01
fd_kwargs["min_distance"] = 2
fd_kwargs["block_size"] = 8

lk_kwargs = {}
lk_kwargs["winsize"] = (15, 15)

oflow_kwargs = {}
oflow_kwargs["fd_kwargs"] = fd_kwargs
oflow_kwargs["lk_kwargs"] = lk_kwargs
oflow_kwargs["decl_scale"] = 10

oflow = motion.get_method("lucaskanade")

# transform the input data to logarithmic scale
rainrate_field_log, _ = utils.transformation.dB_transform(rainrate_field,
                                                          metadata=metadata)
velocity = oflow(rainrate_field_log, **oflow_kwargs)

##################################################################################
# Compute extrapolation and ANVIL nowcasts and threshold rain rates below 0.5 mm/h
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
forecast_extrap = extrapolation.forecast(rainrate_field[-1], velocity, 3,
                                         extrap_kwargs={"allow_nonfinite_values": True})
forecast_extrap[forecast_extrap < 0.5] = 0.0

forecast_anvil = anvil.forecast(rainrate_field[-4:], None, velocity, 3,
                                ar_window_radius=25, ar_order=2)
forecast_anvil[forecast_anvil < 0.5] = 0.0

#########################################################################
# Read the reference observation field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
filenames = io.archive.find_by_date(date, root_path, path_fmt, fn_pattern,
                                    fn_ext, timestep=5, num_next_files=3)

refobs_field, quality, metadata = io.read_timeseries(filenames, importer,
                                                     **importer_kwargs)

refobs_field, metadata = utils.to_rainrate(refobs_field[-1], metadata)
refobs_field[refobs_field < 0.5] = 0.0

############################################################################
# Plot the observed rain rate fields and extrapolation and ANVIL nowcasts.
# Mark growth and decay areas with red and blue circles, respectively.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_growth_decay_circles(ax):
    circle = plt.Circle((360, 300), 25, color="b", clip_on=False, fill=False,
                        zorder=1e9)
    ax.add_artist(circle)
    circle = plt.Circle((420, 350), 30, color="b", clip_on=False, fill=False,
                        zorder=1e9)
    ax.add_artist(circle)
    circle = plt.Circle((405, 380), 30, color="b", clip_on=False, fill=False,
                        zorder=1e9)
    ax.add_artist(circle)
    circle = plt.Circle((420, 500), 25, color="b", clip_on=False, fill=False,
                        zorder=1e9)
    ax.add_artist(circle)
    circle = plt.Circle((480, 535), 30, color="b", clip_on=False, fill=False,
                        zorder=1e9)
    ax.add_artist(circle)
    circle = plt.Circle((330, 470), 35, color="b", clip_on=False, fill=False,
                        zorder=1e9)
    ax.add_artist(circle)
    circle = plt.Circle((505, 205), 30, color="b", clip_on=False, fill=False,
                        zorder=1e9)
    ax.add_artist(circle)
    circle = plt.Circle((440, 180), 30, color="r", clip_on=False, fill=False,
                        zorder=1e9)
    ax.add_artist(circle)
    circle = plt.Circle((590, 240), 30, color="r", clip_on=False, fill=False,
                        zorder=1e9)
    ax.add_artist(circle)
    circle = plt.Circle((585, 160), 15, color="r", clip_on=False, fill=False,
                        zorder=1e9)
    ax.add_artist(circle)

plt.figure()

ax = plt.subplot(221)
rainrate_field[-1][rainrate_field[-1] < 0.5] = 0.0
plot_precip_field(rainrate_field[-1])
plot_growth_decay_circles(ax)
ax.set_title("Obs. %s" % str(date))

ax = plt.subplot(222)
plot_precip_field(refobs_field)
plot_growth_decay_circles(ax)
ax.set_title("Obs. %s" % str(date + timedelta(minutes=15)))

ax = plt.subplot(223)
plot_precip_field(forecast_extrap[-1])
plot_growth_decay_circles(ax)
ax.set_title("Extrapolation +15 minutes")

ax = plt.subplot(224)
plot_precip_field(forecast_anvil[-1])
plot_growth_decay_circles(ax)
ax.set_title("ANVIL +15 minutes")
plt.show()
