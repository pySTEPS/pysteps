#!/bin/env python
"""
LINDA nowcasts
==============

This example shows how to compute and plot a deterministic and ensemble LINDA
nowcasts using Swiss radar data.

"""

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from pysteps import io, rcparams
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.nowcasts import linda
from pysteps.utils import conversion, dimension
from pysteps.visualization import plot_precip_field

###############################################################################
# Read the input rain rate fields
# -------------------------------

date = datetime.strptime("201701311200", "%Y%m%d%H%M")
data_source = "mch"

# Read the data source information from rcparams
datasource_params = rcparams.data_sources[data_source]

# Find the radar files in the archive
fns = io.find_by_date(
    date,
    datasource_params["root_path"],
    datasource_params["path_fmt"],
    datasource_params["fn_pattern"],
    datasource_params["fn_ext"],
    datasource_params["timestep"],
    num_prev_files=2,
)

# Read the data from the archive
importer = io.get_method(datasource_params["importer"], "importer")
reflectivity, _, metadata = io.read_timeseries(
    fns, importer, **datasource_params["importer_kwargs"]
)

# Convert reflectivity to rain rate
rainrate, metadata = conversion.to_rainrate(reflectivity, metadata)

# Upscale data to 2 km to reduce computation time
rainrate, metadata = dimension.aggregate_fields_space(rainrate, metadata, 2000)

# Plot the most recent rain rate field
plot_precip_field(rainrate[-1, :, :], geodata=metadata)
plt.show()

###############################################################################
# Estimate the advection field
# ----------------------------

# The advection field is estimated using the Lucas-Kanade optical flow
advection = dense_lucaskanade(rainrate, verbose=True)

###############################################################################
# Deterministic nowcast (LINDA-D)
# -------------------------------

# Compute 30-minute LINDA nowcast with 8 parallel workers
# Restrict the number of features to 15 to reduce computation time
rainrate_nowcast = linda.forecast(
    rainrate,
    advection,
    6,
    max_num_features=15,
    add_perturbations=False,
    num_workers=8,
    measure_time=True,
)[0]

# Plot the nowcast
plot_precip_field(
    rainrate_nowcast[-1, :, :],
    geodata=metadata,
    title="LINDA-D (+ 30 min)",
)
plt.show()

###############################################################################
# Probabilistic nowcast (LINDA-P)
# -------------------------------

# Compute 30-minute LINDA nowcast ensemble with 40 members and 8 parallel workers
rainrate_nowcast = linda.forecast(
    rainrate,
    advection,
    6,
    max_num_features=15,
    add_perturbations=True,
    num_ens_members=40,
    num_workers=8,
    measure_time=True,
)[0]

# Plot the ensemble mean
rainrate_ensemble_mean = np.mean(rainrate_nowcast[:, -1, :, :], axis=0)
plot_precip_field(
    rainrate_ensemble_mean,
    geodata=metadata,
    title="LINDA ensemble mean (+ 30 min)",
)
plt.show()

# Plot four ensemble members
fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(221 + i)
    ax = plot_precip_field(
        rainrate_nowcast[i, -1, :, :], geodata=metadata, colorbar=False, axis="off"
    )
    ax.set_title(f"Member {i:02d}")
plt.tight_layout()
plt.show()
