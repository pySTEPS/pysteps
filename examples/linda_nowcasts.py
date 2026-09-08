#!/bin/env python
"""
LINDA nowcasts
==============

This example shows how to compute and plot a deterministic and ensemble LINDA
nowcasts using Swiss radar data.

"""

from datetime import datetime
import warnings

warnings.simplefilter("ignore")

import matplotlib.pyplot as plt

from pysteps import io, rcparams
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.nowcasts import linda, sprog, steps
from pysteps.utils import conversion, dimension, transformation
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
precip_dataset = io.read_timeseries(
    fns, importer, **datasource_params["importer_kwargs"]
)

# Convert reflectivity to rain rate
precip_dataset = conversion.to_rainrate(precip_dataset)
precip_var = precip_dataset.attrs["precip_var"]

# Upscale data to 2 km to reduce computation time
precip_dataset = dimension.aggregate_fields_space(precip_dataset, 2000)

# Build the geodata dict expected by the plotting routines
geodata = {
    "projection": precip_dataset.attrs["projection"],
    "x1": precip_dataset.x.values[0],
    "x2": precip_dataset.x.values[-1],
    "y1": precip_dataset.y.values[0],
    "y2": precip_dataset.y.values[-1],
    "yorigin": "lower",
}

# Plot the most recent rain rate field
plt.figure()
plot_precip_field(precip_dataset[precip_var][-1])
plt.show()

###############################################################################
# Estimate the advection field
# ----------------------------

# The advection field is estimated using the Lucas-Kanade optical flow
precip_dataset_w_motion = dense_lucaskanade(precip_dataset, verbose=True)

###############################################################################
# Deterministic nowcast
# ---------------------

# Compute 30-minute LINDA nowcast with 8 parallel workers
# Restrict the number of features to 15 to reduce computation time
nowcast_linda_dataset = linda.forecast(
    precip_dataset_w_motion,
    6,
    max_num_features=15,
    add_perturbations=False,
    num_workers=8,
    measure_time=True,
)[0]
nowcast_linda = nowcast_linda_dataset[precip_var].values

# Compute S-PROG nowcast for comparison
precip_dataset_w_motion_db = transformation.dB_transform(
    precip_dataset_w_motion, threshold=0.1, zerovalue=-15.0
)
nowcast_sprog_dataset = sprog.forecast(
    precip_dataset_w_motion_db.isel(time=slice(-3, None)),
    6,
    n_cascade_levels=6,
    precip_thr=-10.0,
)

# Convert reflectivity nowcast to rain rate
nowcast_sprog_dataset = transformation.dB_transform(
    nowcast_sprog_dataset, threshold=-10.0, inverse=True
)
nowcast_sprog = nowcast_sprog_dataset[precip_var].values

# Plot the nowcasts
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 2, 1)
plot_precip_field(
    nowcast_linda[-1, :, :],
    title="LINDA (+ 30 min)",
)

ax = fig.add_subplot(1, 2, 2)
plot_precip_field(
    nowcast_sprog[-1, :, :],
    title="S-PROG (+ 30 min)",
)

plt.show()

###############################################################################
# The above figure shows that the filtering scheme implemented in LINDA preserves
# small-scale and band-shaped features better than S-PROG. This is because the
# former uses a localized elliptical convolution kernel instead of the
# cascade-based autoregressive process, where the parameters are estimated over
# the whole domain.

###############################################################################
# Probabilistic nowcast
# ---------------------

# Compute 30-minute LINDA nowcast ensemble with 40 members and 8 parallel workers
nowcast_linda_dataset = linda.forecast(
    precip_dataset_w_motion,
    6,
    max_num_features=15,
    add_perturbations=True,
    vel_pert_method=None,
    n_ens_members=40,
    num_workers=8,
    measure_time=True,
    seed=42,  # Fixed seed for reproducible ensemble members
)[0]
nowcast_linda = nowcast_linda_dataset[precip_var].values

# Compute 40-member STEPS nowcast for comparison
nowcast_steps_dataset = steps.forecast(
    precip_dataset_w_motion_db.isel(time=slice(-3, None)),
    6,
    40,
    n_cascade_levels=6,
    precip_thr=-10.0,
    mask_method="incremental",
    kmperpixel=2.0,
    timestep=datasource_params["timestep"],
    vel_pert_method=None,
    seed=42,  # Fixed seed for reproducible ensemble members
)

# Convert reflectivity nowcast to rain rate
nowcast_steps_dataset = transformation.dB_transform(
    nowcast_steps_dataset, threshold=-10.0, inverse=True
)
nowcast_steps = nowcast_steps_dataset[precip_var].values

# Plot two ensemble members of both nowcasts
fig = plt.figure()
for i in range(2):
    ax = fig.add_subplot(2, 2, i + 1)
    ax = plot_precip_field(
        nowcast_linda[i, -1, :, :], geodata=geodata, colorbar=False, axis="off"
    )
    ax.set_title(f"LINDA Member {i+1}")

for i in range(2):
    ax = fig.add_subplot(2, 2, 3 + i)
    ax = plot_precip_field(
        nowcast_steps[i, -1, :, :], geodata=geodata, colorbar=False, axis="off"
    )
    ax.set_title(f"STEPS Member {i+1}")

###############################################################################
# The above figure shows the main difference between LINDA and STEPS. In
# addition to the convolution kernel, another improvement in LINDA is a
# localized perturbation generator using the short-space Fourier transform
# (SSFT) and a spatially variable marginal distribution. As a result, the
# LINDA ensemble members preserve the anisotropic and small-scale structures
# considerably better than STEPS.

plt.tight_layout()
plt.show()
