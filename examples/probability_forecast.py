#!/bin/env python
"""
Probability forecasts
=====================

This example script shows how to forecast the probability of exceeding an
intensity threshold.

The method is based on the local Lagrangian approach described in Germann and
Zawadzki (2004).
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pysteps.nowcasts.lagrangian_probability import forecast
from pysteps.visualization import plot_precip_field

###############################################################################
# Numerical example
# -----------------
#
# First, we use some dummy data to show the basic principle of this approach.
# The probability forecast is produced by sampling a spatial neighborhood that is
# increased as a function of lead time. As a result, the edges of
# the yellow square becomes more and more smooth as t increases. This represents
# the strong loss of predictability with lead time of any extrapolation nowcast.

# parameters
precip = xr.DataArray(
    np.zeros((100, 100), dtype=float),
    dims=("y", "x"),
    coords={"y": np.arange(100), "x": np.arange(100)},
    name="synthetic_precip",
)
precip.loc[dict(y=slice(10, 49), x=slice(10, 49))] = 1.0

# constant unit velocity field (1 px/step in x AND y)
motion = xr.Dataset(
    data_vars=dict(
        velocity_x=(["y", "x"], np.ones_like(precip.values)),
        velocity_y=(["y", "x"], np.ones_like(precip.values)),
    ),
    coords=precip.coords,
)

# assemble the single-time input dataset required by forecast()
timesteps = [1, 2, 6, 12]   # start at 1 (t0+1, t0+2, ...)
thr = 0.5
slope = 1  # pixels / timestep

toy_time0 = np.datetime64("2000-01-01T00:00")
toy_ds = xr.Dataset(
    data_vars={
        "precip_intensity": (("time", "y", "x"), precip.values[None, ...]),
        "velocity_x": (("y", "x"), motion["velocity_x"].values),
        "velocity_y": (("y", "x"), motion["velocity_y"].values),
    },
    coords={"time": [toy_time0], "y": precip["y"], "x": precip["x"]},
    attrs={"precip_var": "precip_intensity"},
)
# REQUIRED by extrapolation.convert_output_to_xarray_dataset: seconds per step
toy_ds["time"].attrs["stepsize"] = 60  # 1 minute per step in the toy example

# compute probability forecast 
out = forecast(toy_ds, timesteps, thr, slope=slope)

# plot
for n in range(out.sizes["time"]):
    plt.subplot(2, 2, n + 1)
    plt.imshow(out["precip_intensity"].isel(time=n).values, interpolation="nearest", vmin=0, vmax=1)
    plt.title(f"t={timesteps[n]}")
    plt.xticks([])
    plt.yticks([])
plt.show()

###############################################################################
# Real-data example
# -----------------
#
# We now apply the same method to real data. We use a slope of 1 km / minute
# as suggested by  Germann and Zawadzki (2004), meaning that after 30 minutes,
# the probabilities are computed by using all pixels within a neighborhood of 30
# kilometers.

from datetime import datetime
from pysteps import io, rcparams
from pysteps.utils import conversion
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.verification import reldiag_init, reldiag_accum, plot_reldiag

# data source
root = rcparams.data_sources["mch"]["root_path"]
fmt = rcparams.data_sources["mch"]["path_fmt"]
pattern = rcparams.data_sources["mch"]["fn_pattern"]
ext = rcparams.data_sources["mch"]["fn_ext"]
timestep = rcparams.data_sources["mch"]["timestep"]  # minutes per native step
importer_name = rcparams.data_sources["mch"]["importer"]
importer_kwargs = rcparams.data_sources["mch"]["importer_kwargs"]

# read precip field: last 3 frames ending at t0 (for motion estimation)
date = datetime.strptime("201607112100", "%Y%m%d%H%M")
fns = io.archive.find_by_date(date, root, fmt, pattern, ext, timestep, num_prev_files=2)
importer = io.get_method(importer_name, "importer")
ds_in = io.read_timeseries(fns, importer, **importer_kwargs)

# convert to rain rate
ds_in = conversion.to_rainrate(ds_in)
ds_in.attrs.setdefault("precip_var", "precip_intensity")
precip_var = ds_in.attrs["precip_var"]

# ensure 'stepsize' (seconds) on the time coordinate
ds_in["time"].attrs.setdefault("stepsize", int(timestep) * 60)

# estimate motion on the 3-frame dataset (adds velocity_x/velocity_y)
ds_with_vel = dense_lucaskanade(ds_in)

# build the single-time dataset for forecast: keep time dim of size 1
ds0 = ds_with_vel.isel(time=[-1])  # NOT isel(time=-1), we must keep the time dimension
# carry over stepsize (safeguard)
ds0["time"].attrs.setdefault("stepsize", int(timestep) * 60)

# parameters
nleadtimes = 6
thr = 1  # mm / h
slope = 1 * timestep  # km / min → pixels per step if ~1 km/pixel

# compute probability forecast
extrap_kwargs = dict(allow_nonfinite_values=True)
fct = forecast(ds0, nleadtimes, thr, slope=slope)

# plot raw probability maps
for n in range(fct.sizes["time"]):
    plt.subplot(2, 3, n + 1)
    plt.imshow(fct[precip_var].isel(time=n).values, interpolation="nearest", vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
plt.show()

################################################################################
# Let's plot one single leadtime in more detail using the pysteps visualization
# functionality.

# Build geodata for plotting
metadata = {
    "projection": ds_in.attrs.get("projection", None),
    "x1": float(ds_in["x"].values[0]),
    "x2": float(ds_in["x"].values[-1]),
    "y1": float(ds_in["y"].values[0]),
    "y2": float(ds_in["y"].values[-1]),
    "yorigin": "lower",
}

plt.close()
plot_precip_field(
    fct[precip_var].isel(time=2).values,
    geodata=metadata,
    ptype="prob",
    probthr=thr,
    title="Exceedence probability (+ %i min)" % (nleadtimes * timestep),
)
plt.show()

###############################################################################
# Verification
# ------------

# verifying observations
fns = io.archive.find_by_date(date, root, fmt, pattern, ext, timestep,
                              num_next_files=nleadtimes)
ds_obs = io.read_timeseries(fns, importer, **importer_kwargs)
ds_obs = conversion.to_rainrate(ds_obs)
ds_obs.attrs.setdefault("precip_var", precip_var)
obs = ds_obs[precip_var]

# Align obs to forecast lead times (skip t0)
n_fc = fct.sizes["time"]
# safe slice in case ds_obs has exactly n_fc+1 frames
obs = obs.isel(time=slice(1, 1 + n_fc))

# Now shapes match: (n_fc, y, x)
fct_np = fct[precip_var].values
obs_np = np.nan_to_num(obs.values, nan=0.0)

# reliability diagram
reldiag = reldiag_init(thr)
reldiag_accum(reldiag, fct_np, obs_np)
fig, ax = plt.subplots()
plot_reldiag(reldiag, ax)
ax.set_title("Reliability diagram")
plt.show()

# sphinx_gallery_thumbnail_number = 3