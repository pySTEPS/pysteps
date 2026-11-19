
# -*- coding: utf-8 -*-
"""
Compare four nowcasting configurations:

1. STEPS with xarray
2. STEPS with STAN script with QPE only (var1=QPE, var2=None)
3. STAN with xarray (var1=VIL, var2=QPE)
4. STAN with NumPy (var1=VIL, var2=QPE)

Only the first ensemble member is compared.

Download example data:
https://github.com/user-attachments/files/34318168/input_QPE_VIL_fields.zip
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from datetime import datetime

# ================================================================
# PATHS AND IMPORTS
# ================================================================

os.chdir(
    "C:/Workdir/Develop/Data/OpenNowcastingHackathon/pysteps_stan"
)

from pysteps import motion, nowcasts
from pysteps.utils import transformation
from pysteps.visualization import plot_precip_field


root_path = "C:/Workdir/Develop/Data/OpenNowcastingHackathon"
obs_root = f"{root_path}/obs"


# ================================================================
# CONFIGURATION
# ================================================================

# Observation thresholds
precip_thr = 0.1       # mm/h
vil_thr = 10            # g/m2

# Convert VIL to kg/m2
vil_thr_kg = vil_thr / 1000

# Convert thresholds to dB
precip_thr_db = 10 * np.log10(precip_thr)
vil_thr_db_kg = 10 * np.log10(vil_thr_kg)

# Values used to replace values below threshold
precip_nan_db = precip_thr_db - 5
vil_kg_nan_db = vil_thr_db_kg - 5


# ================================================================
# NOWCAST PARAMETERS
# ================================================================

timesteps = 12
n_ens_members = 1
n_cascade_levels = 6

# STAN parameters
ar_window_radius = "central"

noise_method = "nonparametric"
noise_method_var2 = None

vel_pert_method = None

adaptive_localization = int(n_cascade_levels / 2)

prob_conversion = True

# Common parameters
kmperpixel = 1
timestep = 5
seed = 24
num_workers = 4


# ================================================================
# READ OBSERVATIONS
# ================================================================

xr_qpe = xr.open_dataset(
    f"{obs_root}/RADQPE_20210704103000.nc",
    engine="netcdf4",
)

xr_vil = xr.open_dataset(
    f"{obs_root}/RADVIL_20210704103000.nc",
    engine="netcdf4",
)


# ================================================================
# QPE PREPROCESSING
# ================================================================

precip = xr_qpe["RADQPE"].values.copy()

precip_db = 10 * np.log10(precip)
precip_db[precip < precip_thr] = precip_nan_db
precip_db[np.isnan(precip_db)] = precip_nan_db


# ================================================================
# VIL PREPROCESSING
# ================================================================

vil_kg = xr_vil["RADVIL"].values.copy() / 1000

vil_kg_db = 10 * np.log10(vil_kg)
vil_kg_db[vil_kg < vil_thr_kg] = vil_kg_nan_db
vil_kg_db[np.isnan(vil_kg_db)] = vil_kg_nan_db


# ================================================================
# CREATE XARRAY DATASETS
# ================================================================

xr_precip_db = xr_qpe.copy(deep=True)

xr_precip_db["RADQPE"] = (
    xr_qpe["RADQPE"].dims,
    precip_db,
)

xr_precip_db = xr_precip_db.rename(
    {"RADQPE": "precip"}
)

xr_precip_db.attrs["precip_var"] = "precip"


xr_vil_kg_db = xr_vil.copy(deep=True)

xr_vil_kg_db["RADVIL"] = (
    xr_vil["RADVIL"].dims,
    vil_kg_db,
)

xr_vil_kg_db = xr_vil_kg_db.rename(
    {"RADVIL": "vil"}
)

xr_vil_kg_db.attrs["proxy_var"] = "vil"


# ================================================================
# OPTICAL FLOW
# ================================================================

print("\nCalculating optical flow...")

oflow_method = motion.get_method("LK")

xr_precip_velocity_db = oflow_method(
    xr_precip_db
)

velocity_x = xr_precip_velocity_db["velocity_x"].values
velocity_y = xr_precip_velocity_db["velocity_y"].values

# NumPy velocity: (2, y, x)
precip_velocity_db = np.stack(
    [velocity_x, velocity_y],
    axis=0,
)


# ================================================================
# CREATE COMBINED XARRAY DATASET
# ================================================================

dataset_QPE = xr_precip_velocity_db.copy(deep=True)
dataset_QPE.attrs["precip_var"] = "precip"


# ================================================================
# ADD VIL
# ================================================================
dataset = dataset_QPE.copy(deep=True)
dataset["vil"] = xr_vil_kg_db["vil"]
dataset.attrs["proxy_var"] = "vil"


# ================================================================
# FIX TIME COORDINATE
# ================================================================

for ds in [dataset_QPE, dataset]:

    time = [
        datetime.strptime(str(t), "%Y%m%d%H%M%S")
        for t in ds["time"].values
    ]

    ds.coords["time"] = (
        "time",
        np.asarray(time, dtype="datetime64[ns]"),
    )

    ds["time"].attrs["stepsize"] = 300.0


# ================================================================
# INPUT INFORMATION
# ================================================================

print("\nInput dimensions:")
print("QPE:", precip_db.shape)
print("VIL:", vil_kg_db.shape)
print("Velocity:", precip_velocity_db.shape)

print("\nCombined Dataset:")
print(dataset)


# ================================================================
# GEOGRAPHICAL METADATA
# ================================================================

metadata = {
    "projection": (
        "+proj=lcc +lat_1=49.83333333333334 "
        "+lat_2=51.16666666666666 "
        "+lat_0=50.797815 +lon_0=4.359215833333333 "
        "+x_0=649328 +y_0=665262 +ellps=GRS80 "
        "+towgs84=0,0,0,0,0,0,0 +units=m +no_defs "
    ),
    "ll_lon": -0.2666973996088157,
    "ll_lat": 47.41679117656605,
    "ur_lon": 9.664159875778674,
    "ur_lat": 53.69199685747096,
    "x1": 300000.0000063947,
    "y1": 300000.0001025071,
    "x2": 999999.9999935812,
    "y2": 1000000.0000994017,
    "xpixelsize": 1000.0,
    "ypixelsize": 1000.0,
    "cartesian_unit": "m",
    "yorigin": "upper",
    "institution": "Odyssey datacentre",
    "accutime": 5.0,
    "unit": "mm/h",
    "transform": None,
    "zerovalue": 0.0,
    "threshold": precip_thr,
}


# ================================================================
# INITIALIZE METHODS
# ================================================================

stan_method = nowcasts.get_method("stan")
steps_method = nowcasts.get_method("steps")


# ================================================================
# 1. STEPS WITH XARRAY
# ================================================================

print("\n" + "=" * 60)
print("1. RUNNING STEPS WITH XARRAY")
print("=" * 60)

# STEPS xarray interface
#
# This call assumes your local STEPS wrapper accepts a Dataset
# containing precip, velocity_x and velocity_y.

fct_steps_xr_db = steps_method(
    dataset=dataset_QPE,
    timesteps=timesteps,
    n_ens_members=n_ens_members,
    n_cascade_levels=n_cascade_levels,
    precip_thr=precip_thr_db,
    noise_method=noise_method,
    vel_pert_method=vel_pert_method,
    conditional=True,
    kmperpixel=kmperpixel,
    timestep=timestep,
    seed=seed,
    num_workers=num_workers,
)

# Inverse dB transformation
fct_steps_xr_db["precip"].attrs["transform"] = "dB"

fct_steps_xr = transformation.dB_transform(
    fct_steps_xr_db,
    threshold=precip_thr_db,
    inverse=True,
)["precip"].values

print("STEPS xarray finished.")
print("Forecast shape:", fct_steps_xr.shape)


# ================================================================
# 2. STEPS RUN FROM STAN SCRIPT WITH QPE ONLY
# ================================================================

print("\n" + "=" * 60)
print("2. RUNNING STAN WITH QPE ONLY (SAME AS STEPS)")
print("=" * 60)

# Use precipitation as var1.
# No secondary variable is supplied.
# No localization in AR model

fct_steps_stan_db = stan_method(
    dataset=dataset_QPE,
    timesteps=timesteps,
    n_ens_members=n_ens_members,
    n_cascade_levels=n_cascade_levels,
    ar_window_radius=None,
    var1_thr=precip_thr_db,
    var1_name="precip",
    noise_method=noise_method,
    noise_method_var2=None,
    vel_pert_method=vel_pert_method,
    adaptive_localization=False,
    prob_conversion=False,
    kmperpixel=kmperpixel,
    timestep=timestep,
    seed=seed,
    num_workers=num_workers,
)

# Inverse dB transformation
fct_steps_stan_db["precip"].attrs["transform"] = "dB"

fct_steps_stan_xr = transformation.dB_transform(
    fct_steps_stan_db,
    threshold=precip_thr_db,
    inverse=True,
)["precip"].values

print("STEPS with STAN script QPE-only run finished.")
print("Forecast shape:", fct_steps_stan_xr.shape)


# ================================================================
# 3. STAN WITH XARRAY (VIL + QPE)
# ================================================================

print("\n" + "=" * 60)
print("3. RUNNING STAN WITH XARRAY (VIL + QPE)")
print("=" * 60)

fct_stan_xr_db = stan_method(
    dataset=dataset,
    timesteps=timesteps,
    n_ens_members=n_ens_members,
    n_cascade_levels=n_cascade_levels,
    ar_window_radius=ar_window_radius,
    var1_thr=vil_thr_db_kg,
    var2_thr=precip_thr_db,
    var1_name="vil",
    var2_name="precip",
    noise_method=noise_method,
    noise_method_var2=noise_method_var2,
    vel_pert_method=vel_pert_method,
    adaptive_localization=adaptive_localization,
    prob_conversion=prob_conversion,
    kmperpixel=kmperpixel,
    timestep=timestep,
    seed=seed,
    num_workers=num_workers,
)

# Inverse dB transformation
fct_stan_xr_db["precip"].attrs["transform"] = "dB"

fct_stan_xr = transformation.dB_transform(
    fct_stan_xr_db,
    threshold=precip_thr_db,
    inverse=True,
)["precip"].values

print("STAN xarray finished.")
print("Forecast shape:", fct_stan_xr.shape)


# ================================================================
# 4. STAN WITH NUMPY (VIL + QPE)
# ================================================================

print("\n" + "=" * 60)
print("4. RUNNING STAN WITH NUMPY (VIL + QPE)")
print("=" * 60)

fct_stan_np_db = stan_method(
    var1=vil_kg_db,
    var2=precip_db,
    velocity=precip_velocity_db,
    timesteps=timesteps,
    n_ens_members=n_ens_members,
    n_cascade_levels=n_cascade_levels,
    ar_window_radius=ar_window_radius,
    var1_thr=vil_thr_db_kg,
    var2_thr=precip_thr_db,
    var1_name="vil",
    var2_name="precip",
    noise_method=noise_method,
    noise_method_var2=noise_method_var2,
    vel_pert_method=vel_pert_method,
    adaptive_localization=adaptive_localization,
    prob_conversion=prob_conversion,
    kmperpixel=kmperpixel,
    timestep=timestep,
    seed=seed,
    num_workers=num_workers,
)

# Inverse dB transformation
fct_stan_np = 10 ** (fct_stan_np_db / 10)

fct_stan_np[fct_stan_np < precip_thr] = 0.0

print("STAN NumPy finished.")
print("Forecast shape:", fct_stan_np.shape)


# ================================================================
# FORECAST COMPARISON
# ================================================================

print("\n" + "=" * 60)
print("FORECAST COMPARISON")
print("=" * 60)

print("STEPS xarray:", fct_steps_xr.shape)
print("STAN QPE only:", fct_steps_stan_xr.shape)
print("STAN xarray:", fct_stan_xr.shape)
print("STAN NumPy:", fct_stan_np.shape)


# ================================================================
# SELECT FIRST ENSEMBLE MEMBER
# ================================================================

# Forecast shape is expected to be:
# (ensemble_members, lead_times, y, x)

# Select final forecast lead time
leadtime_index = -1

# First ensemble member, final lead time
fct_steps_xr_plot = fct_steps_xr[0, leadtime_index]

fct_steps_stan_plot = fct_steps_stan_xr[0, leadtime_index]

fct_stan_xr_plot = fct_stan_xr[0, leadtime_index]

fct_stan_np_plot = fct_stan_np[0, leadtime_index]


# ================================================================
# 2 x 2 PLOT
# ================================================================

forecasts = [
    fct_steps_xr_plot,
    fct_steps_stan_plot,
    fct_stan_xr_plot,
    fct_stan_np_plot,
]

titles = [
    "STEPS - xarray",
    "STEPS with STAN script - QPE only",
    "STAN - xarray (VIL + QPE)",
    "STAN - NumPy (VIL + QPE)",
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9, 8),
    dpi=200,
)

for ax, forecast, title in zip(
    axes.flat,
    forecasts,
    titles,
):

    # Select the current subplot
    plt.sca(ax)
    ax.axis('off')
    
    # Plot precipitation
    plot_precip_field(
        forecast,
        geodata=metadata,
        colorscale="STEPS-BE",
        axis="off",
    )

    ax.set_title(title, fontsize=12)

plt.suptitle(
    f"Nowcast comparison - lead time {timesteps * timestep} min",
    fontsize=15,
    y=0.95
)

plt.show()
plt.close()


# ================================================================
# NUMERICAL COMPARISON
# ================================================================

print("\n" + "=" * 60)
print("NUMERICAL COMPARISON - FIRST ENSEMBLE MEMBER")
print("=" * 60)

for title, forecast in zip(titles, forecasts):

    print(f"\n{title}")

    print(
        "Maximum precipitation:",
        np.nanmax(forecast),
        "mm/h",
    )

    print(
        "Mean precipitation:",
        np.nanmean(forecast),
        "mm/h",
    )


# ================================================================
# CLOSE INPUT DATASETS
# ================================================================

xr_qpe.close()
xr_vil.close()

print("\nAll four experiments completed.")
