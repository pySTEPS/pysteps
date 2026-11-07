# -*- coding: utf-8 -*-
"""

Example to run STAN nowcast using VIL and QPE as inputs.

First download VIL and QPE example data from
https://github.com/user-attachments/files/34318168/input_QPE_VIL_fields.zip

and fix the directories

"""

import os
import numpy as np
import xarray as xr
from datetime import datetime

os.chdir('C:/Workdir/Develop/Data/OpenNowcastingHackathon/pysteps_stan')
from pysteps import motion, nowcasts
from pysteps.utils import transformation

root_path = "C:/Workdir/Develop/Data/OpenNowcastingHackathon"
obs_root = f"{root_path}/obs"

precip_thr = 0.1 #mm/hr
vil_thr = 10 #g m-2
vil_thr_kg = vil_thr / 1000
precip_thr_db = 10*np.log10(precip_thr)
vil_thr_db_kg = 10*np.log10(vil_thr_kg)

precip_nan_db = precip_thr_db - 5
vil_kg_nan_db = vil_thr_db_kg - 5

# Open observations
xr_qpe = xr.open_dataset(
    f"{obs_root}/RADQPE_20210704103000.nc",
    engine="netcdf4"
)

xr_vil = xr.open_dataset(
    f"{obs_root}/RADVIL_20210704103000.nc",
    engine="netcdf4"
)

# --------------------------------
# QPE preprocessing
# --------------------------------

precip = xr_qpe["RADQPE"].values.copy()

precip_db = 10 * np.log10(precip)
precip_db[precip < precip_thr] = precip_nan_db

# --------------------------------
# VIL preprocessing
# --------------------------------

vil_kg = xr_vil["RADVIL"].values.copy() / 1000

vil_kg_db = 10 * np.log10(vil_kg)
vil_kg_db[vil_kg < vil_thr_kg] = vil_kg_nan_db

# --------------------------------
# Create xarray datasets
# --------------------------------

xr_precip_db = xr_qpe.copy(deep=True)
xr_precip_db["RADQPE"] = (
    xr_qpe["RADQPE"].dims,
    precip_db
)
xr_precip_db = xr_precip_db.rename({"RADQPE": "precip"})
xr_precip_db.attrs["precip_var"] = "precip"

xr_vil_kg_db = xr_vil.copy(deep=True)
xr_vil_kg_db["RADVIL"] = (
    xr_vil["RADVIL"].dims,
    vil_kg_db
)
xr_vil_kg_db = xr_vil_kg_db.rename({"RADVIL": "vil"})
xr_vil_kg_db.attrs["proxy_var"] = "vil"

# --------------------------------
# Optical flow
# --------------------------------

oflow_method = motion.get_method("LK")

xr_precip_velocity_db = oflow_method(xr_precip_db)

#individual velocity np array
velocity_x = xr_precip_velocity_db["velocity_x"].values
velocity_y = xr_precip_velocity_db["velocity_y"].values
precip_velocity_db = np.stack([velocity_x, velocity_y], axis=0)


# Create Dataset containing both variables and optical flow
dataset = xr_precip_velocity_db.copy(deep=True)

# Add VIL using their preprocessed dB values
dataset["vil"] = xr_vil_kg_db["vil"]

# Identify the precipitation and proxy variables
dataset.attrs["precip_var"] = "precip"
dataset.attrs["proxy_var"] = "vil"

# Fix time format
time = [
    datetime.strptime(str(t), "%Y%m%d%H%M%S")
    for t in dataset["time"].values
]
# 2. Convert time coordinate
dataset = dataset.assign_coords(
    time=("time", np.asarray(time, dtype="datetime64[ns]"))
)
# 4. Set time interval
dataset["time"].attrs["stepsize"] = 300.0


#Nowcast params
timesteps = 6
n_ens_members = 2
n_cascade_levels = 6
ar_window_radius = 20
var1_name = "vil"
var2_name = "precip"
noise_method = "nonparametric"
noise_method_var2 = "nonparametric"
vel_pert_method = None
adaptive_localization = int(n_cascade_levels/2) #half adloc
kmperpixel = 1
timestep = 5
seed = 24
num_workers = 4

nwc = nowcasts.get_method("stan")
fct_db = nwc(
    dataset = dataset,
    # var1 = vil_kg_db,
    # var2 = precip_db,
    # velocity = precip_velocity_db,
    timesteps = timesteps,
    n_ens_members = n_ens_members,
    n_cascade_levels = n_cascade_levels,
    ar_window_radius = ar_window_radius,
    var1_thr = vil_thr_db_kg,
    var2_thr = precip_thr_db,
    var1_name = var1_name,
    var2_name = var2_name,
    noise_method = noise_method,
    noise_method_var2 = noise_method_var2,
    vel_pert_method = vel_pert_method,
    adaptive_localization = adaptive_localization,
    kmperpixel = kmperpixel,
    timestep = timestep,
    seed = seed,
    num_workers=num_workers,
)
fct_db["precip"].attrs["transform"] = "dB"
fct = transformation.dB_transform(fct_db, threshold=precip_thr_db, inverse=True)["precip"]


