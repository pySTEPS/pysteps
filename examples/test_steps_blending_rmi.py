# -*- coding: utf-8 -*-

from matplotlib import cm, pyplot as plt
import numpy as np
import xarray as xr
import os
from datetime import datetime
from pprint import pprint
from pysteps import io, rcparams
from pysteps.utils import conversion, transformation, reprojection
from pysteps.visualization import plot_precip_field

# Set workdir and number of output time steps
work_dir = "c:\\Users\\imhof_rn\\pysteps"
os.chdir(work_dir)
n_timesteps = 36

# Give date as input
date_str = "202107141010"  # input("Give date and time (e.g.: 201609281600):    ")

date = datetime.strptime(date_str, "%Y%m%d%H%M")
data_source = rcparams.data_sources["rmi"]

# Import the radar composite to reproject NWP
root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
timestep = data_source["timestep"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]

# Find the radar files in the archive
fns = io.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2
)

importer = io.get_method(importer_name, "importer")
R_input = io.read_timeseries(fns, importer, legacy=False, **importer_kwargs)

radar_data_xr = R_input[-1, :, :]
radar_metadata = radar_data_xr.x.attrs.copy()
radar_metadata.update(**radar_data_xr.y.attrs)
radar_metadata.update(**radar_data_xr.attrs)

# PLOTTING

# R, _, metadata = io.import_odim_hdf5(filename, "RATE", legacy=True, **importer_kwargs)
# Convert to rain rate
# R, metadata = conversion.to_rainrate(R, metadata)
# Nicely print the metadata
# pprint(metadata)
# Plot the rainfall field
# plot_precip_field(R, geodata=metadata)
# plt.show()


# Import the NWP data
date_str = "202107140600"  # input("Give date and time (e.g.: 201609281600):    ")

date = datetime.strptime(date_str, "%Y%m%d%H%M")
nwp_data_source = rcparams.data_sources["rmi_nwp"]
filename = os.path.join(
    nwp_data_source["root_path"],
    datetime.strftime(date, nwp_data_source["path_fmt"]),
    datetime.strftime(date, nwp_data_source["fn_pattern"])
    + "."
    + nwp_data_source["fn_ext"],
)

nwp_data_xr = io.import_rmi_nwp_xr(filename)
nwp_metadata = nwp_data_xr.x.attrs.copy()
nwp_metadata.update(**nwp_data_xr.y.attrs)
nwp_metadata.update(**nwp_data_xr.attrs)

# plot_precip_field(nwp_data_xr.values[1,:,:], geodata=nwp_metadata)
# plt.show()

# Reproject the projection of the nwp data to the projection of the radar data
R_NWP = reprojection(nwp_data_xr[0 : n_timesteps + 1, :, :], radar_data_xr)
nwp_rep_metadata = R_NWP.x.attrs.copy()
nwp_rep_metadata.update(**R_NWP.y.attrs)
nwp_rep_metadata.update(**R_NWP.attrs)

# plot_precip_field(R_NWP[0,:,:], geodata=nwp_rep_metadata)
# plt.show()

########## Now start the blending! ################

import pysteps
from pysteps import cascade, blending

R_input.data[R_input.data < radar_metadata["threshold"]] = 0.0
R_NWP.data[R_NWP.data < nwp_rep_metadata["threshold"]] = 0.0

converter = pysteps.utils.get_method("mm/h")
R_input, radar_metadata = converter(R_input, radar_metadata)
converter = pysteps.utils.get_method("mm/h")
R_NWP, nwp_rep_metadata = converter(R_NWP, nwp_rep_metadata)

# transform the data
transformer = pysteps.utils.get_method("dB")
R_input, radar_metadata = transformer(R_input.values, radar_metadata, threshold=0.1)
transformer = pysteps.utils.get_method("dB")
R_NWP, nwp_rep_metadata = transformer(R_NWP.values, nwp_rep_metadata, threshold=0.1)

# # set NaN equal to zero
# R_input[~np.isfinite(R_input)] = radar_metadata["zerovalue"]
# R_NWP[~np.isfinite(R_NWP)] = nwp_rep_metadata["zerovalue"]

# Initial decomposition settings
decomp_method, recomp_method = cascade.get_method("fft")
bandpass_filter_method = "gaussian"
M, N = R_input.shape[1:]
n_cascade_levels = 8
n_ens_members = 1
n_models = 1
mask_method = "incremental"
probmatching_method = "cdf"
expected_n_ens_members = 4
filter_method = cascade.get_method(bandpass_filter_method)
filter = filter_method((M, N), n_cascade_levels)

# If we only use one model:
if R_NWP.ndim == 3:
    R_NWP = R_NWP[None, :]

R_d_models = []
# Loop through the n_models
for i in range(R_NWP.shape[0]):
    R_d_models_ = []
    # Loop through the time steps
    for j in range(R_NWP.shape[1]):
        R_ = decomp_method(
            field=R_NWP[i, j, :, :],
            bp_filter=filter,
            normalize=True,
            compute_stats=True,
            compact_output=True,
        )
        R_d_models_.append(R_)
    R_d_models.append(R_d_models_)

R_d_models = np.array(R_d_models)

assert R_d_models.ndim == 2, "Wrong number of dimensions in R_d_models"

###
# Determine the velocity fields
###
oflow_method = pysteps.motion.get_method("lucaskanade")
V_radar = oflow_method(R_input)
V_NWP = []
# Loop through the models
for n_model in range(R_NWP.shape[0]):
    # Loop through the timesteps. We need two images to construct a motion
    # field, so we can start from timestep 1. Timestep 0 will be the same
    # as timestep 0.
    _V_NWP_ = []
    for t in range(1, R_NWP.shape[1]):
        V_NWP_ = oflow_method(R_NWP[n_model, t - 1 : t + 1, :])
        _V_NWP_.append(V_NWP_)
        V_NWP_ = None
    _V_NWP_ = np.insert(_V_NWP_, 0, _V_NWP_[0], axis=0)
    V_NWP.append(_V_NWP_)
V_NWP = np.stack(V_NWP)

assert V_NWP.ndim == 5, "V_NWP must be a five-dimensional array"

###
# The nowcasting
###
print("Start the nowcast")
precip_forecast = blending.steps.forecast(
    R=R_input,
    R_d_models=R_d_models,
    V=V_radar,
    V_models=V_NWP,
    timesteps=n_timesteps,
    timestep=5.0,
    n_ens_members=n_ens_members,
    n_cascade_levels=n_cascade_levels,
    blend_nwp_members=False,
    R_thr=radar_metadata["threshold"],
    kmperpixel=1.0,
    extrap_method="semilagrangian",
    decomp_method="fft",
    bandpass_filter_method="gaussian",
    noise_method="nonparametric",
    noise_stddev_adj="auto",
    ar_order=2,
    vel_pert_method=None,
    conditional=False,
    probmatching_method=probmatching_method,
    mask_method=mask_method,
    callback=None,
    return_output=True,
    seed=None,
    num_workers=2,
    fft_method="numpy",
    domain="spatial",
    extrap_kwargs=None,
    filter_kwargs=None,
    noise_kwargs=None,
    vel_pert_kwargs=None,
    clim_kwargs=None,
    mask_kwargs=None,
    measure_time=False,
)

# assert precip_forecast.ndim == 4, "Wrong amount of dimensions in forecast output"
# assert (
#     precip_forecast.shape[0] == expected_n_ens_members
# ), "Wrong amount of output ensemble members in forecast output"
# assert (
#     precip_forecast.shape[1] == n_timesteps
# ), "Wrong amount of output time steps in forecast output"

# Transform the data back into mm/h
precip_forecast, _ = converter(precip_forecast, radar_metadata)
R_input, _ = converter(R_input, radar_metadata)
R_NWP, _ = converter(R_NWP, nwp_rep_metadata)


# assert (
#     precip_forecast.ndim == 4
# ), "Wrong amount of dimensions in converted forecast output"
# assert (
#     precip_forecast.shape[0] == expected_n_ens_members
# ), "Wrong amount of output ensemble members in converted forecast output"
# assert (
#     precip_forecast.shape[1] == n_timesteps
# ), "Wrong amount of output time steps in converted forecast output"


# from pysteps.visualization.animations import animate

# Plot the blended forecast
for t in range(0, precip_forecast.shape[1]):
    plot_precip_field(
        precip_forecast[0, t, :, :],
        geodata=radar_metadata,
        title=f"Blended forecast for t + {str(t*5+5)}",
    )
    plt.savefig(
        f"c:\\Users\\imhof_rn\\OneDrive - Stichting Deltares\\Documents\\PhD\\NWP_Blending\\Model_runs\\Dummy_Figs\\RMI_blending_test\\Blended_forecast_{str(t)}.png",
        bbox_inches="tight",
    )
    plt.close()

# Plot the radar data from t-2 to t = 0
for t in range(0, R_input.shape[0]):
    plot_precip_field(
        R_input[t, :, :],
        geodata=radar_metadata,
        title=f"Radar data for t = {str(t*5-10)}",
    )
    plt.savefig(
        f"c:\\Users\\imhof_rn\\OneDrive - Stichting Deltares\\Documents\\PhD\\NWP_Blending\\Model_runs\\Dummy_Figs\\RMI_blending_test\\Radar_{str(t)}.png",
        bbox_inches="tight",
    )
    plt.close()

# Plot the NWP forecast
for t in range(0, R_NWP.shape[1]):
    plot_precip_field(
        R_NWP[0, t, :, :],
        geodata=nwp_rep_metadata,
        title=f"NWP forecast for t + {str(t*5+5)}",
    )
    plt.savefig(
        f"c:\\Users\\imhof_rn\\OneDrive - Stichting Deltares\\Documents\\PhD\\NWP_Blending\\Model_runs\\Dummy_Figs\\RMI_blending_test\\NWP_forecast_{str(t)}.png",
        bbox_inches="tight",
    )
    plt.close()


# plot_precip_field(precip_forecast[0, 5, :, :], geodata=nwp_rep_metadata)
# plt.show()
# animate(
#     R_input,
#     precip_forecast,
#     display_animation=True,
#     savefig=True,
#     path_outputs="c:\\Users\\imhof_rn\\OneDrive - Stichting Deltares\\Documents\\PhD\\NWP_Blending\\Model_runs\\Dummy_Figs\\RMI_blending_test",
#     fig_dpi=20,
# )
