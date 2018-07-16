#!/bin/env python

"""Tutorial 2: Stochastic ensemble precipitation nowcasting

The tutorial introduces stochastic perturbations in order to generate a stochastic 
ensemble of precipitation nowcasts. 

More info: https://pysteps.github.io/
"""
import ast
import configparser
import datetime
import matplotlib.pylab as plt
import numpy as np
import pickle
import os
import pysteps as st

# List of case studies that can be used in this tutorial

#+-------+--------------+-------------+----------------------------------------+
#| event |  start_time  | data_source | description                            |
#+=======+==============+=============+========================================+
#|  01   | 201701311030 |     mch     | orographic precipitation               |
#+-------+--------------+-------------+----------------------------------------+
#|  02   | 201505151630 |     mch     | non-stationary field, apparent rotation|
#+-------+--------------+------------------------------------------------------+
#|  03   | 201609281530 |     fmi     | stratiform rain band                   |
#+-------+--------------+-------------+----------------------------------------+
#|  04   | 201705091130 |     fmi     | widespread convective activity         |
#+-------+--------------+-------------+----------------------------------------+

# Set parameters for this tutorial

## input data (copy/paste values from table above)
startdate_str = "201701311030"
data_source   = "mch"

## methods
oflow_method        = "lucaskanade" # lucaskanade or DARTS 
nwc_method          = "STEPS"
adv_method          = "semilagrangian"
noise_method        = "parametric" # nonparametric or nested
bandpass_filter     = "gaussian" 
decomp_method       = "fft"

## forecast parameters
n_prvs_times        = 9 
n_lead_times        = 12
n_ens_members       = 2
n_cascade_levels    = 6
ar_order            = 2
R_threshold         = 0.1 # [mm/h]
prob_matching       = True

## visualization parameters
colorscale      = "MeteoSwiss" # MeteoSwiss or STEPS-BE
motion_plot     = None # None, streamplot, quiver

# Read-in the data
print('Read the data...')

# Read the tutorial configuration file
config = configparser.RawConfigParser()
config.read("tutorials.cfg")

path_outputs = config["paths"]["output"]

# Read the data source configuration file
config = configparser.RawConfigParser()
config.read("datasource_%s.cfg" % data_source)

config_ds = config["datasource"]

root_path       = config_ds["root_path"]
path_fmt        = config_ds["path_fmt"]
fn_pattern      = config_ds["fn_pattern"]
fn_ext          = config_ds["fn_ext"]
importer        = config_ds["importer"]
timestep        = float(config_ds["timestep"])

# Read the keyword arguments into importer_kwargs
importer_kwargs = {}
for v in config["importer_kwargs"].items():
    importer_kwargs[str(v[0])] = ast.literal_eval(v[1])
    
startdate  = datetime.datetime.strptime(startdate_str, "%Y%m%d%H%M")

## find radar field filenames
input_files = st.io.find_by_date(startdate, root_path, path_fmt, fn_pattern, 
                                 fn_ext, timestep, n_prvs_times, 0)

importer = st.io.get_method(importer)

## read radar field files
R, _, metadata = st.io.read_timeseries(input_files, importer, **importer_kwargs)

# Prepare input files
print("Prepare the data...")

## make sure we work with a square domain
print(R.shape)
orig_field_dim = R.shape[1:]
R = st.utils.square_domain(R, "pad")
print(R.shape)

## convert units
data_units = metadata["unit"]
if data_units is "dBZ":
    R = st.utils.dBZ2mmhr(R, R_threshold)

## convert linear rainrates to logarithimc dBR units
dBR, dBRmin = st.utils.mmhr2dBR(R, R_threshold)
dBR[~np.isfinite(dBR)] = dBRmin

# Compute motion field
print("Compute the motion vectors...")

oflow_method = st.optflow.get_method(oflow_method)
UV = oflow_method(dBR) 

# Perform the nowcast
print("Compute the ensemble nowcast...")

# Generate the ensemble nowcast
nwc_method = st.nowcasts.get_method(nwc_method)
dBR_forecast = nwc_method(dBR, UV, n_lead_times, n_ens_members, 
                    n_cascade_levels, R_threshold, adv_method, decomp_method, 
                    bandpass_filter, noise_method, 
                    metadata["xpixelsize"]/1000, timestep, ar_order, 
                    None, None, False, False, prob_matching)    

## convert the forecasted dBR to mm/h
R_forecast = st.utils.dBR2mmhr(dBR_forecast, R_threshold)

## readjust to initial domain shape
R          = st.utils.unsquare_domain(R,          orig_field_dim)
R_forecast = st.utils.unsquare_domain(R_forecast, orig_field_dim)

print(R_forecast.shape)    

## plot the nowcast...
st.plt.animate(R, nloops=2, timestamps=metadata["timestamps"],
               R_for=R_forecast, timestep_min=timestep,
               UV=UV, motion_plot=motion_plot,
               geodata=metadata, colorscale=colorscale,
               plotanimation=True, savefig=False, path_outputs=path_outputs) 

# Forecast verification
print("Forecast verification...")

## find the verifying observations
input_files_verif = st.io.find_by_date(startdate, root_path, path_fmt, fn_pattern, 
                                        fn_ext, timestep, 0, n_lead_times)
if all(fpath is None for fpath in input_files_verif[0]):
    raise ValueError("Verification data not found")

## read observations
Robs, _, _ = st.io.read_timeseries(input_files_verif, importer, **importer_kwargs)
Robs = Robs[1:,:,:]

## convert units
if data_units is "dBZ":
    Robs = st.utils.dBZ2mmhr(Robs, R_threshold)

## compute the average continuous ranked probability score (CRPS)
scores = np.zeros(n_lead_times)*np.nan
for i in range(n_lead_times):
    scores[i] = st.vf.CRPS(R_forecast[:,i,:,:].reshape((n_ens_members, -1)).transpose(), 
                           Robs[i,:,:].flatten())

## if already exists, load the figure object to append the new verification results
filename = "%s/%s" % (path_outputs, "tutorial2_fig_verif")
if os.path.exists("%s.dat" % filename):
    ax = pickle.load(open("%s.dat" % filename, "rb"))
    print("Figure object loaded: %s.dat" % filename) 
else:
    fig, ax = plt.subplots()
    
## plot the scores
nplots = len(ax.lines)
x = (np.arange(n_lead_times) + 1)*timestep
ax.plot(x, scores, color="C%i"%(nplots + 1), label = "run %02d" % (nplots + 1))
ax.set_xlabel("Lead-time [min]")
ax.set_ylabel("CRPS")
plt.legend()

## dump the figure object
pickle.dump(plt.gca(), open("%s.dat" % filename, "wb"))
print("Figure object saved: %s.dat" % filename)
# remove the pickle object to plot a new figure

plt.show()
