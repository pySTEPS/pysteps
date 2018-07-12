#!/bin/env python

"""Tutorial 1: Motion field estimation and extrapolation forecast

The tutorial guides you into the basic notions and techniques for extrapolation 
nowcasting. 

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
oflow_method    = "lucaskanade" # lucaskanade or DARTS
adv_method      = "semilagrangian"

## forecast parameters
n_lead_times    = 24
R_threshold     = 0.1 # [mm/h]

## visualization parameters
colorscale      = "MeteoSwiss" # MeteoSwiss or STEPS-BE
motion_plot     = "quiver" # streamplot or quiver

## verification parameters
skill_score     = "CSI"
verif_thr       = 1 # [mm/h]
 
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
                                 fn_ext, timestep, 9)
if all(fpath is None for fpath in input_files[0]):
    raise IOError("input data not found in %s" % root_path)

importer = st.io.get_method(importer)

## read radar field files
R, _, metadata = st.io.read_timeseries(input_files, importer, **importer_kwargs)
print("The data array has size [nleadtimes,nrows,ncols] =", R.shape)
orig_field_dim = R.shape

# Prepare input files
print("Prepare the data...")

# TODO: This is currently hard-coded.
data_units = metadata["unit"]

## convert units
if data_units is "dBZ":
    R = st.utils.dBZ2mmhr(R, R_threshold)

## make sure we work with a square domain
R = st.utils.square_domain(R, "crop")

## convert linear rainrates to logarithimc dBR units
dBR, dBRmin = st.utils.mmhr2dBR(R, R_threshold)
dBR[~np.isfinite(dBR)] = dBRmin

## visualize the input radar fields
doanimation = True
nloops = 2 # how many times to loop
# TODO: wrap code for animations into a nicely written function

loop = 0
while loop < nloops:
    for i in range(R.shape[0]):
        plt.clf()
        if doanimation:
            st.plt.plot_field(R[i,:,:], None, units="mmhr", 
                          colorscale=colorscale, 
                          title=input_files[1][i].strftime("%Y-%m-%d %H:%M"), 
                          colorbar=True)
            plt.pause(.2)
    if doanimation:
        plt.pause(.5)
    loop += 1

if doanimation == True:
    plt.close()
    
# Compute motion field
print("Computing motion vectors...")

oflow_method = st.optflow.get_method(oflow_method)
UV = oflow_method(dBR) 

## plot the motion field
doanimation = True
nloops = 2
# TODO: wrap code for animations into a nicely written function

loop = 0
while loop < nloops:
    
    for i in range(R.shape[0]):
        plt.clf()
        if doanimation:
            st.plt.plot_field(R[i,:,:], None, units="mmhr", 
                          colorscale=colorscale, 
                          title="Motion field", colorbar=True)
            if motion_plot == "quiver":
                st.plt.quiver(UV, None, 20)
            if motion_plot == "streamplot":    
                st.plt.streamplot(UV, None)        
            plt.pause(.2)
        
    if doanimation:
        plt.pause(.5)
    loop += 1

if doanimation == True:
    plt.close()

# Perform the advection of the radar field
print('Computing extrapolation...')

adv_method = st.advection.get_method(adv_method) 
dBR_forecast = adv_method(dBR[-1,:,:], UV, n_lead_times) 

## convert the forecasted dBR to mmhr
R_forecast = st.utils.dBR2mmhr(dBR_forecast, R_threshold)
print("The forecast array has size [nleadtimes,nrows,ncols] =", R_forecast.shape)

## plot the nowcast...
doanimation     = True
savefig         = False
nloops = 2
# TODO: wrap code for animations into a nicely written function

loop = 0
while loop < nloops:
    
    for i in range(R.shape[0] + n_lead_times):
        plt.clf()
        if doanimation:
            if i < R.shape[0]:
                # Plot last observed rainfields
                st.plt.plot_field(R[i,:,:], None, units="mmhr",
                              colorscale=colorscale, 
                              title=input_files[1][i].strftime("%Y-%m-%d %H:%M"), 
                              colorbar=True)
                if savefig & (loop == 0):
                    figname = "%s/%s_%s_simple_advection_%02d_obs.png" % (path_outputs, startdate_str, data_source, i)
                    plt.savefig(figname)
                    print(figname, 'saved.')
            else:
                # Plot nowcast
                st.plt.plot_field(R_forecast[i - R.shape[0],:,:], 
                              None, units="mmhr", 
                              title="%s +%02d min" % 
                              (input_files[1][-1].strftime("%Y-%m-%d %H:%M"),
                              (1 + i - R.shape[0])*timestep),
                              colorscale=colorscale, colorbar=True)
                if savefig & (loop == 0):
                    figname = "%s/%s_%s_simple_advection_%02d_nwc.png" % (path_outputs, startdate_str, data_source, i)
                    plt.savefig(figname)
                    print(figname, "saved.")
            plt.pause(.2)
    if doanimation:
        plt.pause(.5)
    loop += 1

if doanimation == True:
    plt.close()

# Forecast verification
print('Forecast verification...')

## find the verifying observations
input_files_verif = st.io.find_by_date(
                                   startdate + datetime.timedelta(minutes=n_lead_times*timestep), 
                                   root_path, path_fmt, fn_pattern, fn_ext, 
                                   timestep, n_lead_times - 1)
if all(fpath is None for fpath in input_files_verif[0]):
    raise ValueError("Verification data not found")

## read observations
Robs, _, _ = st.io.read_timeseries(input_files_verif, importer, **importer_kwargs)

## convert units
if data_units is 'dBZ':
    Robs = st.utils.dBZ2mmhr(Robs, R_threshold)

## and square domain
Robs_ = st.utils.square_domain(Robs, "crop")

## compute verification scores
scores = np.zeros(n_lead_times)*np.nan
for i in range(n_lead_times):
    scores[i] = st.vf.scores_det_cat_fcst(R_forecast[i,:,:], Robs_[i,:,:], 
                                           verif_thr, [skill_score])[0]

## if already exists, load the figure object to append the new verification results
filename = "%s/%s" % (path_outputs, "tutorial1_fig_verif")
if os.path.exists("%s.dat" % filename):
    ax = pickle.load(open("%s.dat" % filename, "rb"))
    print("Figure object loaded: %s.dat" % filename) 
else:
    fig, ax = plt.subplots()
    
## plot the scores
nplots = len(ax.lines)
x = (np.arange(n_lead_times) + 1)*timestep
ax.plot(x, scores, color='C%i'%(nplots + 1), label = "run %02d" % (nplots + 1))
ax.set_xlabel("Lead-time [min]")
ax.set_ylabel("%s" % skill_score)
plt.legend()

## dump the figure object
pickle.dump(plt.gca(), open("%s.dat" % filename, "wb"))
print("Figure object saved: %s.dat" % filename)
# remove the pickle object to plot a new figure

plt.show()
