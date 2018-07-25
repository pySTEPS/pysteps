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
#|  05   | 201806161100 |     bom     | bom example data                       |
#+-------+--------------+-------------+----------------------------------------+

# Set parameters for this tutorial

## input data (copy/paste values from table above)
startdate_str = "201701311030"
data_source   = "mch"

## methods
oflow_method        = "lucaskanade" # lucaskanade or darts
nwc_method          = "STEPS"
adv_method          = "semilagrangian"
noise_method        = "nonparametric" # paramtric or nonparametric
bandpass_filter     = "gaussian" 
decomp_method       = "fft"

## forecast parameters
n_prvs_times        = 3 # use at least 9 with DARTS
n_lead_times        = 12
n_ens_members       = 3
n_cascade_levels    = 6
ar_order            = 2
r_threshold         = 0.1 # [mm/h]
prob_matching       = True
precip_mask         = True
conditional         = True
unit                = "mm/h" # mm/h or dBZ
transformation      = "dB"   # None or dB 
adjust_domain       = "square_domain"

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
Rmask = np.isnan(R)

# Prepare input files
print("Prepare the data...")

## make sure we work with a square domain
reshaper = st.utils.get_method(adjust_domain)
R, metadata = reshaper(R, metadata, method="pad")

## if necessary, convert to rain rates [mm/h]    
converter = st.utils.get_method(unit)
R, metadata = converter(R, metadata)

## threshold the data
R[R<r_threshold] = 0.0
metadata["threshold"] = r_threshold

## transform the data
transformer = st.utils.get_method(transformation)
R, metadata = transformer(R, metadata)

## set NaN equal to zero
R[~np.isfinite(R)] = metadata["zerovalue"]

# Compute motion field
oflow_method = st.optflow.get_method(oflow_method)
UV = oflow_method(R) 

# Perform the nowcast
nwc_method = st.nowcasts.get_method(nwc_method)
R_fct = nwc_method(R, UV, n_lead_times, n_ens_members, 
                    n_cascade_levels, metadata["threshold"], adv_method, decomp_method, 
                    bandpass_filter, noise_method, 
                    metadata["xpixelsize"]/1000, timestep, ar_order=ar_order,
                    conditional=conditional, use_precip_mask=precip_mask, 
                    use_probmatching=prob_matching)

## trasnform back values to mm/h
R_fct, _    = transformer(R_fct, metadata, inverse=True)
R, metadata = transformer(R, metadata, inverse=True)

## readjust to initial domain shape
R_fct, _    = reshaper(R_fct, metadata, inverse=True)
R, metadata = reshaper(R, metadata, inverse=True)

## plot the nowcast..
R[Rmask] = np.nan # reapply radar mask
st.plt.animate(R, nloops=2, timestamps=metadata["timestamps"],
               R_for=R_fct, timestep_min=timestep,
               UV=UV, motion_plot=motion_plot,
               geodata=metadata, colorscale=colorscale,
               plotanimation=True, savefig=False, path_outputs=path_outputs) 

# Forecast verification
print("Forecast verification...")

## find the verifying observations
input_files_verif = st.io.find_by_date(startdate, root_path, path_fmt, fn_pattern, 
                                        fn_ext, timestep, 0, n_lead_times)

## read observations
R_obs, _, metadata_obs = st.io.read_timeseries(input_files_verif, importer, **importer_kwargs)
R_obs = R_obs[1:,:,:]
metadata_obs["timestamps"] = metadata_obs["timestamps"][1:]

## if necessary, convert to rain rates [mm/h]    
R_obs, metadata_obs = converter(R_obs, metadata_obs)

## threshold the data
R_obs[R_obs<r_threshold] = 0.0
metadata_obs["threshold"] = r_threshold

## compute the average continuous ranked probability score (CRPS)
scores = np.zeros(n_lead_times)*np.nan
for i in range(n_lead_times):
    scores[i] = st.vf.CRPS(R_fct[:,i,:,:].reshape((n_ens_members, -1)).transpose(), 
                           R_obs[i,:,:].flatten())

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
