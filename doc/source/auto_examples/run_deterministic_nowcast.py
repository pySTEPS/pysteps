#!/bin/env python

"""Motion field estimation and extrapolation forecast

The script shows how to run a deterministic radar-based extrapolation nowcast with
pysteps.

More info: https://pysteps.github.io/
"""
import datetime
import matplotlib.pylab as plt
import numpy as np
import pickle
import os

import pysteps as stp

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
oflow_method    = "lucaskanade"      # lucaskanade, darts, None
adv_method      = "semilagrangian"   # semilagrangian, eulerian

## forecast parameters
n_prvs_times    = 3                  # use at least 9 with DARTS
n_lead_times    = 24
unit            = "mm/h"             # mm/h or dBZ
transformation  = "dB"               # None or dB
r_threshold     = 0.1                # rain/no-rain threshold [mm/h]

## verification parameters
skill_score     = "CSI"
v_threshold     = 1 # [mm/h]

# Read-in the data
print('Read the data...')
startdate  = datetime.datetime.strptime(startdate_str, "%Y%m%d%H%M")

## import data specifications
ds = stp.rcparams.data_sources[data_source]

## find radar field filenames
input_files = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern,
                                 ds.fn_ext, ds.timestep, n_prvs_times, 0)

## read radar field files
importer = stp.io.get_method(ds.importer, "importer")
R, _, metadata = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
Rmask = np.isnan(R)
print("The data array has size [nleadtimes,nrows,ncols] =", R.shape)

# Prepare input files
print("Prepare the data...")

## if necessary, convert to rain rates [mm/h]
converter = stp.utils.get_method("mm/h")
R, metadata = converter(R, metadata)

## threshold the data
R[R<r_threshold] = 0.0
metadata["threshold"] = r_threshold

## convert the data
converter = stp.utils.get_method(unit)
R, metadata = converter(R, metadata)

## transform the data
transformer = stp.utils.get_method(transformation)
R, metadata = transformer(R, metadata)

## set NaN equal to zero
R[~np.isfinite(R)] = metadata["zerovalue"]

# Compute motion field
oflow_method = stp.motion.get_method(oflow_method)
UV = oflow_method(R)

# Perform the advection of the radar field
adv_method = stp.extrapolation.get_method(adv_method)
R_fct = adv_method(R[-1,:,:], UV, n_lead_times, verbose=True)
print("The forecast array has size [nleadtimes,nrows,ncols] =", R_fct.shape)

## if necessary, transform back all data
R_fct, _    = transformer(R_fct, metadata, inverse=True)
R, metadata = transformer(R, metadata, inverse=True)

## convert all data to mm/h
converter   = stp.utils.get_method("mm/h")
R_fct, _    = converter(R_fct, metadata)
R, metadata = converter(R, metadata)

## plot the nowcast...
R[Rmask] = np.nan # reapply radar mask
stp.plt.animate(R, nloops=2, timestamps=metadata["timestamps"],
                R_fct=R_fct, timestep_min=ds.timestep,
                UV=UV,
                motion_plot=stp.rcparams.plot.motion_plot,
                geodata=metadata,
                colorscale=stp.rcparams.plot.colorscale,
                plotanimation=True, savefig=False,
                path_outputs=stp.rcparams.outputs.path_outputs)

# Forecast verification
print("Forecast verification...")

## find the verifying observations
input_files_verif = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern,
                                        ds.fn_ext, ds.timestep, 0, n_lead_times)

## read observations
R_obs, _, metadata_obs = stp.io.read_timeseries(input_files_verif, importer, **ds.importer_kwargs)
R_obs = R_obs[1:,:,:]

## if necessary, convert to rain rates [mm/h]
R_obs, metadata_obs = converter(R_obs, metadata_obs)

## threshold the data
R_obs[R_obs<r_threshold] = 0.0
metadata_obs["threshold"] = r_threshold

## compute verification scores
scores = np.zeros(n_lead_times)*np.nan
for i in range(n_lead_times):
    scores[i] = stp.vf.det_cat_fct(R_fct[i,:,:], R_obs[i,:,:],
                                    v_threshold)[skill_score]

## if already exists, load the figure object to append the new verification results
filename = "%s/%s" % (stp.rcparams.outputs.path_outputs, "verif_deterministic_nwc_example")
if os.path.exists("%s.dat" % filename):
    ax = pickle.load(open("%s.dat" % filename, "rb"))
    print("Figure object loaded: %s.dat" % filename)
else:
    fig, ax = plt.subplots()

## plot the scores
nplots = len(ax.lines)
x = (np.arange(n_lead_times) + 1)*ds.timestep
ax.plot(x, scores, color='C%i'%(nplots + 1), label = "run %02d" % (nplots + 1))
ax.set_xlabel("Lead-time [min]")
ax.set_ylabel("%s" % skill_score)
plt.legend()

## dump the figure object
pickle.dump(plt.gca(), open("%s.dat" % filename, "wb"))
print("Figure object saved: %s.dat" % filename)
# remove the pickle object to plot a new figure

plt.show()
