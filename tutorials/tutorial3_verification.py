#!/bin/env python

"""Tutorial 3: verification of an ensemble nowcast

"""
import ast
import configparser
import datetime
import matplotlib.pylab as plt
import netCDF4
import numpy as np
import os
import pprint
import sys
import time

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

# Verification parameters
overwrite = False
vthreshold = 1.0
vleadtimes = [10, 30, 60, 120] # min
vaccu = 5 # # TODO: add option to verify accumulations

# Ranges of parameters for the verification
## don't change the order of the parameters
verif_pars = {
    "data"              : [("201505151630", "mch")],
    "R_threshold"       : [0.1],
    "n_lead_times"      : [24],
    "oflow_method"      : ["lucaskanade"],
    "n_prvs_times"      : [None],
    "nwc_method"        : ["STEPS"],
    "n_ens_members"     : [20],
    "noise_method"      : ["nonparametric"],
    "ar_order"          : [2],
    "n_cascade_levels"  : [8],
    "decomp_method"     : ["fft"],
    "bandpass_filter"   : [None],
    "adv_method"        : ["semilagrangian"],
    "conditional"       : [True,False],
    "precip_mask"       : [True,False],
    "prob_matching"     : [True,False]
}

# Conditional parameters
def cond_pars(pars):
    for key in list(pars):
        if key == "oflow_method":
            if pars[key].lower() == "darts":  pars["n_prvs_times"] = 9
            else:                     pars["n_prvs_times"] = 3
        elif key.lower() == "n_cascade_levels":
            if pars[key] == 1 : pars["bandpass_filter"] = "uniform"
            else:               pars["bandpass_filter"] = "gaussian"
        elif key.lower() == "nwc_method":
            if pars[key] == "extrapolation" : pars["n_ens_members"] = 1
    return pars
    
# Read the tutorial configuration file
config = configparser.RawConfigParser()
config.read("tutorials.cfg")
path_outputs = config["paths"]["output"]

# Prepare the list of all parameter sets of the verification
parsets = [[]]
for _, items in verif_pars.items():
    parsets = [parset+[item] for parset in parsets for item in items]

# Now loop all parameter sets
for n, parset in enumerate(parsets):
    p = {}
    for m, key in enumerate(verif_pars.keys()):
        p[key] = parset[m]
    ## apply conditional parameters
    p = cond_pars(p)
    
    print("************************")
    print("* Parameter set %02d/%02d: *" % (n+1, len(parsets)))
    print("************************")
    
    ## if necessary, build path to results
    path_to_nwc = path_outputs
    for key, item in p.items():
        if key.lower() == "data":
            path_to_nwc = os.path.join(path_to_nwc, item[0])
            path_to_nwc = os.path.join(path_to_nwc, item[1])
        else:
            path_to_nwc = os.path.join(path_to_nwc, '-'.join([key, str(item)]))
    try:
        os.makedirs(path_to_nwc)
    except FileExistsError:
        pass
        
    # **************************************************************************
    # **************************************************************************   
    # Nowcasting

    ## check if results already exists
    run_exist = False
    outfn = os.path.join(path_to_nwc, "nowcast.netcdf")
    if os.path.isfile(outfn):
        ds = netCDF4.Dataset(outfn, 'r')
        if ds.dimensions["time"].size == p["n_lead_times"]:
            run_exist = True
        else:
            os.remove(outfn)
            
    if run_exist and not overwrite:
        print("Nowcast already exists.")

    else:
        
        print("Computing the nowcast...")
        
        print("Log: %s" % os.path.join(path_to_nwc, "log.txt"))
        
        ## redirect stdout to log file
        orig_stdout = sys.stdout
        f = open(os.path.join(path_to_nwc, "log.txt"), 'w')
        sys.stdout = f
        
        print("--- Start of the run : %s ---" % (datetime.datetime.now()))
        
        ## time
        t0 = time.time()
        
        print("*******************")
        print("* Parameter set : *")
        print("*******************")
        pprint.pprint(p)

        # Read the data source configuration file
        config = configparser.RawConfigParser()
        config.read("datasource_%s.cfg" % p["data"][1])

        config_ds = config["datasource"]

        root_path       = config_ds["root_path"]
        path_fmt        = config_ds["path_fmt"]
        fn_pattern      = config_ds["fn_pattern"]
        fn_ext          = config_ds["fn_ext"]
        importer        = config_ds["importer"]
        timestep        = float(config_ds["timestep"])

        ## read the keyword arguments into importer_kwargs
        importer_kwargs = {}
        for v in config["importer_kwargs"].items():
            importer_kwargs[str(v[0])] = ast.literal_eval(v[1])

        startdate  = datetime.datetime.strptime(p["data"][0], "%Y%m%d%H%M")
        
        # Read inputs
        
        ## find radar field filenames
        input_files = st.io.find_by_date(startdate, root_path, path_fmt, fn_pattern,
                                        fn_ext, timestep, p["n_prvs_times"], 0)
        importer = st.io.get_method(importer)

        ## read radar field files
        R, _, metadata = st.io.read_timeseries(input_files, importer, **importer_kwargs)

        ## make sure we work with a square domain
        orig_field_dim = R.shape[1:]
        R = st.utils.square_domain(R, "pad")

        ## convert units
        if metadata["unit"] is "dBZ":
            R = st.utils.dBZ2mmhr(R, p["R_threshold"])
            metadata["unit"] = "mm/h"
            
        ## convert linear rain rates to logarithimc dBR units
        R, dBRmin = st.utils.mmhr2dBR(R, p["R_threshold"])
        R[~np.isfinite(R)] = dBRmin
        
        # Compute motion field
        oflow_method = st.optflow.get_method(p["oflow_method"])
        UV = oflow_method(R)
        
        # Perform the nowcast       

        ## define the callback function to export the nowcast to netcdf
        def export(X):
            # convert the forecasted dBR to mm/h
            X = st.utils.dBR2mmhr(X, p["R_threshold"])
            # readjust to initial domain shape
            X = st.utils.unsquare_domain(X, orig_field_dim)
            # export to netcdf
            st.io.export_forecast_dataset(X, exporter)
        
        ## initialize netcdf file
        incremental = "timestep" if p["nwc_method"].lower() == "steps" else None
        exporter = st.io.initialize_forecast_exporter_netcdf(outfn, startdate,
        timestep, p["n_lead_times"], (orig_field_dim), p["n_ens_members"], metadata,
        incremental=incremental)
        
        ## start the nowcast
        nwc_method = st.nowcasts.get_method(p["nwc_method"])
        R_fct = nwc_method(R, UV, p["n_lead_times"], p["n_ens_members"],
                    p["n_cascade_levels"], p["R_threshold"], p["adv_method"],
                    p["decomp_method"], p["bandpass_filter"], p["noise_method"],
                    metadata["xpixelsize"]/1000, timestep, p["ar_order"],
                    conditional=p["conditional"], use_precip_mask=p["precip_mask"], 
                    use_probmatching=p["prob_matching"], callback=export, return_output=False)

        # save results
        st.io.close_forecast_file(exporter)
        R_fct = None
            
        # save log
        print("--- End of the run : %s ---" % (datetime.datetime.now()))
        print("--- Total time : %s seconds ---" % (time.time() - t0))
        sys.stdout = orig_stdout
        f.close()
    
    # **************************************************************************
    # **************************************************************************   
    # Verification
        
    if p["nwc_method"].lower() == "steps":
        rankhists = {}
        reldiags = {}
        rocs = {}
        for lt in vleadtimes:
            rankhists[lt] = st.verification.ensscores.rankhist_init(p["n_ens_members"], vthreshold)
            reldiags[lt]  = st.verification.probscores.reldiag_init(vthreshold)
            rocs[lt]      = st.verification.probscores.ROC_curve_init(vthreshold)
    
    
    # Read the data source configuration file
    config = configparser.RawConfigParser()
    config.read("datasource_%s.cfg" % p["data"][1])

    config_ds = config["datasource"]

    root_path       = config_ds["root_path"]
    path_fmt        = config_ds["path_fmt"]
    fn_pattern      = config_ds["fn_pattern"]
    fn_ext          = config_ds["fn_ext"]
    importer        = config_ds["importer"]
    timestep        = float(config_ds["timestep"])

    ## read the keyword arguments into importer_kwargs
    importer_kwargs = {}
    for v in config["importer_kwargs"].items():
        importer_kwargs[str(v[0])] = ast.literal_eval(v[1])
        
    importer = st.io.get_method(importer) 
        
    # Load the nowcast
    R_fct, metadata = st.io.import_netcdf_pysteps(outfn)
    timestamps = metadata["timestamps"]
    leadtimes = np.arange(1,len(timestamps)+1)*timestep # min
    
    # Loop leadtimes and do verification
    for i,lt in enumerate(vleadtimes):
        
        idlt = leadtimes == lt
        fct_date = timestamps[idlt][0]
        obs_fn = st.io.archive.find_by_date(fct_date, root_path, path_fmt, 
                                            fn_pattern, fn_ext, timestep)                                       
        R_obs, _, metadata = st.io.read_timeseries(obs_fn, importer, **importer_kwargs) 

        fig = plt.figure()
        im = plt.imshow(R_obs[0,:,:])
        fig.colorbar(im)
        plt.savefig(os.path.join(path_to_nwc, "R_obs_%03d.png" % (lt)))
        plt.close()        
        
        ## convert units
        if metadata["unit"].lower() is "dbz":
            R_obs = st.utils.dBZ2mmhr(R_obs, p["R_threshold"])
            metadata["unit"] = "mm/h"
        
        R_fct_ = np.vstack([R_fct[j, idlt, :, :].flatten() for j in range(p["n_ens_members"])]).T
        st.verification.ensscores.rankhist_accum(rankhists[lt], 
            R_fct_, R_obs.flatten())
        P_fct = 1.0*np.sum(R_fct_ >= vthreshold, axis=1) / p["n_ens_members"]
        st.verification.probscores.reldiag_accum(reldiags[lt], P_fct, R_obs.flatten())
        st.verification.probscores.ROC_curve_accum(rocs[lt], P_fct, R_obs.flatten())
            
    for i,lt in enumerate(vleadtimes):
    
        idlt = leadtimes == lt
        fig = plt.figure()
        im = plt.imshow(R_fct[0, idlt, :, :].squeeze())
        fig.colorbar(im)
        plt.savefig(os.path.join(path_to_nwc, "R_fct_%s_%03d.png" % (p["nwc_method"], lt)))
        plt.close()
        
        fig = plt.figure()
        st.verification.plot_rankhist(rankhists[lt], ax=fig.gca())
        plt.savefig(os.path.join(path_to_nwc, "rankhist_%03d_thr%.1f.png" % (lt, vthreshold)), 
                bbox_inches="tight")
        plt.close()
        
        fig = plt.figure()
        st.verification.plot_reldiag(reldiags[lt], ax=fig.gca())
        plt.savefig(os.path.join(path_to_nwc, "reldiag_%03d_thr%.1f.png" % (lt, vthreshold)), 
                bbox_inches="tight")
        plt.close()
        
        fig = plt.figure()
        st.verification.plot_ROC(rocs[lt], ax=fig.gca())
        plt.savefig(os.path.join(path_to_nwc, "roc_%03d_thr%.1f.png" % (lt, vthreshold)), 
                bbox_inches="tight")
        plt.close()        