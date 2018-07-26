#!/bin/env python

"""Verification of an ensemble nowcast

The script shows how to run verification experiments for ensemble precipitation
nowcasting with pysteps

More info: https://pysteps.github.io/
"""
import datetime
import matplotlib.pylab as plt
import netCDF4
import numpy as np
import os
import pprint
import sys
import time

import pysteps as stp
import config as cfg

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

# Verification settings
verification = {
    "experiment_name"   : "ex01_cascade_decomp",
    "overwrite"         : True,
    "v_threshold"       : 1.0,  # [mm/h]                 
    "v_leadtimes"       : [10, 30, 60], # [min]
    "v_accu"            : 5.0,
    "seed"              : 42   # for reproducibility
}

# Forecast settings
forecast = {
    "n_lead_times"      : 12,
    "r_threshold"       : 0.1,      # [mm/h]
    "unit"              : "mm/h",   # mm/h or dBZ
    "transformation"    : "dB",     # None or dB 
    "adjust_domain"     : "square"  # None or square
}

# The experiment set-up
## this includes tuneable parameters
experiment = {
    ## the events           event start     event end       update cycle  data source
    "data"              : [("201505151630", "201505151900", 30,           "mch"),
                           ("201701311030", "201701311300", 30,           "mch"),
                           ("201609281530", "201609281800", 30,           "fmi"),
                           ("201705091130", "201705091400", 30,           "fmi")],
    
    ## the methods
    "oflow_method"      : ["lucaskanade"],
    "adv_method"        : ["semilagrangian"],
    "nwc_method"        : ["steps"],
    "noise_method"      : ["nonparametric"],
    "decomp_method"     : ["fft"],
    
    ## the parameters
    "n_ens_members"     : [10],
    "ar_order"          : [2],
    "n_cascade_levels"  : [1,3,6,9],
    "conditional"       : [False],
    "precip_mask"       : [False],
    "mask_method"       : ["sprog"], # sprog, obs or incremental
    "prob_matching"     : [True],
}

# Conditional parameters
def cond_pars(pars):
    for key in list(pars):
        if key == "oflow_method":
            if pars[key].lower() == "darts":  pars["n_prvs_times"] = 9
            else:                             pars["n_prvs_times"] = 3
        elif key.lower() == "n_cascade_levels":
            if pars[key] == 1 : pars["bandpass_filter"] = "uniform"
            else:               pars["bandpass_filter"] = "gaussian"
        elif key.lower() == "nwc_method":
            if pars[key] == "extrapolation" : pars["n_ens_members"] = 1
    return pars
    
# Prepare the list of all parameter sets of the verification
parsets = [[]]
for _, items in experiment.items():
    parsets = [parset+[item] for parset in parsets for item in items]

# Now loop all parameter sets
for n, parset in enumerate(parsets):
    
    # Build parameter set
    
    p = {}
    for m, key in enumerate(experiment.keys()):
        p[key] = parset[m]
    ## apply conditional parameters
    p = cond_pars(p)
    ## include all remaining parameters
    p.update(verification)
    p.update(forecast)
    
    print("************************")
    print("* Parameter set %02d/%02d: *" % (n+1, len(parsets)))
    print("************************")
    
    pprint.pprint(p)
    
    # If necessary, build path to results
    path_to_nwc = os.path.join(cfg.path_outputs, p["experiment_name"])
    for key, item in p.items():
        if key.lower() == "data":
            path_to_nwc = os.path.join(path_to_nwc, '-'.join([item[0], item[3]]))
        elif len(experiment.get(key,[None])) > 1: # include only variables that change
            path_to_nwc = os.path.join(path_to_nwc, '-'.join([key, str(item)]))
    try:
        os.makedirs(path_to_nwc)
    except FileExistsError:
        pass
        
    # **************************************************************************
    # NOWCASTING
    # ************************************************************************** 
    
    # Loop forecasts within given event using the prescribed update cycle interval

    ## import data specifications
    ds = cfg.get_specifications(p["data"][3])
    
    # Loop forecasts for given event
    startdate   = datetime.datetime.strptime(p["data"][0], "%Y%m%d%H%M")
    enddate     = datetime.datetime.strptime(p["data"][1], "%Y%m%d%H%M")
    countnwc = 0
    while startdate + datetime.timedelta(minutes = p["n_lead_times"]*ds.timestep) <= enddate:
    
        # filename of the nowcast netcdf
        outfn = os.path.join(path_to_nwc, "%s_nowcast.netcdf" % startdate.strftime("%Y%m%d%H%M"))
    
        ## check if results already exists
        run_exist = False
        if os.path.isfile(outfn):
            fid = netCDF4.Dataset(outfn, 'r')
            if fid.dimensions["time"].size == p["n_lead_times"]:
                run_exist = True
                if p["overwrite"]:
                    os.remove(outfn)
                    run_exist = False    
            else:
                os.remove(outfn)
                
        if run_exist:
            print("Nowcast %s_nowcast already exists in %s" % (startdate.strftime("%Y%m%d%H%M"),path_to_nwc))

        else:
            countnwc += 1
            print("Computing the nowcast (%02d) ..." % countnwc)
            
            ## redirect stdout to log file
            logfn =  os.path.join(path_to_nwc, "%s_log.txt" % startdate.strftime("%Y%m%d%H%M")) 
            print("Log: %s" % logfn)
            orig_stdout = sys.stdout
            f = open(logfn, 'w')
            sys.stdout = f
            
            print("*******************")
            print("* %s *****" % startdate.strftime("%Y%m%d%H%M"))
            print("* Parameter set : *")
            pprint.pprint(p)
            print("*******************")
            
            print("--- Start of the run : %s ---" % (datetime.datetime.now()))
            
            ## time
            t0 = time.time()
        
            # Read inputs
            print("Read the data...")
            
            ## find radar field filenames
            input_files = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern,
                                              ds.fn_ext, ds.timestep, p["n_prvs_times"])
            
    
            ## read radar field files
            importer    = stp.io.get_method(ds.importer)
            R, _, metadata0 = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
            metadata = metadata0.copy()
            
            # Prepare input files
            print("Prepare the data...")
            
            ## make sure we work with a square domain
            reshaper = stp.utils.get_method(p["adjust_domain"])
            R, metadata = reshaper(R, metadata)
    
            ## if necessary, convert to rain rates [mm/h]    
            converter = stp.utils.get_method(p["unit"])
            R, metadata = converter(R, metadata)
            
            ## threshold the data
            R[R < p["r_threshold"]] = 0.0
            metadata["threshold"] = p["r_threshold"]
                
            ## transform the data
            transformer = stp.utils.get_method(p["transformation"])
            R, metadata = transformer(R, metadata)
            
            ## set NaN equal to zero
            R[~np.isfinite(R)] = metadata["zerovalue"]
            
            # Compute motion field
            oflow_method = stp.optflow.get_method(p["oflow_method"])
            UV = oflow_method(R)
            
            # Perform the nowcast       
    
            ## define the callback function to export the nowcast to netcdf
            def export(X):
                ## transform back values to mm/h
                X,_    = transformer(X, metadata, inverse=True)
                # readjust to initial domain shape
                X,_    = reshaper(X, metadata, inverse=True)
                # export to netcdf
                stp.io.export_forecast_dataset(X, exporter)
            
            ## initialize netcdf file
            incremental = "timestep" if p["nwc_method"].lower() == "steps" else None
            exporter = stp.io.initialize_forecast_exporter_netcdf(outfn, startdate,
                              ds.timestep, p["n_lead_times"], metadata["orig_domain"], 
                              p["n_ens_members"], metadata0, incremental=incremental)
            
            ## start the nowcast
            nwc_method = stp.nowcasts.get_method(p["nwc_method"])
            R_fct = nwc_method(R, UV, p["n_lead_times"], p["n_ens_members"],
                            p["n_cascade_levels"], metadata["xpixelsize"]/1000, 
                            ds.timestep, R_thr=metadata["threshold"], extrap_method=p["adv_method"],
                            decomp_method=p["decomp_method"], bandpass_filter_method=p["bandpass_filter"], 
                            noise_method=p["noise_method"], ar_order=p["ar_order"],
                            conditional=p["conditional"], use_probmatching=p["prob_matching"], 
                            mask_method=p["mask_method"], use_precip_mask=p["precip_mask"], 
                            callback=export, return_output=False, seed=p["seed"])
    
            ## save results
            stp.io.close_forecast_file(exporter)
            R_fct = None
            
            # save log
            print("--- End of the run : %s ---" % (datetime.datetime.now()))
            print("--- Total time : %s seconds ---" % (time.time() - t0))
            sys.stdout = orig_stdout
            f.close()
            
        # next forecast
        startdate += datetime.timedelta(minutes = p["data"][2])
    
    # **************************************************************************
    # VERIFICATION
    # **************************************************************************  
        
    rankhists = {}
    reldiags = {}
    rocs = {}
    for lt in p["v_leadtimes"]:
        rankhists[lt] = stp.verification.ensscores.rankhist_init(p["n_ens_members"], p["v_threshold"])
        reldiags[lt]  = stp.verification.probscores.reldiag_init(p["v_threshold"])
        rocs[lt]      = stp.verification.probscores.ROC_curve_init(p["v_threshold"])
    
    # Loop the forecasts
    startdate   = datetime.datetime.strptime(p["data"][0], "%Y%m%d%H%M")
    enddate     = datetime.datetime.strptime(p["data"][1], "%Y%m%d%H%M")
    countnwc = 0
    while startdate + datetime.timedelta(minutes = p["n_lead_times"]*ds.timestep) <= enddate:
        
        countnwc+=1
        
        print("Verifying the nowcast (%02d) ..." % countnwc)
        
        # Read observations
        
        ## find radar field filenames
        input_files = stp.io.find_by_date(startdate, ds.root_path, ds.path_fmt, ds.fn_pattern,
                                          ds.fn_ext, ds.timestep, 0, p["n_lead_times"])
                                          
        ## read radar field files
        importer = stp.io.get_method(ds.importer)
        R_obs, _, metadata_obs = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
        R_obs = R_obs[1:,:,:]
        metadata_obs["timestamps"] = metadata_obs["timestamps"][1:]
        
        ## if necessary, convert to rain rates [mm/h]   
        converter = stp.utils.get_method(p["unit"])        
        R_obs, metadata_obs = converter(R_obs, metadata_obs)  
        
        ## threshold the data
        R_obs[R_obs < p["r_threshold"]] = 0.0
        metadata_obs["threshold"] = p["r_threshold"]
            
        # Load the nowcast
        
        ## filename of the nowcast netcdf
        infn = os.path.join(path_to_nwc, "%s_nowcast.netcdf" % startdate.strftime("%Y%m%d%H%M"))
        
        print("     read: %s" % infn)
            
        ## read netcdf
        R_fct, metadata_fct = stp.io.import_netcdf_pysteps(infn)
        timestamps = metadata_fct["timestamps"]
        leadtimes = np.arange(1,len(timestamps)+1)*ds.timestep # min
        
        ## threshold the data
        R_fct[R_fct < p["r_threshold"]] = 0.0
        metadata_fct["threshold"] = p["r_threshold"]
        
        # If needed, compute accumulations
        # aggregator = stp.utils.get_method("aggregate")
        # R_obs, metadata_obs = aggregator(R_obs, metadata_obs, p["v_accu"], method="mean")
        # R_fct, metadata_fct = aggregator(R_fct, metadata_fct, p["v_accu"], method="mean")
    
        # Loop leadtimes and do verification
        for i,lt in enumerate(p["v_leadtimes"]):
            
            idlt = leadtimes == lt
            
            fig = plt.figure()
            im = stp.plt.plot_precip_field(R_obs[idlt,:,:].squeeze())
            plt.savefig(os.path.join(path_to_nwc, "%s_R_obs_%03d_%03d.png" % (startdate.strftime("%Y%m%d%H%M"),lt,p["v_accu"])))
            plt.close()        
            
            fig = plt.figure()
            im = stp.plt.plot_precip_field(R_fct[0, idlt, :, :].squeeze())
            plt.savefig(os.path.join(path_to_nwc, "%s_R_fct_%03d_%03d.png" % (startdate.strftime("%Y%m%d%H%M"),lt,p["v_accu"])))
            plt.close()
            
            R_fct_ = np.vstack([R_fct[j, idlt, :, :].flatten() for j in range(p["n_ens_members"])]).T
            stp.verification.ensscores.rankhist_accum(rankhists[lt], 
                R_fct_, R_obs[idlt, :, :].flatten())
            P_fct = 1.0*np.sum(R_fct_ >= p["v_threshold"], axis=1) / p["n_ens_members"]
            stp.verification.probscores.reldiag_accum(reldiags[lt], P_fct, R_obs[idlt, :, :].flatten())
            stp.verification.probscores.ROC_curve_accum(rocs[lt], P_fct, R_obs[idlt, :, :].flatten())
      
        # next forecast
        startdate += datetime.timedelta(minutes = p["data"][2])
      
    for i,lt in enumerate(p["v_leadtimes"]):
    
        idlt = leadtimes == lt
        
        fig = plt.figure()
        stp.verification.plot_rankhist(rankhists[lt], ax=fig.gca())
        plt.savefig(os.path.join(path_to_nwc, "rankhist_%03d_thr%.1f.png" % (lt, p["v_threshold"])), 
                bbox_inches="tight")
        plt.close()
        
        fig = plt.figure()
        stp.verification.plot_reldiag(reldiags[lt], ax=fig.gca())
        plt.savefig(os.path.join(path_to_nwc, "reldiag_%03d_thr%.1f.png" % (lt, p["v_threshold"])), 
                bbox_inches="tight")
        plt.close()
        
        fig = plt.figure()
        stp.verification.plot_ROC(rocs[lt], ax=fig.gca())
        plt.savefig(os.path.join(path_to_nwc, "roc_%03d_thr%.1f.png" % (lt, p["v_threshold"])), 
                bbox_inches="tight")
        plt.close()        