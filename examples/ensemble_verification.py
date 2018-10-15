#!/bin/env python

"""Verification of an ensemble nowcast

The script shows how to run verification experiments for ensemble precipitation
nowcasting with pysteps.

More info: https://pysteps.github.io/
"""
import csv
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

# Verification settings
verification = {
    "experiment_name"   : "pysteps_default",
    "overwrite"         : False,            # to recompute nowcasts
    "v_thresholds"      : [0.1, 1.0],       # [mm/h]                 
    "v_leadtimes"       : [10, 30, 60],     # [min]
    "v_accu"            : None,             # [min]
    "seed"              : 42,               # for reproducibility
    "doplot"            : True,            # save figures
    "dosaveresults"     : True              # save verification scores to csv
}

# Forecast settings
forecast = {
    "n_lead_times"      : 12,       # timesteps per nowcast
    "r_threshold"       : 0.1,      # rain/no rain threshold [mm/h]
    "unit"              : "mm/h",   # mm/h or dBZ
    "transformation"    : "dB",     # None or dB 
    "adjust_domain"     : None      # None or square
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
    "oflow_method"      : ["lucaskanade"],      # lucaskanade, darts
    "adv_method"        : ["semilagrangian"],   # semilagrangian, eulerian
    "nwc_method"        : ["steps"],
    "noise_method"      : ["nonparametric"],    # parametric, nonparametric, ssft
    "decomp_method"     : ["fft"],
    
    ## the parameters
    "n_ens_members"     : [20],
    "ar_order"          : [2],
    "n_cascade_levels"  : [6],
    "noise_adjustment"  : [True],
    "conditional"       : [False],
    "precip_mask"       : [True],
    "mask_method"       : ["incremental"],      # obs, incremental, sprog
    "prob_matching"     : [True],
}

# Conditional parameters
## parameters that can be directly related to other parameters
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
    path_to_experiment = os.path.join(cfg.path_outputs, p["experiment_name"])
    for key, item in p.items():
        if key.lower() == "data":
            path_to_nwc = os.path.join(path_to_experiment, '-'.join([item[0], item[3]]))
        elif len(experiment.get(key,[None])) > 1: # include only variables that change
            path_to_nwc = os.path.join(path_to_experiment, '-'.join([key, str(item)]))
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
    
    if p["v_accu"] is None:
        p["v_accu"] = ds.timestep
    
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
            
            print("Starttime: %s" % startdate.strftime("%Y%m%d%H%M"))
            
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
            importer    = stp.io.get_method(ds.importer, type="importer")
            R, _, metadata = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
            metadata0 = metadata.copy()
            metadata0["shape"] = R.shape[1:]
            
            # Prepare input files
            print("Prepare the data...")
            
            ## if requested, make sure we work with a square domain
            reshaper = stp.utils.get_method(p["adjust_domain"])
            R, metadata = reshaper(R, metadata)
    
            ## if necessary, convert to rain rates [mm/h]    
            converter = stp.utils.get_method("mm/h")
            R, metadata = converter(R, metadata)
            
            ## threshold the data
            R[R < p["r_threshold"]] = 0.0
            metadata["threshold"] = p["r_threshold"]
            
            ## convert the data
            converter = stp.utils.get_method(p["unit"])
            R, metadata = converter(R, metadata)
                
            ## transform the data
            transformer = stp.utils.get_method(p["transformation"])
            R, metadata = transformer(R, metadata)
            
            ## set NaN equal to zero
            R[~np.isfinite(R)] = metadata["zerovalue"]
            
            # Compute motion field
            oflow_method = stp.motion.get_method(p["oflow_method"])
            UV = oflow_method(R)
            
            # Perform the nowcast       
    
            ## define the callback function to export the nowcast to netcdf
            converter   = stp.utils.get_method("mm/h")
            def export(X):
                ## convert to mm/h
                X,_ = converter(X, metadata)
                # readjust to initial domain shape
                X,_ = reshaper(X, metadata, inverse=True)
                # export to netcdf
                stp.io.export_forecast_dataset(X, exporter)
            
            ## initialize netcdf file
            incremental = "timestep" if p["nwc_method"].lower() == "steps" else None
            exporter = stp.io.initialize_forecast_exporter_netcdf(outfn, startdate,
                              ds.timestep, p["n_lead_times"], metadata0["shape"], 
                              p["n_ens_members"], metadata0, incremental=incremental)
            
            ## start the nowcast
            nwc_method = stp.nowcasts.get_method(p["nwc_method"])
            R_fct = nwc_method(R, UV, p["n_lead_times"], p["n_ens_members"],
                            p["n_cascade_levels"], kmperpixel=metadata["xpixelsize"]/1000, 
                            timestep=ds.timestep, R_thr=metadata["threshold"], 
                            extrap_method=p["adv_method"], 
                            decomp_method=p["decomp_method"], 
                            bandpass_filter_method=p["bandpass_filter"], 
                            noise_method=p["noise_method"], 
                            noise_stddev_adj=p["noise_adjustment"],
                            ar_order=p["ar_order"],conditional=p["conditional"], 
                            use_probmatching=p["prob_matching"], 
                            mask_method=p["mask_method"], 
                            use_precip_mask=p["precip_mask"], callback=export, 
                            return_output=False, seed=p["seed"])
            
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
        rankhists[lt] = stp.verification.ensscores.rankhist_init(p["n_ens_members"], p["r_threshold"])
        for thr in p["v_thresholds"]:
            reldiags[lt, thr]  = stp.verification.probscores.reldiag_init(thr)
            rocs[lt, thr]      = stp.verification.probscores.ROC_curve_init(thr) 
    
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
        importer = stp.io.get_method(ds.importer, type="importer")
        R_obs, _, metadata_obs = stp.io.read_timeseries(input_files, importer, **ds.importer_kwargs)
        R_obs = R_obs[1:,:,:]
        metadata_obs["timestamps"] = metadata_obs["timestamps"][1:]
        
        ## if necessary, convert to rain rates [mm/h]   
        converter = stp.utils.get_method("mm/h")        
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
        metadata_fct["leadtimes"] = leadtimes
        
        ## threshold the data
        R_fct[R_fct < p["r_threshold"]] = 0.0
        metadata_fct["threshold"] = p["r_threshold"]
        
        ## if needed, compute accumulations
        aggregator = stp.utils.get_method("accumulate")
        R_obs, metadata_obs = aggregator(R_obs, metadata_obs, p["v_accu"])
        R_fct, metadata_fct = aggregator(R_fct, metadata_fct, p["v_accu"])
        leadtimes = metadata_fct["leadtimes"]
        
        # Do verification
        
        ## loop leadtimes
        for i,lt in enumerate(p["v_leadtimes"]):
            
            idlt = leadtimes == lt
                        
            ## rank histogram
            R_fct_ = np.vstack([R_fct[j, idlt, :, :].flatten() for j in range(p["n_ens_members"])]).T
            stp.verification.ensscores.rankhist_accum(rankhists[lt], 
                R_fct_, R_obs[idlt, :, :].flatten())

            ## loop thresholds
            for thr in p["v_thresholds"]:    
                P_fct = 1.0*np.sum(R_fct_ >= thr, axis=1) / p["n_ens_members"]
                ## reliability diagram
                stp.verification.probscores.reldiag_accum(reldiags[lt, thr], P_fct, R_obs[idlt, :, :].flatten())
                ## roc curve
                stp.verification.probscores.ROC_curve_accum(rocs[lt, thr], P_fct, R_obs[idlt, :, :].flatten())
      
        ## next forecast
        startdate += datetime.timedelta(minutes = p["data"][2])
    
    # Write out and plot verification scores for the event
    for i,lt in enumerate(p["v_leadtimes"]):
    
        idlt = leadtimes == lt
        
        ## write rank hist results to csv file
        if verification["dosaveresults"]:
            fn = os.path.join(path_to_nwc, "rankhist_%03d_%03d.csv" % (lt, p["v_accu"]))
            with open(fn, 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in rankhists[lt].items():
                   writer.writerow([key, value])
        
        ## plot rank hist
        if verification["doplot"]:
            fig = plt.figure()
            stp.verification.plot_rankhist(rankhists[lt], ax=fig.gca())
            plt.savefig(os.path.join(path_to_nwc, "rankhist_%03d_%03d.png" % (lt, p["v_accu"])), 
                    bbox_inches="tight")
            plt.close()
        
        for thr in p["v_thresholds"]:
        
            if verification["dosaveresults"]:
                ## write rel diag results to csv file
                fn = os.path.join(path_to_nwc, "reldiag_%03d_%03d_thr%.1f.csv" % (lt, p["v_accu"], thr))
                with open(fn, 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    for key, value in reldiags[lt, thr].items():
                       writer.writerow([key, value])
                
                ## write roc curve results to csv file                
                fn = os.path.join(path_to_nwc, "roc_%03d_%03d_thr%.1f.csv" % (lt, p["v_accu"], thr))
                with open(fn, 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    for key, value in rocs[lt, thr].items():
                       writer.writerow([key, value])
        
            if verification["doplot"]:
                fig = plt.figure()
                stp.verification.plot_reldiag(reldiags[lt, thr], ax=fig.gca())
                plt.savefig(os.path.join(path_to_nwc, "reldiag_%03d_%03d_thr%.1f.png" % (lt, p["v_accu"], thr)), 
                        bbox_inches="tight")
                plt.close()
                
                fig = plt.figure()
                stp.verification.plot_ROC(rocs[lt, thr], ax=fig.gca())
                plt.savefig(os.path.join(path_to_nwc, "roc_%03d_%03d_thr%.1f.png" % (lt, p["v_accu"], thr)), 
                        bbox_inches="tight")
                plt.close()
