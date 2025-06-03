from typing import List, Callable
import argparse
import logging
import datetime
import numpy as np
import copy
import os
import sys
import pandas as pd
from cascade.bandpass_filters import filter_gaussian
from utils import transformation
from cascade.decomposition import decomposition_fft
from mongo.nc_utils import generate_geo_data, make_nc_name_dt, write_netcdf
from mongo.gridfs_io import get_states, load_rain_field
from mongo.mongo_access import get_base_time, get_parameters_df
from steps_params import StochasticRainParameters, blend_parameters
from shared_utils import initialize_config
from shared_utils import zero_state, update_field

  
def get_weight(lag):
    width = 3 * 3600
    weight = np.exp(-(lag/width)**2)
    return weight


def main():

    parser = argparse.ArgumentParser(description="Run nwpblend forecasts")
    parser.add_argument('-b', '--base_time', required=True,
                        help='Base time in ISO 8601 format')
    parser.add_argument('-n', '--name', required=True,
                        help='Domain name (e.g., AKL)')
    args = parser.parse_args()


    # Include app name (module name) in log output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout
    )

    logger = logging.getLogger(__name__)
    logger.info("Gemerating nwpblend ensembles")

    name = args.name
    db, config, out_base_time = initialize_config(args.base_time, name)

    param_coll = db[f"{name}.params"]
    meta_coll = db[f"{name}.rain.files"]

    time_step_seconds = config['pysteps']['timestep']
    time_step = datetime.timedelta(seconds=time_step_seconds)
    ar_order = config['pysteps']['ar_order']
    n_levels = config['pysteps']['n_cascade_levels']
    db_threshold = config['pysteps']['threshold']
    scale_break = config['pysteps']['scale_break']

    # Set up the georeferencing data for the output forecasts
    domain = config['domain']
    start_x = domain['start_x']
    start_y = domain['start_y']
    p_size = domain['p_size']
    n_rows = domain['n_rows']
    n_cols = domain['n_cols']
    x = [start_x + i * p_size for i in range(n_cols)]
    y = [start_y + i * p_size for i in range(n_rows)]
    geo_data = generate_geo_data(x, y)
    geo_data["projection"] = config['projection']["epsg"]

    # Set up the bandpass filter
    p_size_km = p_size / 1000.0
    bp_filter = filter_gaussian((n_rows, n_cols), n_levels, d=p_size_km)

    # Configure the output product
    out_product = "nwpblend"
    out_config = config['output'][out_product]
    n_ens = out_config.get('n_ens_members', 10)
    n_forecasts = out_config.get('n_forecasts', 12)
    rad_product = out_config.get('rad_product', None)
    nwp_product = out_config.get('nwp_product', None)
    gridfs_out = out_config.get('gridfs_out', False)
    nc_out = out_config.get('nc_out', False)
    out_dir_name = out_config.get('out_dir_name', None)
    out_file_name = out_config.get(
        'out_file_name', "$N_$P_$V{%Y-%m-%dT%H:%M:%S}_$B{%Y-%m-%dT%H:%M:%S}_$E.nc")

    # Validate the output configuration details
    if rad_product is None:
        logging.error(f"Radar product not specified")
        return
    if n_ens < 1:
        logging.error(f"Invalid number of ensemble members: {n_ens}")
        return
    if n_forecasts < 1:
        logging.error(f"Invalid number of lead times: {n_forecasts}")
        return
    if not gridfs_out and not nc_out:
        logging.error(
            "No output format specified. Please set either gridfs_out or nc_out to True.")
        return

    if nc_out:
        if out_dir_name is None:
            logging.error(f"No output directory name found")
            return

    logging.info(f"Generating nwpblend for {out_base_time}")

    # Make the list of output forecast times
    forecast_times = [out_base_time + ia *
                      time_step for ia in range(0, n_forecasts+1)] 
    

    # Get the initial state(s) for the input radar field at this base time
    # Set any missing states to None
    base_time_key = "NA"
    ensemble_key = "NA"
    init_times = [out_base_time]
    if ar_order == 2:
        init_times = [out_base_time - time_step, out_base_time]

    query = {
        "metadata.product": rad_product,
        "metadata.valid_time": {"$in": init_times}
    }
    init_state = get_states(db, name, query,
                            get_cascade=True, get_optical_flow=True)
    rad_params_df = get_parameters_df(query, param_coll)

    for vtime in init_times:
        key = (vtime, base_time_key, ensemble_key)
        if key not in init_state:
            init_state[key] = zero_state(config)
            logging.debug(f"Found missing QPE oflow for {vtime}")

        # Check if row exists for this combination
        mask = (
            (rad_params_df["valid_time"] == vtime) &
            (rad_params_df["base_time"] == "NA") &
            (rad_params_df["ensemble"] == "NA")
        )

        if rad_params_df[mask].empty:
            logging.debug(f"Found missing QPE parametersfor {vtime}")

            def_param = StochasticRainParameters()
            def_param.calc_acor(config) 
            def_param.kmperpixel = p_size_km 
            def_param.scale_break = scale_break
            def_param.threshold = db_threshold

            new_row = {
                "valid_time": vtime,
                "base_time": "NA",
                "ensemble": "NA",
                "param": def_param
            }
            rad_params_df = pd.concat([rad_params_df, pd.DataFrame([new_row])], ignore_index=True)

    # Get the base_time for the nwp run nearest to the output base_time
    nwp_base_time = get_base_time(out_base_time, nwp_product, name, db)

    # Get the list of ensemble members for this nwp_base_time
    query = {
        "metadata.product": nwp_product,
        "metadata.base_time": nwp_base_time}
    nwp_ensembles = meta_coll.distinct("metadata.ensemble", query)
    if nwp_ensembles is None:
        logging.warning(
            f"Failed to find ensembles for {nwp_product} data for {out_base_time}")
    nwp_ensembles.sort()
    n_nwp_ens = len(nwp_ensembles)

    # Get the NWP parameters and optical flows for the NWP ensemble
    query = {
        "metadata.product": nwp_product,
        "metadata.valid_time": {"$in": forecast_times},
        'metadata.base_time': nwp_base_time
    }
    nwp_params_df = get_parameters_df(query, param_coll)
    nwp_oflows = get_states(
        db, name, query, get_cascade=False, get_optical_flow=True)

    # Start the loop over the ensemble members
    for iens in range(n_ens):
        
        # Calculate the set of blended parameters for this output ensemble 
        # Get the radar parameter 
        qpe_rows = rad_params_df[
            (rad_params_df["valid_time"] == out_base_time) &
            (rad_params_df["base_time"] == "NA") &
            (rad_params_df["ensemble"] == "NA")
        ]
        rad_param = qpe_rows.iloc[0]["param"]

        # Randomly select an ensemble member from the NWP
        nwp_ens = np.random.randint(low=0, high=n_nwp_ens)
        nwp_ensemble_df = nwp_params_df[
            (nwp_params_df["base_time"] == nwp_base_time) &
            (nwp_params_df["ensemble"] == nwp_ens)
        ][["valid_time", "param"]].copy()
        nwp_ensemble_df["valid_time"] = pd.to_datetime(nwp_ensemble_df["valid_time"])
        nwp_ensemble_df.set_index("valid_time", inplace=True) 
        nwp_ensemble_df = nwp_ensemble_df.sort_index()

        # Fill in any missing forecast times with default parameters
        for vtime in forecast_times:
            if vtime not in nwp_ensemble_df.index:
                def_param = StochasticRainParameters()
                def_param.calc_acor(config)     
                def_param.kmperpixel = p_size_km 
                def_param.scale_break = scale_break
                def_param.threshold = db_threshold
                nwp_ensemble_df.loc[vtime,"param"] = def_param

        # Blend the parameters 
        blend_params_df = blend_parameters(config, out_base_time, nwp_ensemble_df, rad_param)

        # Set up the initial conditions for the forecast loop
        # The order is [t-1, t0] in init_times for AR(2)
        if ar_order == 1:
            key = (init_times[0], "NA", "NA")
            state = init_state.get(key)

            if state is not None:
                cascade = state.get("cascade")
                optical = state.get("optical_flow")  
                fx_cascades = [copy.deepcopy(cascade)] if cascade is not None else [None]
                fx_oflow = copy.deepcopy(optical) if optical is not None else None
            else:
                fx_cascades = [None]
                fx_oflow = None

        else:  # AR(2)
            key_0 = (init_times[0], "NA", "NA")
            key_1 = (init_times[1], "NA", "NA")

            state_0 = init_state.get(key_0)
            state_1 = init_state.get(key_1)

            if state_0 is not None and state_1 is not None:
                casc_0 = state_0.get("cascade")
                casc_1 = state_1.get("cascade")
                optical = state_1.get("optical_flow")

                fx_cascades = [
                    copy.deepcopy(casc_0) if casc_0 is not None else None,
                    copy.deepcopy(casc_1) if casc_1 is not None else None
                ]
                fx_oflow = copy.deepcopy(optical) if optical is not None else None
            else:
                fx_cascades = [None, None]
                fx_oflow = None

        # Start the forecast loop 
        for ifx in range(1, n_forecasts+1):
            valid_time = forecast_times[ifx]
            fx_param = blend_params_df.loc[valid_time, "param"]

            fx_dbrain = update_field(
                fx_cascades, fx_oflow, fx_param, bp_filter, config)
            has_nan = np.isnan(fx_dbrain).any() if fx_dbrain is not None else True

            if has_nan :
                fx_rain = np.zeros((n_rows, n_cols))
            else:
                fx_rain, _ = transformation.dB_transform(
                        fx_dbrain, inverse=True, threshold=db_threshold, zerovalue=0)

            # Make the output file name
            fx_file_name = make_nc_name_dt(
                out_file_name, name, out_product, valid_time, out_base_time, iens)

            # Write the NetCDF data to a memoryview buffer
            # This is an ugly hack on time zones
            vtime = valid_time
            if vtime.tzinfo is None:
                vtime = vtime.replace(tzinfo=datetime.timezone.utc)
            btime = out_base_time
            if btime.tzinfo is None:
                btime = btime.replace(tzinfo=datetime.timezone.utc)
            vtime_stamp = vtime.timestamp()

            nc_buf = write_netcdf(fx_rain, geo_data, vtime_stamp)

            if gridfs_out:
                # Create metadata
                rain_mask = fx_rain.copy()
                rain_mask[rain_mask < 1] = 0
                rain_mask[rain_mask > 0] = 1
                war = rain_mask.sum() / (n_cols * n_rows)
                mean = np.nanmean(fx_rain)
                std_dev = np.nanstd(fx_rain)
                max = np.nanmax(fx_rain)
                metadata = {
                    "product": out_product,
                    "domain": name,
                    "ensemble": int(iens),
                    "base_time": btime,
                    "valid_time": vtime,
                    "mean": float(mean),
                    "wetted_area_ratio": float(war),
                    "std_dev": float(std_dev),
                    "max": float(max),
                    "forecast_lead_time": int(ifx*time_step_seconds)
                }
                load_rain_field(db, name, fx_file_name, nc_buf, metadata)

            if nc_out:
                fx_dir_name = make_nc_name_dt(
                    out_dir_name, name, out_product, valid_time, out_base_time, iens)
                if not os.path.exists(fx_dir_name):
                    os.makedirs(fx_dir_name)
                fx_file_path = os.path.join(fx_dir_name, fx_file_name)
                with open(fx_file_path, 'wb') as f:
                    f.write(nc_buf.tobytes())

            # Update the cascade state list for the next forecast step
            if ar_order == 2:
                # Push the cascade history down (t0 → t-1)
                fx_cascades[0] = copy.deepcopy(fx_cascades[1])

                # Update the latest cascade (t0) from current forecast brain
                if fx_dbrain is not None:
                    if has_nan:
                        fx_cascades[1] = zero_state(config)["cascade"]
                        logging.warning(f"NaNs found for {valid_time}, {iens} ")
                    else:
                        fx_cascades[1] = decomposition_fft(
                            fx_dbrain, bp_filter, compute_stats=True, normalize=True
                        )
                else:
                    fx_cascades[1] = zero_state(config)["cascade"]

            elif ar_order == 1:
                # Only update the current cascade
                if fx_dbrain is not None:
                    if has_nan:
                        fx_cascades[0] = zero_state(config)["cascade"]
                        logging.warning(f"NaNs found for {valid_time}, {iens} ")
                    else:
                        fx_cascades[0] = decomposition_fft(
                            fx_dbrain, bp_filter, compute_stats=True, normalize=True
                        )
                else:
                    fx_cascades[0] = zero_state(config)["cascade"]

            # Update the optical flow field using radar–NWP blending
            if ifx < n_forecasts:
                rad_key = (out_base_time, "NA", "NA")
                nwp_key = (out_base_time, nwp_base_time, nwp_ens)

                lag = (valid_time - out_base_time).total_seconds()
                weight = get_weight(lag)

                # Check availability of both radar and NWP optical flows
                rad_oflow = init_state.get(rad_key, {}).get("optical_flow")
                nwp_oflow_entry = nwp_oflows.get(nwp_key)
                nwp_oflow = nwp_oflow_entry.get("optical_flow") if nwp_oflow_entry else None

                if rad_oflow is not None and nwp_oflow is not None:
                    fx_oflow = weight * rad_oflow + (1 - weight) * nwp_oflow
                else:
                    fx_oflow = None

if __name__ == "__main__":
    main()
