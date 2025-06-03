import datetime
import logging
import numpy as np
from pysteps.cascade.decomposition import decomposition_fft, recompose_fft
from pysteps.timeseries import autoregression
from pysteps import extrapolation
from models.mongo_access import get_config, get_db
from models.steps_params import StochasticRainParameters
from models.stochastic_generator import gen_stoch_field, normalize_db_field

def initialize_config(base_time_str, name):
    try:
        base_time = datetime.datetime.fromisoformat(base_time_str).replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        raise ValueError(f"Invalid base time format: {base_time_str}")

    db = get_db()
    config = get_config(db, name)
    if config is None:
        raise RuntimeError(f"Configuration not found for domain {name}")

    return db, config, base_time


def prepare_forecast_loop(db, config, base_time, name, product):
    print(f"Running {product} for domain {name} at {base_time}")
    # Placeholder for forecast generation logic
    # This is where you'd insert the time loop, forecast logic, and output handling
    pass


def update_field(cascades: list, optical_flow: np.ndarray, params: StochasticRainParameters, bp_filter: dict, config: dict) -> np.ndarray:
    """
    Update a rainfall field using the parametric STEPS algorithm.

    Args:
        cascades (list): List of rainfall cascades for previous ar_order time steps.
        optical_flow (np.ndarray): Optical flow field for Lagrangian updates.
        params (StochasticRainParameters): Parameters for the update.
        bp_filter: Bandpass filter dictionary returned by pysteps.cascade.bandpass_filters.filter_gaussian
        config: The configuration dictionary 

    Returns:
        np.ndarray: Updated rainfall field in decibels (dB) of rain intensity 
    """
    ar_order = config['pysteps']['ar_order']
    n_levels = config['pysteps']['n_cascade_levels']
    n_rows = config['domain']['n_rows']
    n_cols = config['domain']['n_cols']

    # Ensure that we have valid input parameters
    number_none_states = sum(1 for v in cascades if v is None)
    if (number_none_states != 0) or (optical_flow is None) or (params is None):
        logging.debug(
            "Missing cascade values, skipping forecast.")
        return None

    # Calculate the AR phi parameters, check if there any cascade parameters 
    if params.cascade_lag1 is None:
        logging.debug(
            "No valid cascade lag1 values found in the parameters. Skipping forecast.")
        return None
    if ar_order == 2 and params.cascade_lag2 is None:
        logging.debug(
            "No valid cascade lag2 values found in the parameters. Skipping forecast.")
        return None

    # Check if the lag 1 and lag 2 are all valid 
    number_none_lag1 = sum(1 for v in params.cascade_lag1 if np.isnan(v))  
    number_none_lag2 = 0
    if ar_order == 2:
        number_none_lag2 = sum(1 for v in params.cascade_lag2 if np.isnan(v)) 
    
    # Fill the lag1 and lag2 with the default parameters 
    if number_none_lag1 != 0 or number_none_lag2 != 0: 
        params.corl_zero = config["dynamic_scaling"]["cor_len_pvals"][1]
        params.calc_acor(config)

    phi = np.zeros((n_levels, ar_order + 1))
    for ilev in range(n_levels):
        gamma_1 = params.cascade_lag1[ilev]
        if ar_order == 2:
            gamma_2 = autoregression.adjust_lag2_corrcoef2(
                gamma_1, params.cascade_lag2[ilev])
            phi[ilev] = autoregression.estimate_ar_params_yw(
                [gamma_1, gamma_2])
        else:
            phi[ilev] = autoregression.estimate_ar_params_yw(
                [gamma_1])

    # Generate the noise field and cascade
    noise_field = gen_stoch_field(params, n_cols, n_rows)
    max_dbr = 10*np.log10(150)
    min_dbr = 10*np.log10(0.05)
    noise_field = np.clip(noise_field, min_dbr, max_dbr)
    noise_cascade = decomposition_fft(
        noise_field, bp_filter, compute_stats=True, normalize=True)

    # Update the cascade
    extrapolation_method = extrapolation.get_method("semilagrangian")
    lag_0 = np.zeros((n_levels, n_rows, n_cols))
    if ar_order == 1:
        lag_1 = cascades[0]["cascade_levels"]
    else:
        lag_2 = cascades[0]["cascade_levels"]
        lag_1 = cascades[1]["cascade_levels"]

    # Loop over cascade levels
    for ilev in range(n_levels):
        # Set the outside pixels to zero
        adv_lag1 = extrapolation_method(
            lag_1[ilev], optical_flow, 1, outval=0)[0]
        if ar_order == 1:
            lag_0[ilev] = phi[ilev, 0] * adv_lag1 + \
                phi[ilev, 1] * noise_cascade["cascade_levels"][ilev]

        else:
            # Set the outside pixels to zero
            adv_lag2 = extrapolation_method(
                lag_2[ilev], optical_flow, 2, outval=0)[1]
            lag_0[ilev] = phi[ilev, 0] * adv_lag1 + phi[ilev, 1] * \
                adv_lag2 + phi[ilev, 2] * noise_cascade["cascade_levels"][ilev]

        # Make sure we have mean = 0, stdev = 1
        lev_mean = np.mean(lag_0)
        lev_stdev = np.std(lag_0)
        if lev_stdev > 1e-1:
            lag_0 = (lag_0 - lev_mean)/lev_stdev

    # Recompose the cascade into a single field
    updated_cascade = {}
    updated_cascade["domain"] = "spatial"
    updated_cascade["normalized"] = True
    updated_cascade["compact_output"] = False
    updated_cascade["cascade_levels"] = lag_0.copy()

    # Use the noise cascade level stds
    updated_cascade["means"] = noise_cascade["means"].copy()
    updated_cascade["stds"] = noise_cascade["stds"].copy()

    # Reduce the bias in the last cascade level due to the gradient in rain / no rain
    high_freq_bias = 0.80
    updated_cascade["stds"][-1] *= high_freq_bias
    gen_field = recompose_fft(updated_cascade)

    # Normalise the field to have the expected conditional mean and variance
    norm_field = normalize_db_field(gen_field, params)

    return norm_field

def zero_state(config):
    n_cascade_levels = config['pysteps']['n_cascade_levels']
    n_rows = config['domain']['n_rows']
    n_cols = config['domain']['n_cols']
    metadata_dict = {
        "transform": config['pysteps']['transform'],
        "threshold": config['pysteps']['threshold'],
        "zerovalue": config['pysteps']['zerovalue'],
        "mean": float(0),
        "std_dev": float(0),
        "wetted_area_ratio": float(0)
    }
    cascade_dict = {
        "cascade_levels": np.zeros((n_cascade_levels, n_rows, n_cols)),
        "means": np.zeros(n_cascade_levels),
        "stds": np.zeros(n_cascade_levels),
        "domain": 'spatial',
        "normalized": True,
    }
    oflow = np.zeros((2, n_rows, n_cols))
    state = {
        "cascade": cascade_dict,
        "optical_flow": oflow,
        "metadata": metadata_dict
    }
    return state


def is_zero_state(state, tol=1e-6):
    return abs(state["metadata"]["mean"]) < tol

