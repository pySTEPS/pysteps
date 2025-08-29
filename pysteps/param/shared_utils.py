from collections.abc import Callable

import datetime
import copy
import logging
import numpy as np
import pandas as pd

from statsmodels.tsa.api import SimpleExpSmoothing

from pysteps.cascade.decomposition import decomposition_fft, recompose_fft
from pysteps.timeseries.autoregression import (
    adjust_lag2_corrcoef2,
    estimate_ar_params_yw,
)
from pysteps import extrapolation

from pysteps.utils.transformer import DBTransformer
from .steps_params import StepsParameters
from .stochastic_generator import gen_stoch_field, normalize_db_field
from .rainfield_stats import correlation_length
from .rainfield_stats import power_spectrum_1D
from .cascade_utils import lagr_auto_cor


def update_field(
    cascades: list,
    oflow: np.ndarray,
    params: StepsParameters,
    bp_filter: dict,
    config: dict,
    dom: dict,
) -> np.ndarray:
    """
    Update a rainfall field using the parametric STEPS algorithm.
    Assumes that the cascades list has the correct number of valid cascades

    Args:
        cascades (list): List of 1 or 2 cascades for initial conditions
        oflow(np.ndarray): Optical flow array
        params (StepsParameters): Parameters for the update.
        bp_filter: Bandpass filter dictionary returned by pysteps.cascade.bandpass_filters.filter_gaussian
        config: The configuration dictionary
        dom: the domain dictionary

    Returns:
        np.ndarray: Updated rainfall field in decibels (dB) of rain intensity
    """

    ar_order = config["ar_order"]
    n_levels = config["n_cascade_levels"]
    n_rows = dom["n_rows"]
    n_cols = dom["n_cols"]

    scale_break_km = config["scale_break"]
    kmperpixel = config["kmperpixel"]

    rain_threshold = config["precip_threshold"]
    db_threshold = 10 * np.log10(rain_threshold)
    transformer = DBTransformer(rain_threshold)
    zerovalue = transformer.zerovalue

    # Set up the AR(2) parameters
    phi = np.zeros((n_levels, ar_order + 1))
    for ilev in range(n_levels):
        gamma_1 = params.lag_1[ilev]
        gamma_2 = params.lag_2[ilev]
        if ar_order == 2:
            gamma_2 = adjust_lag2_corrcoef2(gamma_1, gamma_2)
            phi[ilev] = estimate_ar_params_yw([gamma_1, gamma_2])
        else:
            phi[ilev] = estimate_ar_params_yw([gamma_1])

    # Generate the noise field and cascade
    noise_field = gen_stoch_field(
        params, n_cols, n_rows, kmperpixel, scale_break_km, db_threshold
    )
    noise_cascade = decomposition_fft(
        noise_field, bp_filter, compute_stats=True, normalize=True
    )

    # Update the cascade
    extrapolation_method = extrapolation.get_method("semilagrangian")
    lag_0 = np.zeros((n_levels, n_rows, n_cols))

    if ar_order == 2:
        lag_1 = copy.deepcopy(cascades[0]["cascade_levels"])
        lag_2 = copy.deepcopy(cascades[1]["cascade_levels"])

        for ilev in range(n_levels):
            adv_lag2 = extrapolation_method(lag_2[ilev], oflow, 2, outval=0)[1]
            adv_lag1 = extrapolation_method(lag_1[ilev], oflow, 1, outval=0)[0]
            lag_0[ilev] = (
                phi[ilev, 0] * adv_lag1
                + phi[ilev, 1] * adv_lag2
                + phi[ilev, 2] * noise_cascade["cascade_levels"][ilev]
            )

    else:
        lag_1 = copy.deepcopy(cascades[0]["cascade_levels"])
        for ilev in range(n_levels):
            adv_lag1 = extrapolation_method(lag_1[ilev], oflow, 1, outval=0)[0]
            lag_0[ilev] = (
                phi[ilev, 0] * adv_lag1
                + phi[ilev, 1] * noise_cascade["cascade_levels"][ilev]
            )

    # Make sure we have mean = 0, stdev = 1
    lev_mean = np.mean(lag_0)
    lev_stdev = np.std(lag_0)
    if lev_stdev > 1e-1:
        lag_0 = (lag_0 - lev_mean) / lev_stdev

    # Recompose the cascade into a single field
    updated_cascade = {}
    updated_cascade["domain"] = "spatial"
    updated_cascade["normalized"] = True
    updated_cascade["compact_output"] = False
    updated_cascade["cascade_levels"] = lag_0.copy()

    # Use the noise cascade level stds
    updated_cascade["means"] = noise_cascade["means"].copy()
    updated_cascade["stds"] = noise_cascade["stds"].copy()
    gen_field = recompose_fft(updated_cascade)

    # Normalise the field to have the expected conditional mean and variance
    norm_field = normalize_db_field(gen_field, params, db_threshold, zerovalue)

    return norm_field


def zero_state(config, domain):
    n_cascade_levels = config["n_cascade_levels"]
    n_rows = domain["n_rows"]
    n_cols = domain["n_cols"]
    metadata_dict = {
        "transform": None,
        "threshold": None,
        "zerovalue": None,
        "mean": float(0),
        "std_dev": float(0),
        "wetted_area_ratio": float(0),
    }
    cascade_dict = {
        "cascade_levels": np.zeros((n_cascade_levels, n_rows, n_cols)),
        "means": np.zeros(n_cascade_levels),
        "stds": np.zeros(n_cascade_levels),
        "domain": "spatial",
        "normalized": True,
    }
    oflow = np.zeros((2, n_rows, n_cols))
    state = {"cascade": cascade_dict, "optical_flow": oflow, "metadata": metadata_dict}
    return state


def is_zero_state(state, tol=1e-6):
    return abs(state["metadata"]["mean"]) < tol


#  climatology of the parameters for radar QPE 9000 sets of parameters in Auckland
#                       95%         50%         5%
# nonzero_mean_db      6.883147    4.590082   2.815397
# nonzero_stdev_db     3.793680    2.489131   1.298552
# rain_fraction        0.447717    0.048889   0.008789
# beta_1              -0.452957   -1.681647  -2.726216
# beta_2              -2.322891   -3.251342  -4.009131
# corl_zero         1074.976508  188.058276  23.489147


def qc_params(ens_df, config):
    """
    Apply QC to the 'param' column in the ensemble DataFrame.
    The DataFrame is assumed to have 'valid_time' as the index.

    Smooth corl_zero using exponential smoothing and recompute cascade autocorrelations.
    Clamp smoothed parameters to climatological bounds.
    Returns a deep-copied DataFrame with corrected parameters.
    """
    var_list = [
        "nonzero_mean_db",
        "nonzero_stdev_db",
        "rain_fraction",
        "beta_1",
        "beta_2",
    ]
    var_lower = [2.81, 1.30, 0.0, -2.73, -4.01]
    var_upper = [9.50, 5.00, 1.0, -2.05, -2.32]

    qc_df = ens_df.copy(deep=True)
    qc_dict = {}

    # Smooth each variable and clamp to bounds
    for iv, var in enumerate(var_list):
        x_list = [
            (
                np.nan
                if qc_df.at[idx, "param"].get(var) is None
                else qc_df.at[idx, "param"].get(var)
            )
            for idx in qc_df.index
        ]

        model = SimpleExpSmoothing(x_list, initialization_method="estimated").fit(
            smoothing_level=0.10, optimized=False
        )
        qc_dict[var] = np.clip(model.fittedvalues, var_lower[iv], var_upper[iv])

    # Extract correlation length thresholds from config
    corl_pvals = config["dynamic_scaling"]["cor_len_pvals"]
    corl_min = min(corl_pvals)
    corl_max = max(corl_pvals)
    corl_def = corl_pvals[1]  # median

    # Prepare and smooth corl_zero
    corl_list = []
    for idx in qc_df.index:
        corl = qc_df.at[idx, "param"].get("corl_zero", corl_def)
        corl = corl_def if corl is None else max(corl_min, min(corl, corl_max))
        corl_list.append(corl)

    model = SimpleExpSmoothing(corl_list, initialization_method="estimated").fit(
        smoothing_level=0.1, optimized=False
    )
    qc_dict["corl_zero"] = model.fittedvalues

    # Assign smoothed parameters and compute lags
    for i, idx in enumerate(qc_df.index):
        param = copy.deepcopy(qc_df.at[idx, "param"])

        # Ensure valid spectral slope order
        if qc_dict["beta_2"][i] > qc_dict["beta_1"][i]:
            qc_dict["beta_2"][i] = qc_dict["beta_1"][i]

        # Assign smoothed & clamped values
        for var in var_list:
            setattr(param, var, qc_dict[var][i])
        param.corl_zero = qc_dict["corl_zero"][i]

        # Compute lag-1 and lag-2 for this correlation length
        lags, _ = calc_auto_corls(config, param.corl_zero)
        param.lag_1 = list(lags[:, 0])
        param.lag_2 = list(lags[:, 1])

        # Save updated object
        qc_df.at[idx, "param"] = param

    return qc_df


def blend_param(qpe_params, nwp_params, param_names, weight):
    for pname in param_names:

        qval = getattr(qpe_params, pname, None)
        nval = getattr(nwp_params, pname, None)
        if isinstance(qval, (int, float)) and isinstance(nval, (int, float)):
            setattr(nwp_params, pname, weight * qval + (1 - weight) * nval)
        elif (
            isinstance(qval, list) and isinstance(nval, list) and len(qval) == len(nval)
        ):
            setattr(
                nwp_params,
                pname,
                [weight * q + (1 - weight) * n for q, n in zip(qval, nval)],
            )
    return nwp_params


def blend_parameters(
    config: dict[str, object],
    blend_base_time: datetime.datetime,
    nwp_param_df: pd.DataFrame,
    rad_param: StepsParameters,
    weight_fn: Callable[[float], float] | None = None,
) -> pd.DataFrame:
    """
    Function to blend the radar and NWP parameters

    Args:
        config (dict): Configuration dictionary
        blend_base_time (datetime.datetime): Time of the radar parameter set
        nwp_param_df (pd.DataFrame): Dataframe of valid_time and parameters,
                                     with valid_time as index and of type datetime.datetime
        rad_param (StochasticRainParameters): Parameter object with radar parameters
        weight_fn (Optional[Callable[[float], float]], optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """

    def default_weight_fn(lag_sec: float) -> float:
        return np.exp(-((lag_sec / 10800) ** 2))  # 3h Gaussian

    if weight_fn is None:
        weight_fn = default_weight_fn

    blended_param_names = [
        "nonzero_mean_db",
        "nonzero_stdev_db",
        "rain_fraction",
        "beta_1",
        "beta_2",
        "corl_zero",
    ]
    blended_df = copy.deepcopy(nwp_param_df)
    for vtime in blended_df.index:
        lag_sec = (vtime - blend_base_time).total_seconds()
        weight = weight_fn(lag_sec)

        # Select the parameter object for this vtime and blend
        original = blended_df.loc[vtime, "param"]
        clean_original = copy.deepcopy(original)
        updated = blend_param(rad_param, clean_original, blended_param_names, weight)

        # Compute lag-1 and lag-2 for this correlation length
        lags, _ = calc_auto_corls(config, updated.corl_zero)
        updated.lag_1 = list(lags[:, 0])
        updated.lag_2 = list(lags[:, 1])

        blended_df.loc[vtime, "param"] = updated

    return blended_df


def fill_param_gaps(
    ens_df: pd.DataFrame, forecast_times: list[datetime.datetime]
) -> pd.DataFrame:
    """
    Fill gaps in the time series of parameters with the most recent *original* observation
    if the gap is smaller than a threshold.

    Assumes that all the parameters have the same domain, product, base_time, ensemble.

    Args:
        ens_df (pd.DataFrame): DataFrame with columns 'valid_time' and 'param'.
        forecast_times (list): List of datetime.datetime in UTC.

    Returns:
        pd.DataFrame: DataFrame with gaps filled.
    """
    max_gap = datetime.timedelta(hours=6)

    ens_df = ens_df.copy()
    ens_df["valid_time"] = pd.to_datetime(ens_df["valid_time"], utc=True)
    ens_df = ens_df.sort_values("valid_time").reset_index(drop=True)

    filled_map = dict(zip(ens_df["valid_time"], ens_df["param"]))
    original_times = set(ens_df["valid_time"])

    # Extract default metadata
    first_param = ens_df.iloc[0].at["param"]
    def_metadata_base = first_param.metadata.copy()

    for vtime in forecast_times:
        if vtime in filled_map:
            continue

        metadata = def_metadata_base.copy()
        metadata["valid_time"] = vtime

        # Find the nearest valid time
        if original_times:
            nearest_time = min(original_times, key=lambda t: abs(t - vtime))
            gap = abs(nearest_time - vtime)

            if gap <= max_gap:
                logging.debug(
                    f"Filling {vtime} with params from nearest time {nearest_time} (gap = {gap})"
                )
                def_param = copy.deepcopy(filled_map[nearest_time])
                def_param.metadata = metadata
                def_param.rain_fraction = 0
            else:
                logging.debug(
                    f"Nearest gap too large to fill for {vtime}, using default"
                )
                def_param = StepsParameters(metadata=metadata)
        else:
            logging.debug(f"No valid parameter found near {vtime}, using default")
            def_param = StepsParameters(metadata=metadata)

        filled_map[vtime] = def_param

    records = [{"valid_time": t, "param": p} for t, p in sorted(filled_map.items())]
    return pd.DataFrame(records)


def calc_auto_corls(config: dict, T_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute lag-1 and lag-2 autocorrelations for each cascade level using a power-law model.

    Args:
        config (dict): Configuration dictionary with 'pysteps.timestep' (in seconds)
                       and 'dynamic_scaling' parameters.
        T_ref (float): Reference correlation length T(t, L) at the largest scale (in minutes).

    Returns:
        np.ndarray: Array of shape (n_levels, 2) with [lag1, lag2] for each level.
        np.ndarray: Array of corelation lengths per level
    """
    dt_seconds = config["timestep"]
    dt_mins = dt_seconds / 60.0

    ds_config = config.get("dynamic_scaling", {})
    scales = ds_config["central_wave_lengths"]
    ht = ds_config["space_time_exponent"]
    a = ds_config["lag2_constants"]
    b = ds_config["lag2_exponents"]

    L = scales[0]
    T_levels = [T_ref * (l / L) ** ht for l in scales]

    lags = np.empty((len(scales), 2), dtype=np.float32)
    for ia, T_l in enumerate(T_levels):
        pl_lag1 = np.exp(-dt_mins / T_l)
        pl_lag2 = a[ia] * (pl_lag1 ** b[ia])
        lags[ia, 0] = pl_lag1
        lags[ia, 1] = pl_lag2

    levels = np.array(T_levels)
    return lags, levels


def fit_auto_cors(
    clen: float,
    alpha: float,
    d_mins: int,
    *,
    allow_negative: bool = False,
    return_diagnostics: bool = False,
):
    """
    Find lag1, lag2 (with lag2 = lag1**alpha) such that
    correlation_length(lag1, lag2, d_mins) ~= clen.

    Parameters
    ----------
    clen : float
        Target correlation length (minutes), must be > 0.
    alpha : float
        Exponent linking lag2 and lag1 via lag2 = lag1**alpha.
    d_mins : int
        Time step between lag1 and lag2 (minutes).
    allow_negative : bool, optional
        If True, search lag1 in (-1, 1). Otherwise restrict to (0, 1).
    return_diagnostics : bool, optional
        If True, also return achieved correlation length and absolute error.

    Returns
    -------
    lag1 : float
    lag2 : float
    (achieved_clen, abs_error) : tuple[float, float], only if return_diagnostics=True
    """
    if not np.isfinite(clen) or clen <= 0:
        raise ValueError("clen must be a positive, finite number.")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")
    if not np.isfinite(d_mins) or d_mins <= 0:
        raise ValueError("d_mins must be a positive, finite number.")

    # Stability / search bounds for lag1
    eps = 1e-3
    lo, hi = (-0.999999, 0.999999) if allow_negative else (eps, 0.999999)

    # Objective: squared error on correlation length
    def _obj(l1: float) -> float:
        # Quick rejection of out-of-bounds
        if not (lo < l1 < hi):
            return np.inf
        l2 = l1**alpha

        # Keep |lag2| < 1 as well to stay in a stable region
        if not (abs(l2) < 1.0):
            return np.inf

        # Make sure that we have a valid lag1, lag2 combination
        l2 = adjust_lag2_corrcoef2(l1, l2)

        c = correlation_length(l1, l2, d_mins)
        if not np.isfinite(c):
            return np.inf
        return (c - clen) ** 2

    # Try SciPy first
    lag1 = None
    try:
        from scipy.optimize import minimize_scalar  # type: ignore

        res = minimize_scalar(
            _obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10}
        )
        lag1 = float(res.x)
    except Exception:
        # Pure NumPy golden-section fallback
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        a, b = lo, hi
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        fc = _obj(c)
        fd = _obj(d)
        # Max ~100 iterations gives ~1e-8 bracket typically
        for _ in range(100):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - (b - a) / phi
                fc = _obj(c)
            else:
                a, c, fc = c, d, fd
                d = a + (b - a) / phi
                fd = _obj(d)
            if (b - a) < 1e-10:
                break
        lag1 = float((a + b) / 2.0)

    lag2 = float(lag1**alpha)
    achieved = float(correlation_length(lag1, lag2, d_mins))
    err = abs(achieved - clen)

    if return_diagnostics:
        return lag1, lag2, (achieved, err)
    return lag1, lag2


def calc_corls(scales, czero, ht):
    # Power law function for correlation length
    corls = [czero]
    lzero = scales[0]
    for scale in scales[1:]:
        corl = czero * (scale / lzero) ** ht
        corls.append(corl)
    return corls


def calculate_parameters(
    db_field: np.ndarray,
    cascades: dict,
    oflow: np.ndarray,
    scale_break: float,
    zero_value: float,
    dt: int,
):
    p_dict = {}

    # Probability distribution moments
    nonzero_mask = db_field > zero_value
    p_dict["nonzero_mean_db"] = (
        np.mean(db_field[nonzero_mask]) if np.any(nonzero_mask) else np.nan
    )
    p_dict["nonzero_stdev_db"] = (
        np.std(db_field[nonzero_mask]) if np.any(nonzero_mask) else np.nan
    )
    p_dict["rain_fraction"] = np.sum(nonzero_mask) / db_field.size

    # Power spectrum slopes
    _, ps_model = power_spectrum_1D(db_field, scale_break)
    if ps_model:
        p_dict["beta_1"] = ps_model.get("beta_1", -2.05)
        p_dict["beta_2"] = ps_model.get("beta_2", -3.2)
    else:
        p_dict["beta_1"] = -2.05
        p_dict["beta_2"] = -3.2

    # Stack the (k,m,n) arrays in order t-2, t-1, t0 to get (t,k,m,n) array
    data = []
    for ia in range(3):
        data.append(cascades[ia]["cascade_levels"])
    data = np.stack(data)
    a_corls = lagr_auto_cor(data, oflow)
    n_levels = a_corls.shape[0]
    lag_1 = []
    lag_2 = []
    clens = []
    for ilag in range(n_levels):
        r1 = float(a_corls[ilag][0])
        r2 = float(a_corls[ilag][1])
        clen = correlation_length(r1, r2, dt)
        lag_1.append(r1)
        lag_2.append(r2)
        clens.append(clen)

    p_dict["lag_1"] = lag_1
    p_dict["lag_2"] = lag_2
    p_dict["corl_zero"] = clens[0]

    return StepsParameters.from_dict(p_dict)
