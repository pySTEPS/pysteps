"""
Estimate the parameters that manage the AR model using the observed QPE data 
Write out the dynamic scaling configuration to a JSON file 
"""

from models import get_db, get_config
import datetime
import numpy as np
import pymongo
import argparse
import logging
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import statsmodels.api as sm
from pysteps.cascade.bandpass_filters import filter_gaussian
import json
import matplotlib.pyplot as plt


def is_valid_iso8601(time_str: str) -> bool:
    try:
        datetime.datetime.fromisoformat(time_str)
        return True
    except ValueError:
        return False


def get_auto_corls(db: pymongo.MongoClient, product: str, start_time: datetime.datetime, end_time: datetime.datetime):
    params_col = db["AKL.params"]
    query = {
        'metadata.valid_time': {'$gte': start_time, '$lte': end_time},
        'metadata.product': product
    }
    projection = {"_id": 0, "metadata": 1, "cascade": 1}
    data_cursor = params_col.find(query, projection=projection)
    data_list = list(data_cursor)

    logging.info(f'Found {len(data_list)} documents')

    rows = []
    for doc in data_list:
        row = {"valid_time": doc["metadata"]["valid_time"]}

        lag1 = doc.get("cascade", {}).get("lag1")
        lag2 = doc.get("cascade", {}).get("lag2")
        stds = doc.get("cascade", {}).get("stds")

        if lag1 is None or lag2 is None or stds is None:
            continue

        for ia, val in enumerate(lag1):
            row[f"lag1_{ia}"] = val
        for ia, val in enumerate(lag2):
            row[f"lag2_{ia}"] = val
        for ia, val in enumerate(stds):
            row[f"stds_{ia}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def power_law(x, a, b):
    return a * np.power(x, b)


def fit_power_law(qpe_df, lev):
    lag1_vals = qpe_df[f"lag1_{lev}"].values
    lag2_vals = qpe_df[f"lag2_{lev}"].values

    q05 = np.quantile(lag1_vals, 0.05)
    mask = lag1_vals > q05
    x = lag1_vals[mask]
    y = lag2_vals[mask]

    coefs, _ = curve_fit(power_law, x, y)

    y_pred = power_law(x, coefs[0], coefs[1])
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return (coefs[0], coefs[1], r_squared)

def is_stationary(phi1, phi2):
    return abs(phi2) < 1 and (phi1 + phi2) < 1 and (phi2 - phi1) < 1

def correlation_length(lag1, lag2, tol=1e-4, max_lag=1000):
    if lag1 is None or lag2 is None:
        return np.nan

    A = np.array([[1.0, lag1], [lag1, 1.0]])
    b = np.array([lag1, lag2])

    try:
        phi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.nan

    phi1, phi2 = phi
    if not is_stationary(phi1, phi2):
        return np.nan

    rho_vals = [1.0, lag1, lag2]
    for _ in range(3, max_lag):
        next_rho = phi1 * rho_vals[-1] + phi2 * rho_vals[-2]
        if abs(next_rho) < tol:
            break
        rho_vals.append(next_rho)

    return np.trapz(rho_vals, dx=10)

import os
def generate_qaqc_plots(cor_len_df, ht, scales, lag2_constants, lag2_exponents, n_levels, output_prefix=""):
    # Set up color map
    cmap = plt.colormaps.get_cmap('tab10')

    # Create output directory if it doesn't exist
    figs_dir = os.path.join("..", "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Plot fitted vs observed lag1 & lag2 for three percentiles of cl_0
    percentiles = [95, 50, 5]
    cl0_values = cor_len_df["cl_0"].values
    pvals = np.percentile(cl0_values, percentiles)
    L = scales[0]  # reference scale in km

    for pval, pstr in zip(pvals, percentiles):
        # Find closest row
        idx = (np.abs(cl0_values - pval)).argmin()
        row = cor_len_df.iloc[idx]
        time_str = row["valid_time"].strftime("%Y-%m-%d %H:%M")

        # Set up the scaling correlation lengths for this case 
        T_ref = row["cl_0"]  # T(t, L) at largest scale
        T_levels = [T_ref * (l / L) ** ht for l in scales]
        dt = 10 

        obs_lag1 = []
        obs_lag2 = []
        fit_lag1 = []
        fit_lag2 = [] 
        levels = []
        for ilevel in range(n_levels):
            lag1 = row[f"lag1_{ilevel}"]
            lag2 = row[f"lag2_{ilevel}"] 

            a = lag2_constants[ilevel]
            b = lag2_exponents[ilevel]
            pl_lag1 = np.exp(-dt / T_levels[ilevel])
            pl_lag2 = a * (pl_lag1 ** b)
            obs_lag1.append(lag1) 
            obs_lag2.append(lag2)
            fit_lag1.append(pl_lag1) 
            fit_lag2.append(pl_lag2)
            levels.append(ilevel)

        plt.figure(figsize=(6, 4))
        color_lag1 = cmap(1)
        color_lag2 = cmap(2) 

        plt.plot(scales, obs_lag1, 'x-', label='Observed lag1', color=color_lag1)
        plt.plot(scales, fit_lag1, 'o-', label='Fit lag1',color= color_lag1)
        plt.plot(scales, obs_lag2, 'x--', label='Observed lag2', color=color_lag2)
        plt.plot(scales, fit_lag2, 'o--', label='Fit lag2', color=color_lag2)

        plt.xscale("log")
        plt.xlabel("Scale (km)")
        plt.ylabel("Autocorrelation")
        plt.title(f"Fit vs Obs @ cl_0 ~ {pstr}th percentile \n{time_str}, corl len = {T_ref:.0f} min")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        filename = f"{output_prefix}lags_{pstr}th_percentile.png"
        plt.savefig(os.path.join(figs_dir, filename))
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the parameters for the dynamic scaling model")
    parser.add_argument('-s', '--start', type=str,
                        required=True, help='Start time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-e', '--end', type=str, required=True,
                        help='End time yyyy-mm-ddTHH:MM:SS')
    parser.add_argument('-n', '--name', type=str,
                        required=True, help='Name of domain [AKL]')
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help='Path to output dynamic scaling configuration file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Parse start and end time
    try:
        start_time = datetime.datetime.fromisoformat(
            args.start).replace(tzinfo=datetime.timezone.utc)
        end_time = datetime.datetime.fromisoformat(
            args.end).replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        logging.error("Invalid ISO 8601 date format for start or end time.")
        return

    name = args.name
    config_file_name = args.config
    product = "QPE"

    db = get_db()
    config = get_config(db, name)
    n_rows = config["domain"]["n_rows"]
    n_cols = config["domain"]["n_cols"]
    n_levels = config["pysteps"]["n_cascade_levels"]
    kmperpixel = config["pysteps"]["kmperpixel"]

    corl_df = get_auto_corls(db, product, start_time, end_time).dropna()

    if corl_df.empty:
        logging.error(
            "No valid correlation data found in the selected time range.")
        return

    lag2_constants, lag2_exponents = [], []

    for ilevel in range(n_levels):
        a, b, rsq = fit_power_law(corl_df, ilevel)
        if rsq < 0.5:
            logging.info(
                f"Warning: Rsq = {rsq:.2f}, using default power law for level {ilevel}")
            a, b = 1.0, 2.4
        logging.info(
            f"Level {ilevel}: lag2 = {a:.3f} * lag1^{b:.3f}, R² = {rsq:.2f}")
        lag2_constants.append(a)
        lag2_exponents.append(b)

    records = []
    for ilevel in range(n_levels):
        lag1_col = f"lag1_{ilevel}"
        lag2_col = f"lag2_{ilevel}"

        level_df = corl_df[["valid_time", lag1_col, lag2_col]].copy()
        level_df["pl_lag2"] = lag2_constants[ilevel] * \
            np.power(level_df[lag1_col], lag2_exponents[ilevel])
        level_df[f"cl_{ilevel}"] = level_df.apply(
            lambda row: correlation_length(row[lag1_col], row["pl_lag2"]), axis=1)
        records.append(
            level_df[["valid_time", f"cl_{ilevel}", lag1_col, lag2_col]])

    cor_len_df = records[0]
    for df in records[1:]:
        cor_len_df = cor_len_df.merge(df, on="valid_time", how="outer")

    cor_len_df = cor_len_df.sort_values(
        "valid_time").dropna().reset_index(drop=True)

    bp_filter = filter_gaussian((n_rows, n_cols), n_levels, kmperpixel)
    scales = 1 / bp_filter["central_freqs"]
    log_scales = np.log(scales)

    cl_columns = [f"cl_{i}" for i in range(n_levels)]
    cl_data = cor_len_df[cl_columns].values
    valid_mask = cl_data > 0
    log_cl_data = np.where(valid_mask, np.log(cl_data), np.nan)

    x_vals = np.tile(log_scales, (log_cl_data.shape[0], 1)).flatten()
    y_vals = log_cl_data.flatten()
    valid_idx = ~np.isnan(y_vals)
    x_valid, y_valid = x_vals[valid_idx], y_vals[valid_idx]

    X = sm.add_constant(x_valid)
    model = sm.OLS(y_valid, X).fit()
    a, b = model.params 

    print(model.summary())

    # Median correlation length per scale (ignoring NaNs)
    median_cl = np.nanmedian(log_cl_data, axis=0)

    # Scatter plot: log-scale vs median log(correlation length)
    plt.figure(figsize=(8, 5))
    plt.scatter(log_scales, median_cl, label="Median log(correlation length)", color='blue')

    # Regression line
    x_fit = np.linspace(min(log_scales), max(log_scales), 100)
    y_fit = a + b * x_fit
    plt.plot(x_fit, y_fit, color='red', label=f"OLS fit: y = {a:.2f} + {b:.2f}x")

    # Labels and formatting
    plt.xlabel("log(Spatial scale [km])")
    plt.ylabel("log(Correlation length [km])")
    plt.title("Median correlation length vs scale (log-log)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    percentiles = [95, 50, 5]
    cl0_values = cor_len_df["cl_0"].values
    pvals = np.percentile(cl0_values, percentiles)

    conf_dir = os.path.join("..", "run")
    conf_path = os.path.join(conf_dir, config_file_name)
    logging.info(f"Writing output dynamic scaling config to {conf_path} ") 
    with open(conf_path, "w") as f:
        dynamic_scaling_config = {"dynamic_scaling": {
            "central_wave_lengths": scales.tolist(),
            "space_time_exponent": float(b),
            "lag2_constants": lag2_constants,
            "lag2_exponents": lag2_exponents,
            "cor_len_percentiles": percentiles,
            "cor_len_pvals": pvals.tolist()
        }}
        json.dump(dynamic_scaling_config, f, indent=2)

    generate_qaqc_plots(cor_len_df, b, scales,
                        lag2_constants, lag2_exponents, n_levels)


if __name__ == "__main__":
    main()
