import argparse
import datetime
import logging
import numpy as np
import pandas as pd
from pymongo import UpdateOne
from models.mongo_access import get_db, get_config
from models.steps_params import power_law_acor, StochasticRainParameters

from statsmodels.tsa.api import SimpleExpSmoothing
import pymongo.collection 
from typing import Dict
logging.basicConfig(level=logging.INFO)


def get_parameters_df(query: Dict, param_coll: pymongo.collection.Collection) -> pd.DataFrame:
    """
    Retrieve STEPS parameters from the database and return a DataFrame
    indexed by (valid_time, base_time, ensemble), using 'NA' as sentinel for missing values.

    Args:
        query (dict): MongoDB query dictionary.
        param_coll (pymongo.collection.Collection): MongoDB collection.

    Returns:
        pd.DataFrame: Indexed by (valid_time, base_time, ensemble), with a 'param' column.
    """
    records = []

    for doc in param_coll.find(query).sort("metadata.valid_time", pymongo.ASCENDING):
        try:
            metadata = doc.get("metadata", {}) 
            if metadata is None:
                continue 

            if doc["cascade"]["lag1"] is None or  doc["cascade"]["lag2"] is None:
                continue
               
            valid_time = metadata.get("valid_time")
            valid_time = pd.to_datetime(valid_time,utc=True)

            base_time = metadata.get("base_time")
            if base_time is None:
                base_time = pd.NaT
            else:
                base_time =    pd.to_datetime(base_time, utc=True) 

            ensemble = metadata.get("ensemble") 

            param = StochasticRainParameters.from_dict(doc)

            param.calc_corl()
            records.append({
                "valid_time": valid_time,
                "base_time": base_time,
                "ensemble": ensemble,
                "param": param
            })
        except Exception as e:
            print(
                f"Warning: could not parse parameter for {metadata.get('valid_time')}: {e}")

    if not records:
        return pd.DataFrame(columns=["valid_time", "base_time", "ensemble", "param"])

    df = pd.DataFrame(records)
    return df

def parse_args():
    parser = argparse.ArgumentParser(
        description="QC and update NWP lag autocorrelations")
    parser.add_argument("-n", "--name", required=True, help="Domain name, e.g., AKL")
    parser.add_argument("-p","--product", required=True,
                        help="Product name, e.g., auckprec")
    parser.add_argument("-b","--base_time", required=True,
                        help="Base time, ISO format UTC (e.g., 2023-01-26T03:00:00)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run without writing to database")
    return parser.parse_args()


def qc_update_autocorrelations(dry_run: bool, name: str, product: str, base_time: datetime.datetime):
    db = get_db()
    config = get_config(db, name)
    dt = datetime.timedelta(seconds=config["pysteps"]["timestep"])
    dt_seconds = dt.total_seconds()

    corl_pvals = config["dynamic_scaling"]["cor_len_pvals"]
    corl_max = max(corl_pvals) 
    corl_min = min(corl_pvals)

    query = {
        "metadata.product": product,
        "metadata.base_time": base_time,
    }

    param_coll = db[f"{name}.params"]
    df = get_parameters_df(query, param_coll) 
    if df.empty:
        logging.warning("No parameters found for the given base_time.")
        return

    # Build corl_0 time series
    records = []

    ensembles = df["ensemble"].unique()  
    ensembles = np.sort(ensembles)  
    valid_times = df["valid_time"].unique()
    t_min = min(valid_times)
    t_max = max(valid_times)
    all_times = pd.date_range(start=t_min, end=t_max, freq=dt, tz="UTC") 
    
    # Convert the base_time to datetime64 for working with dataframe
    vbase_time = pd.NaT
    if base_time is not None:
        vbase_time = pd.to_datetime(base_time,utc=True)

    for ens in ensembles:    
        ens_df =  df.loc[ (df['base_time'] == vbase_time) & 
                     (df['ensemble'] == ens),["valid_time", "param"] ].set_index("valid_time")  
        if ens_df.empty:
            continue 

        for vt in all_times:
            try:
                param = ens_df.loc[ vt,"param"]
                corl_0 = param.corl_zero
                
                # Threshold at the 5 and 95 percentile values 
                corl_0 = corl_min if corl_0 < corl_min else corl_0
                corl_0 = corl_max if corl_0 > corl_max else corl_0
            except KeyError:
                corl_0 = np.nan

            records.append({
                "valid_time": vt,
                "ensemble": ens,
                "corl_0": corl_0
            })

    corl_df = pd.DataFrame.from_records(records)
    corl_df = corl_df.sort_values(["ensemble", "valid_time"])
    updates = []

    for ens in ensembles:
        ens_df = corl_df[corl_df["ensemble"] == ens].set_index("valid_time")
        if ens_df["corl_0"].isnull().all():
            logging.info(f"No valid corl_0 values for ensemble {ens}, skipping.")
            continue

        mean_corl = ens_df["corl_0"].mean()
        ens_df["corl_0"] = ens_df["corl_0"].fillna(mean_corl)
        ens_df.index.freq = pd.Timedelta(seconds=dt_seconds) 

        # Apply smoothing
        model = SimpleExpSmoothing(ens_df["corl_0"], initialization_method="estimated").fit(
            smoothing_level=0.2, optimized=False)
        ens_df["corl_0_smoothed"] = model.fittedvalues

        for vt in ens_df.index:
            T_ref = ens_df.loc[vt, "corl_0_smoothed"]
            lags, corl = power_law_acor(config, T_ref)
            valid_time = vt.to_pydatetime() 

            updates.append(UpdateOne(
                {
                    "metadata.product": product,
                    "metadata.valid_time": valid_time,
                    "metadata.base_time": base_time,
                    "metadata.ensemble": int(ens)
                },
                {
                    "$set": {
                        "cascade.lag1": [float(x) for x in lags[:, 0]],
                        "cascade.lag2": [float(x) for x in lags[:, 1]],
                        "cascade.corl": [float(x) for x in corl],
                        "cascade.corl_zero":float(corl[0])
                    }
                },
                upsert=False
            ))
    if updates:
        if dry_run:
            logging.info(
                f"{len(updates)} updates prepared (dry run, not written)")
        else:
            result = param_coll.bulk_write(updates)
            logging.info(f"Updated {result.modified_count} documents.")
    else:
        logging.info("No documents to update.")


if __name__ == "__main__":
    args = parse_args()
    dry_run = args.dry_run 
    base_time = datetime.datetime.fromisoformat(
        args.base_time).replace(tzinfo=datetime.timezone.utc)
    qc_update_autocorrelations(
        dry_run, args.name, args.product, base_time)
