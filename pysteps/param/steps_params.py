# Contains: StochasticRainParameters (dataclass, from_dict, to_dict), compute_field_parameters, compute_field_stats
"""
    Functions to implement the parametric version of STEPS 
"""
from typing import Optional, Tuple, Dict, Union, List, Callable
import datetime
import copy
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import xarray as xr
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

MAX_RAIN_RATE = 250
N_BINS = 200

@dataclass
class StochasticRainParameters:
    transform: Optional[str] = None
    zerovalue: Optional[float] = None
    threshold: Optional[float] = None
    kmperpixel: Optional[float] = None

    mean_db: Optional[float] = None
    stdev_db: Optional[float] = None
    nonzero_mean_rain: Optional[float] = None
    nonzero_stdev_rain: Optional[float] = None
    mean_rain: Optional[float] = None
    stdev_rain: Optional[float] = None

    psd: Optional[List[float]] = None
    psd_bins: Optional[List[float]] = None
    c1: Optional[float] = None
    c2: Optional[float] = None
    scale_break: Optional[float] = None
    cdf: Optional[List[float]] = None
    cdf_bins: Optional[List[float]] = None
    cascade_stds: Optional[List[float]] = None
    cascade_means: Optional[List[float]] = None
    cascade_lag1: Optional[List[float]] = None
    cascade_lag2: Optional[List[float]] = None
    cascade_corl: Optional[List[float]] = None

    product: Optional[str] = None
    valid_time: Optional[datetime.datetime] = None
    base_time: Optional[datetime.datetime] = None
    ensemble: Optional[int] = None
    field_id: Optional[str] = None

    # Defaulted parameters
    nonzero_mean_db: float = 2.3
    nonzero_stdev_db: float = 5.6
    rain_fraction: float = 0
    beta_1: float = -2.06
    beta_2: float = 3.2
    corl_zero: float = 260
      
    def get(self, key: str, default: Any = None) -> Any:
        """Mimic dict.get() for selected attributes."""
        return getattr(self, key, default)

    def calc_corl(self):
        """Populate the correlation lengths using lag1 and lag2 values."""
        if self.cascade_lag1 is None or self.cascade_lag2 is None:
            return

        n_levels = len(self.cascade_lag1)
        if len(self.cascade_corl) != n_levels:
            self.cascade_corl = [np.nan] * n_levels

        for ilev in range(n_levels):
            lag1 = self.cascade_lag1[ilev]
            lag2 = self.cascade_lag2[ilev]
            self.cascade_corl[ilev] = correlation_length(lag1, lag2)

        # Convenience for blending with radar
        self.corl_zero = self.cascade_corl[0]

    def calc_acor(self, config) -> None:
        T_ref = self.corl_zero 
        if T_ref is None or np.isnan(T_ref):
            T_ref = config["dynamic_scaling"]["cor_len_pvals"][1]
            
        acor, corl = power_law_acor(config, T_ref) 
        self.cascade_corl = [float(x) for x in corl]
        self.cascade_lag1 = [float(x) for x in acor[:, 0]]
        self.cascade_lag2 = [float(x) for x in acor[:, 1]] 

    @classmethod

    def from_dict(cls, data: Dict[str, Any]) -> "StochasticRainParameters":
        dbr = data.get("dbr_stats", {})
        rain = data.get("rain_stats", {})
        pspec = data.get("power_spectrum", {})
        model = pspec.get("model", {}) if pspec else {}
        cdf_data = data.get("cdf", {})
        cascade = data.get("cascade", {})
        meta = data.get("metadata", {})

        return cls(
            product=meta.get("product"),
            valid_time=meta.get("valid_time"),
            base_time=meta.get("base_time"),
            ensemble=meta.get("ensemble"),
            field_id=meta.get("field_id"),
            transform=dbr.get("transform"),
            zerovalue=dbr.get("zerovalue"),
            threshold=dbr.get("threshold"),
            kmperpixel=meta.get("kmperpixel"),

            nonzero_mean_db=dbr.get("nonzero_mean"),
            nonzero_stdev_db=dbr.get("nonzero_stdev"),
            rain_fraction=dbr.get("nonzero_fraction"),
            mean_db=dbr.get("mean"),
            stdev_db=dbr.get("stdev"),
            nonzero_mean_rain=rain.get("nonzero_mean"),
            nonzero_stdev_rain=rain.get("nonzero_stdev"),
            mean_rain=rain.get("mean"),
            stdev_rain=rain.get("stdev"),

            psd=pspec.get("psd", []),
            psd_bins=pspec.get("psd_bins", []),

            beta_1=model.get("beta_1"),
            beta_2=model.get("beta_2"),
            c1=model.get("c1"),
            c2=model.get("c2"),
            scale_break=model.get("scale_break"),

            cdf=cdf_data.get("cdf", []),
            cdf_bins=cdf_data.get("cdf_bins", []),

            corl_zero=cascade.get("corl_zero"),
            cascade_stds=cascade.get("stds"),
            cascade_means=cascade.get("means"),
            cascade_lag1=cascade.get("lag1"),
            cascade_lag2=cascade.get("lag2"),
            cascade_corl=[np.nan] * len(cascade.get("lag1", []))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dbr_stats": {
                "transform": self.transform,
                "zerovalue": self.zerovalue,
                "threshold": self.threshold,
                "nonzero_mean": self.nonzero_mean_db,
                "nonzero_stdev": self.nonzero_stdev_db,
                "nonzero_fraction": self.rain_fraction,
                "mean": self.mean_db,
                "stdev": self.stdev_db,
            },
            "rain_stats": {
                "nonzero_mean": self.nonzero_mean_rain,
                "nonzero_stdev": self.nonzero_stdev_rain,
                "nonzero_fraction": self.rain_fraction,  # assume same as dbr_stats
                "mean": self.mean_rain,
                "stdev": self.stdev_rain,
                "transform": None,
                "zerovalue": 0,
                "threshold": 0.1,
            },
            "power_spectrum": {
                "psd": self.psd,
                "psd_bins": self.psd_bins,
                "model": {
                    "beta_1": self.beta_1,
                    "beta_2": self.beta_2,
                    "c1": self.c1,
                    "c2": self.c2,
                    "scale_break": self.scale_break,
                } if any(x is not None for x in [self.beta_1, self.beta_2, self.c1, self.c2, self.scale_break]) else None
            },
            "cdf": {
                "cdf": self.cdf,
                "cdf_bins": self.cdf_bins,
            },
            "cascade": {
                "corl_zero":self.corl_zero,
                "stds": self.cascade_stds,
                "means": self.cascade_means,
                "lag1": self.cascade_lag1,
                "lag2": self.cascade_lag2,
                "corl": self.cascade_corl,
            } if self.cascade_stds is not None else None,
            "metadata": {
                "kmperpixel": self.kmperpixel,
                "product": self.product,
                "valid_time": self.valid_time,
                "base_time": self.base_time,
                "ensemble": self.ensemble,
                "field_id": self.field_id,
            }
        }


def compute_field_parameters(db_data: np.ndarray, db_metadata: dict, scale_break_km: Optional[float] = None):
    """
    Compute STEPS parameters for the dB transformed rainfall field 

    Args:
        db_data (np.ndarray): 2D field of dB-transformed rain.
        db_metadata (dict): pysteps metadata dictionary.

    Returns:
        dict: Dictionary containing STEPS parameters.
    """

    # Compute power spectrum model
    if scale_break_km is not None:
        scalebreak = scale_break_km * 1000.0 / db_metadata["xpixelsize"]
    else:
        scalebreak = None
    ps_dataset, ps_model = power_spectrum_1D(db_data, scalebreak)
    power_spectrum = {
        "psd": ps_dataset.psd.values.tolist(),
        "psd_bins": ps_dataset.psd_bins.values.tolist(),
        "model": ps_model
    }

    # Compute cumulative probability distribution
    cdf_dataset = prob_dist(db_data, db_metadata)
    cdf = {
        "cdf": cdf_dataset.cdf.values.tolist(),
        "cdf_bins": cdf_dataset.cdf_bins.values.tolist(),
    }

    # Store parameters in a dictionary
    steps_params = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "power_spectrum": power_spectrum,
        "cdf": cdf
    }
    return steps_params


def power_spectrum_1D(field: np.ndarray, scale_break: Optional[float] = None
                      ) -> Tuple[Optional[xr.Dataset], Optional[Dict[str, float]]]:
    """
    Calculate the 1D isotropic power spectrum and fit a power law model.

    Args:
        field (np.ndarray): 2D input field in [rows, columns] order.
        scale_break (float, optional): Scale break in pixel units. If None, fit single line.

    Returns:
        ps_dataset (xarray.Dataset): 1D isotropic power spectrum in dB.
        model_params (dict): Dictionary with model parameters: beta_1, beta_2, c1, c2, scale_break
    """
    min_stdev = 0.1
    mean = np.nanmean(field)
    stdev = np.nanstd(field)
    if stdev < min_stdev:
        return None, None

    norm_field = (field - mean) / stdev
    np.nan_to_num(norm_field, copy=False)

    field_fft = np.fft.rfft2(norm_field)
    power_spectrum = np.abs(field_fft) ** 2

    freq_x = np.fft.fftfreq(field.shape[1])
    freq_y = np.fft.fftfreq(field.shape[0])
    freq_r = np.sqrt(freq_x[:, None]**2 + freq_y[None, :]**2)
    freq_r = freq_r[: field.shape[0] // 2, : field.shape[1] // 2]
    power_spectrum = power_spectrum[: field.shape[0] //
                                    2, : field.shape[1] // 2]

    n_bins = power_spectrum.shape[0]
    bins = np.logspace(np.log10(freq_r.min() + 1 / n_bins),
                       np.log10(freq_r.max()), num=n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    power_1d = np.zeros(len(bin_centers))

    for i in range(len(bins) - 1):
        mask = (freq_r >= bins[i]) & (freq_r < bins[i + 1])
        power_1d[i] = np.nanmean(
            power_spectrum[mask]) if np.any(mask) else np.nan

    valid = (bin_centers > 0) & (~np.isnan(power_1d))
    bin_centers = bin_centers[valid]
    power_1d = power_1d[valid]

    if len(bin_centers) == 0:
        return None, None

    log_x = 10*np.log10(bin_centers)
    log_y = 10*np.log10(power_1d)

    start_idx = 2
    end_idx = np.searchsorted(log_x, -4.0)

    model_params = {}

    if scale_break is None:
        def str_line(X, m, c): return m * X + c
        popt, _ = curve_fit(
            str_line, log_x[start_idx:end_idx], log_y[start_idx:end_idx])
        beta_1, c1 = popt
        beta_2 = None
        c2 = None
        sb_log = None
    else:
        sb_freq = 1.0 / scale_break
        sb_log = 10*np.log10(sb_freq)

        def piecewise_linear(x, m1, m2, c1):
            c2 = (m1 - m2) * sb_log + c1
            return np.where(x <= sb_log, m1 * x + c1, m2 * x + c2)

        popt, _ = curve_fit(
            piecewise_linear, log_x[start_idx:end_idx], log_y[start_idx:end_idx])
        beta_1, beta_2, c1 = popt
        c2 = (beta_1 - beta_2) * sb_log + c1

    ps_dataset = xr.Dataset(
        {"psd": (["bin"], log_y)},
        coords={"psd_bins": (["bin"], log_x)},
        attrs={"description": "1-D Isotropic power spectrum", "units": "dB"}
    )

    model_params = {
        "beta_1": float(beta_1),
        "beta_2": float(beta_2),
        "c1": float(c1),
        "c2": float(c2),
        "scale_break": float(scale_break)
    }

    return ps_dataset, model_params


def prob_dist(data: np.ndarray, metadata: dict):
    """
    Calculate the cumulative probability distribution for rain > threshold for dB field 

    Args:
        data (np.ndarray): 2D field of dB-transformed rain.
        metadata (dict): pysteps metadata dictionary.

    Returns:
        tuple: 
            - xarray Dataset containing the cumulative probability distribution and bin edges
            - fraction of field with rain > threshold (float)
    """

    rain_mask = data > metadata["zerovalue"]

    # Compute cumulative probability distribution
    min_db = metadata["zerovalue"] + 0.1
    max_db = 10 * np.log10(MAX_RAIN_RATE)
    bin_edges = np.linspace(min_db, max_db, N_BINS)

    # Histogram of rain values
    hist, _ = np.histogram(data[rain_mask], bins=bin_edges, density=True)

    # Compute cumulative distribution
    cumulative_distr = np.cumsum(hist) / np.sum(hist)

    # Create an xarray Dataset to store both cumulative distribution and bin edges
    cdf_dataset = xr.Dataset(
        {
            "cdf": (["bin"], cumulative_distr),
        },
        coords={
            # bin_edges[:-1] to match the histogram bins
            "cdf_bins": (["bin"], bin_edges[:-1]),
        },
        attrs={
            "description": "Cumulative probability distribution of rain rates",
            "units": "dB",
        }
    )

    return cdf_dataset


def compute_field_stats(data, geodata):
    nonzero_mask = data > geodata["zerovalue"]
    nonzero_mean = np.mean(data[nonzero_mask]) if np.any(
        nonzero_mask) else np.nan
    nonzero_stdev = np.std(data[nonzero_mask]) if np.any(
        nonzero_mask) else np.nan
    nonzero_frac = np.sum(nonzero_mask) / data.size
    mean_rain = np.nanmean(data)
    stdev_rain = np.nanstd(data)

    rain_stats = {
        "nonzero_mean": float(nonzero_mean) if nonzero_mean is not None else None,
        "nonzero_stdev": float(nonzero_stdev) if nonzero_stdev is not None else None,
        "nonzero_fraction": float(nonzero_frac) if nonzero_frac is not None else None,
        "mean": float(mean_rain) if mean_rain is not None else None,
        "stdev": float(stdev_rain) if stdev_rain is not None else None,
        "transform": geodata["transform"],
        "zerovalue": geodata["zerovalue"],
        "threshold": geodata["threshold"]
    }
    return rain_stats


def get_param_by_key(
    params_df: pd.DataFrame,
    valid_time: datetime.datetime,
    base_time: Optional[datetime.datetime] = None,
    ensemble: Optional[Union[int, str]] = None,
    strict: bool = False
) -> Optional[StochasticRainParameters]:
    """
    Retrieve the StochasticRainParameters object from a DataFrame index.

    Uses 'NA' as sentinel for missing base_time/ensemble.

    Args:
        params_df (pd.DataFrame): Indexed by (valid_time, base_time, ensemble).
        valid_time (datetime): Required valid_time.
        base_time (datetime or None): Optional base_time.
        ensemble (int, str, or None): Optional ensemble.
        strict (bool): Raise KeyError if not found (default: False = return None)

    Returns:
        StochasticRainParameters or None
    """
    base_time = base_time if base_time is not None else "NA"
    ensemble = ensemble if ensemble is not None else "NA"
    try:
        return params_df.loc[(valid_time, base_time, ensemble), "param"]
    except KeyError:
        if strict:
            raise
        return None


def is_stationary(phi1, phi2):
    return abs(phi2) < 1 and (phi1 + phi2) < 1 and (phi2 - phi1) < 1


def correlation_length(lag1: float, lag2: float, dx=10, tol=1e-4, max_lag=1000):
    """
    Calculate the correlation length in minutes assuming AR(2) process 
    Args:
        lag1 (float): Lag 1 auto-correltion
        lag2 (float): Lag 2 auto-correlation
        dx (int, optional): time step between lag1 & 2 in minutes. Defaults to 10.
        tol (float, optional): _description_. Defaults to 1e-4.
        max_lag (int, optional): _description_. Defaults to 1000.

    Returns:
        corl (float): Correlation length in minutes
        np.nan on error 
    """
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
    corl = np.trapz(rho_vals, dx=dx)
    return corl


def power_law_acor(config: Dict[str, Any], T_ref: float) -> np.ndarray:
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
    dt_seconds = config["pysteps"]["timestep"]
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

    return lags, T_levels

def blend_param(qpe_params, nwp_params, param_names, weight):
    for pname in param_names:

        qval = getattr(qpe_params, pname, None)
        nval = getattr(nwp_params, pname, None)
        if isinstance(qval, (int, float)) and isinstance(nval, (int, float)):
            setattr(nwp_params, pname, weight * qval + (1 - weight) * nval)
        elif isinstance(qval, list) and isinstance(nval, list) and len(qval) == len(nval):
            setattr(nwp_params, pname, [
                    weight * q + (1 - weight) * n for q, n in zip(qval, nval)])
    return nwp_params


def blend_parameters(config, blend_base_time: datetime.datetime, nwp_param_df: pd.DataFrame, rad_param: StochasticRainParameters,    
                     weight_fn: Callable[[float], float] = None
                     ) -> pd.DataFrame:
    
    if weight_fn is None:
        def weight_fn(lag_sec): return np.exp(-(lag_sec / 10800)
                                              ** 2)  # 3h Gaussian
    blended_param_names = [
        "nonzero_mean_db",
        "nonzero_stdev_db",
        "rain_fraction", 
        "beta_1", 
        "beta_2",
        "corl_zero"
    ]
    blended_df = copy.deepcopy(nwp_param_df)
    for vtime in blended_df.index:
        lag_sec = (vtime - blend_base_time).total_seconds()
        weight = weight_fn(lag_sec)
        
        # Select the parameter object for this vtime and blend
        original = blended_df.loc[vtime, "param"] 
        clean_original = copy.deepcopy(original) 
        updated = blend_param(rad_param, clean_original, blended_param_names, weight)
        
        # Update the auto-correlations using the dynamic scaling parameters 
        updated.calc_acor(config) 
        blended_df.loc[vtime, "param"] = updated

    return blended_df 

