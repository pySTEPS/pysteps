"""
Advection correction (xarray version)
=====================================

This tutorial shows how to use the optical flow routines of pysteps to implement
the advection correction procedure described in Anagnostou and Krajewski (1999).

Advection correction is a temporal interpolation procedure that is often used
when estimating rainfall accumulations to correct for the shift of rainfall patterns
between consecutive radar rainfall maps. This shift becomes particularly
significant for long radar scanning cycles and in presence of fast moving
precipitation features.

Ported to xarray: the workflow now uses an xr.Dataset with a DataArray
'precip_intensity' on dims ('time','y','x'), with projection info in attrs.
"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scipy.ndimage import map_coordinates

from pysteps import io, motion, rcparams
from pysteps.utils import conversion, dimension
from pysteps.visualization import plot_precip_field

###############################################################################
# Read the radar input images
# ---------------------------

# Selected case
date = datetime.strptime("201607112100", "%Y%m%d%H%M")
data_source = rcparams.data_sources["mch"]

# Load the data from the archive
root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]

# Find the input files from the archive (36 images at 5-min --- 3 hours)
fns = io.archive.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep=5, num_next_files=35
)

# Read as xarray Dataset (expects variable 'precip_intensity' and coords x,y,time)
importer = io.get_method(importer_name, "importer")
precip_ds = io.read_timeseries(fns, importer, **importer_kwargs)

# Convert to mm/h
precip_ds = conversion.to_rainrate(precip_ds)

# Upscale to 2 km (to reduce memory)
precip_ds = dimension.aggregate_fields_space(precip_ds, 2000)

# Keep one frame every 10 minutes (every 2 timesteps)
precip_ds = precip_ds.isel(time=slice(None, None, 2))

# Ensure precip_var is present
precip_ds.attrs.setdefault("precip_var", "precip_intensity")
precip_var = precip_ds.attrs["precip_var"]

# Convenience handle to the intensity DataArray (time, y, x)
R = precip_ds[precip_var]

# Build geodata for plotting from xarray metadata/coordinates
geodata = {
    "projection": precip_ds.attrs.get("projection", None),
    "x1": float(R.x.values[0]),
    "x2": float(R.x.values[-1]),
    "y1": float(R.y.values[0]),
    "y2": float(R.y.values[-1]),
    "yorigin": "lower",
}

###############################################################################
# Advection correction
# --------------------
# We evaluate advection between two successive radar images using the
# Lucas–Kanade optical flow routine in pysteps, then do the temporal
# interpolation in between the two images (Appendix A in Anagnostou & Krajewski).

def advection_correction_xr(pair_da: xr.DataArray, T: int = 10, t: int = 1) -> xr.DataArray:
    """
    Advection correction for a pair of successive frames using optical flow motion.

    Parameters
    ----------
    pair_da : xr.DataArray
        'precip_intensity' for two consecutive times with dims ('time','y','x').
        pair_da.sizes['time'] must be 2.
    T : int
        Minutes between the two observations (here 10 after subsampling).
    t : int
        Interpolation timestep in minutes (1 minute).

    Returns
    -------
    xr.DataArray
        Time-averaged field from the temporally interpolated sequence between
        the two frames (same dims as a single frame: ('y','x')).
    """
    assert pair_da.sizes["time"] == 2, "pair_da must have exactly two time steps"

    # Optical flow on log-intensity (avoid log(0)) 
    eps = 1e-6
    oflow = motion.get_method("LK")  
    fd_kwargs = {"buffer_mask": 10}  

    pair_np = pair_da.values  # shape (2, y, x) — kept for interpolation step

    # Build a two-frame Dataset that LK expects and set precip_var attr
    pair_ds = pair_da.to_dataset(name=precip_var)
    # carry over global attrs to satisfy decorators
    pair_ds.attrs.update(precip_ds.attrs)
    pair_ds.attrs.setdefault("precip_var", precip_var)

    # log-transform 
    pair_ds[precip_var] = np.log(pair_ds[precip_var] + eps)

    # Run optical flow motion algorithm (returns velocities as variables on the same grid)
    V_ds = oflow(pair_ds)
    u = V_ds["velocity_x"].values  # x-component
    v = V_ds["velocity_y"].values  # y-component

    # Temporal interpolation loop
    ny, nx = pair_np.shape[-2], pair_np.shape[-1]
    y, x = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij")

    Rd = np.zeros((ny, nx), dtype=float)
    for i in range(t, T + t, t):
        # Backward sample from the first frame
        pos1 = (y - i / T * v, x - i / T * u)
        R1 = map_coordinates(pair_np[0], pos1, order=1, mode="nearest")

        # Forward sample from the second frame
        pos2 = (y + (T - i) / T * v, x + (T - i) / T * u)
        R2 = map_coordinates(pair_np[1], pos2, order=1, mode="nearest")

        Rd += (T - i) * R1 + i * R2

    Rd = (t / (T ** 2)) * Rd  # time-weighted mean over the interval

    # Wrap back into a DataArray with original coords
    return xr.DataArray(
        Rd, dims=("y", "x"), coords={"y": pair_da.y, "x": pair_da.x}, attrs=pair_da.attrs
    )

###############################################################################
# Apply advection correction over the whole sequence and compare with the
# simple time mean

# Naïve mean over time (left panel)
R_mean = R.mean(dim="time")

# Advection-corrected mean:
# start from the first frame, add corrected interpolations for each pair,
# then divide by number of frames
R_ac = R.isel(time=0).copy(deep=True)

for k in range(R.sizes["time"] - 1):
    pair = R.isel(time=slice(k, k + 2))
    R_ac = R_ac + advection_correction_xr(pair, T=10, t=1)

R_ac = R_ac / R.sizes["time"]

###############################################################################
# Results
# -------
plt.figure(figsize=(9, 4))

plt.subplot(1, 2, 1)
plot_precip_field(R_mean, geodata=geodata, title="3-h rainfall accumulation")

plt.subplot(1, 2, 2)
plot_precip_field(R_ac, geodata=geodata, title="Same with advection correction")

plt.tight_layout()
plt.show()

################################################################################
# Reference
# ~~~~~~~~~
#
# Anagnostou, E. N., and W. F. Krajewski. 1999. "Real-Time Radar Rainfall
# Estimation. Part I: Algorithm Formulation." Journal of Atmospheric and
# Oceanic Technology 16: 189–97.
# https://doi.org/10.1175/1520-0426(1999)016<0189:RTRREP>2.0.CO;2