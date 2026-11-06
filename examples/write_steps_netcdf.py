"""
Save a STEPS ensemble and its inputs to NetCDF
==============================================

Generate a small STEPS ensemble from three Swiss radar composites and save
both the physical input rain rates and the forecast in one packed int16
CF-1.8 NetCDF file. Read the file back and plot its contents.

This example uses the MCH data from the :ref:`example_data` archive and the
configured ``mch`` data source in :ref:`pystepsrc`. It requires netCDF4 and
OpenCV for NetCDF output and Lucas-Kanade motion estimation, respectively.
The workflow follows :doc:`plot_steps_nowcast`; see also the
:ref:`cf-netcdf-user-guide` for storage and compatibility options.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pysteps import io, motion, nowcasts, rcparams
from pysteps.dataset_metadata import linked, rename_with_links
from pysteps.utils import conversion, dimension, transformation
from pysteps.visualization import plot_precip_field
from pysteps.xarray_helpers import geodata_from_dataset

###############################################################################
# Read three radar fields
# -----------------------
#
# Keep the input fields in physical units, including their missing pixels.
# Upscale to 2 km to reduce the cost of this demonstration. These are the three
# fields saved in the output file; the original 1 km composites remain in the
# source archive.

date = datetime(2017, 1, 31, 12, 0)
source = rcparams.data_sources["mch"]
n_ens_members = 4
n_leadtimes = 6

filenames = io.find_by_date(
    date,
    source["root_path"],
    source["path_fmt"],
    source["fn_pattern"],
    source["fn_ext"],
    source["timestep"],
    num_prev_files=2,
)
if any(filename is None for filename in filenames[0]):
    raise FileNotFoundError("This example requires all three MCH radar composites.")

importer = io.get_method(source["importer"], "importer")
observations = io.read_timeseries(filenames, importer, **source["importer_kwargs"])
observations = conversion.to_rainrate(observations)
observations = dimension.aggregate_fields_space(observations, 2000)
precip_var = observations.attrs["precip_var"]
assert observations.sizes["time"] == 3

###############################################################################
# Generate an ensemble in model space
# -----------------------------------
#
# Estimate motion and run STEPS on a separate dBR-transformed copy. Leave NaNs
# in place: STEPS handles its computational fill internally and restores its
# domain mask. Filling the saved observations with zero would turn missing
# pixels into observed dry pixels.

model_input = transformation.dB_transform(
    observations.copy(deep=True), threshold=0.1, zerovalue=-15.0
)
model_input = motion.get_method("LK")(model_input)
forecast = nowcasts.get_method("steps")(
    model_input,
    timesteps=n_leadtimes,
    n_ens_members=n_ens_members,
    n_cascade_levels=6,
    precip_thr=-10.0,
    kmperpixel=2.0,
    timestep=source["timestep"],
    noise_method="nonparametric",
    vel_pert_method="bps",
    mask_method="incremental",
    seed=24,
    num_workers=1,
)

###############################################################################
# Return to physical rain rates before export
# -------------------------------------------
#
# NetCDF contains mm/h values, rather than the statistical dBR transform used
# by STEPS. The forecast carries valid times, a forecast reference time, member
# labels and the five-minute interval bounds inherited from this rate product.
# These bounds describe each forecast interval, rather than a running total
# from initialization.

forecast = transformation.dB_transform(forecast, threshold=-10.0, inverse=True)

###############################################################################
# Combine observations and forecast without broadcasting
# ------------------------------------------------------
#
# Use different variable names and time dimensions. The observations have shape
# (observation_time, y, x), while the ensemble has shape
# (ens_number, time, y, x). Both share the projected grid and CRS. Renaming with
# the helper also updates CF bounds links and temporal cell methods, which
# xarray's rename alone does not rewrite.

observation_names = {precip_var: "observed_rainrate", "time": "observation_time"}
observation_bounds = linked(observations.time, "bounds")
if observation_bounds is not None:
    observation_names[observation_bounds] = "observation_time_bounds"
saved_inputs = rename_with_links(observations, observation_names)
saved_forecast = rename_with_links(forecast, {precip_var: "forecast_rainrate"})
product = xr.merge(
    [saved_forecast, saved_inputs],
    compat="no_conflicts",
    join="exact",
    combine_attrs="override",
)
product.attrs.update(
    precip_var="forecast_rainrate",
    title="STEPS ensemble and three radar input fields",
    source="MCH AQC radar composites from the pySTEPS example archive",
    history=product.attrs.get("history", "")
    + "\n"
    + (
        "Converted three five-minute radar depths to interval mean rain rates; "
        "aggregated to 2 km; estimated Lucas-Kanade motion; generated a "
        "four-member STEPS ensemble with seed 24; inverted the dBR transform."
    ),
)

# Use a fresh directory on each run. The exporter protects existing files and
# publishes the completed file atomically. Pack both rain-rate variables at
# 0.1 mm/h per code, with a reserved missing code and zlib compression. This
# represents 0 to 3276.7 mm/h with at most 0.05 mm/h quantization error, plus
# floating-point tolerance. Positive traces below half a step may round to zero.
# Coordinates, times and member labels retain their own lossless encodings.
packing_step = 0.1
fill_code = -32768
field_names = ("observed_rainrate", "forecast_rainrate")
encoding = {
    name: {
        "dtype": "int16",
        "scale_factor": packing_step,
        "add_offset": 0.0,
        "_FillValue": fill_code,
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
    }
    for name in field_names
}
output_dir = Path(tempfile.mkdtemp(prefix="pysteps-steps-netcdf-"))
output_path = io.export_netcdf_cf(
    product, output_dir / "steps_with_inputs.nc", encoding=encoding
)
print(f"Saved inputs and ensemble to {output_path}")

###############################################################################
# Read back and check the stored products
# ---------------------------------------
#
# Select the forecast explicitly because the file has two physical variables.
# Selection keeps the input observations on their own time axis. The default
# eager importer closes the file before returning the dataset. Decoded physical
# values are floating point even though the file stores int16 codes. Check
# masks exactly and values to half a packing step, rather than exact equality.

restored = io.import_netcdf_cf(output_path, variable="forecast_rainrate")
for name in field_names:
    np.testing.assert_array_equal(np.isnan(restored[name]), np.isnan(product[name]))
    np.testing.assert_allclose(
        restored[name],
        product[name],
        rtol=0,
        atol=packing_step / 2 + 1e-12,
        equal_nan=True,
    )

# Inspect the undecoded file independently to verify the integer storage and
# missing codes. Automatic CF decoding would hide these storage details.
with xr.open_dataset(output_path, decode_cf=False) as stored:
    for name in field_names:
        assert stored[name].dtype == np.dtype("int16")
        assert stored[name].attrs["scale_factor"] == packing_step
        assert stored[name].attrs["add_offset"] == 0.0
        assert stored[name].attrs["_FillValue"] == fill_code
        np.testing.assert_array_equal(
            stored[name] == fill_code, np.isnan(product[name])
        )

for name in (
    "x",
    "y",
    "time",
    "observation_time",
    "ens_number",
    "forecast_reference_time",
):
    np.testing.assert_array_equal(restored[name], product[name])
for axis in ("time", "observation_time"):
    bounds = linked(product[axis], "bounds")
    np.testing.assert_array_equal(restored[bounds], product[bounds])
assert restored.observed_rainrate.dims == ("observation_time", "y", "x")
assert restored.forecast_rainrate.dims == ("ens_number", "time", "y", "x")
print(restored[["observed_rainrate", "forecast_rainrate"]])

###############################################################################
# Plot the fields read from NetCDF
# --------------------------------
#
# Compare the last input with the ensemble mean and one member at 30 minutes.
# This illustrates file I/O; it does not evaluate forecast skill.

geodata = geodata_from_dataset(restored)
lead_minutes = n_leadtimes * source["timestep"]
panels = [
    (restored.observed_rainrate.isel(observation_time=-1), "Last radar input"),
    (
        restored.forecast_rainrate.isel(time=-1).mean("ens_number", skipna=False),
        f"Ensemble mean (+{lead_minutes:g} min)",
    ),
    (
        restored.forecast_rainrate.isel(ens_number=0, time=-1),
        f"Member {restored.ens_number.values[0]} (+{lead_minutes:g} min)",
    ),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (field, title) in zip(axes, panels):
    plt.sca(ax)
    plot_precip_field(
        field.values,
        ax=ax,
        geodata=geodata,
        title=title,
        axis="off",
        map_kwargs={"plot_map": None},
    )
plt.tight_layout()
plt.show()
