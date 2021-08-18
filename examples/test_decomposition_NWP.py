import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from pprint import pprint
from pysteps import io, nowcasts, rcparams
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import conversion, dimension, transformation
from pysteps.visualization import plot_precip_field
from pysteps.blending.utils import decompose_NWP
from pysteps.cascade.bandpass_filters import filter_gaussian

num_cascade_levels = 8

###############################################################################
# Read precipitation field
# ------------------------
#
# First thing, the sequence of Swiss radar composites is imported, converted and
# transformed into units of dBR.


date = datetime.strptime("201701311200", "%Y%m%d%H%M")
data_source = "mch"

# Load data source config
root_path = rcparams.data_sources[data_source]["root_path"]
path_fmt = rcparams.data_sources[data_source]["path_fmt"]
fn_pattern = rcparams.data_sources[data_source]["fn_pattern"]
fn_ext = rcparams.data_sources[data_source]["fn_ext"]
importer_name = rcparams.data_sources[data_source]["importer"]
importer_kwargs = rcparams.data_sources[data_source]["importer_kwargs"]
timestep = rcparams.data_sources[data_source]["timestep"]

# Find the radar files in the archive
fns = io.find_by_date(
    date, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=12
)

# Read the data from the archive
importer = io.get_method(importer_name, "importer")
R_NWP, _, metadata = io.read_timeseries(fns, importer, legacy=True, **importer_kwargs)

# Convert to rain rate
R_NWP, metadata = conversion.to_rainrate(R_NWP, metadata)

# Log-transform the data
R_NWP, metadata = transformation.dB_transform(
    R_NWP, metadata, threshold=0.1, zerovalue=-15.0
)

# Fill in the missing data with the threshold value
R_NWP[~np.isfinite(R_NWP)] = metadata["zerovalue"]

# Find the location to save the NWP files
NWP_output = rcparams.outputs["path_workdir"]

# Define the start time of the NWP forecast
analysis_time = metadata["timestamps"][0]

# Decompose the NWP and save to netCDF file
decompose_NWP(
    R_NWP,
    "alaro13_01",
    analysis_time,
    5,
    metadata["timestamps"],
    num_cascade_levels,
    NWP_output,
)
