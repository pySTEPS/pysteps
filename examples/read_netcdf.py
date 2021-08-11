import os, netCDF4
from datetime import datetime
from pysteps import rcparams
from pysteps.blending.utils import load_NWP

NWP_output = rcparams.outputs["NWP_outputs"] + "NWP_cascade_20170131110000.nc"
analysis_time = datetime.strptime("20170131112000", "%Y%m%d%H%M%S")

print(load_NWP(NWP_output, analysis_time, 4))
