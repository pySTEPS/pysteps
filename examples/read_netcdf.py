import os, netCDF4
import numpy as np
from pysteps import rcparams
from pysteps.blending.utils import load_NWP

NWP_output = rcparams.outputs["path_workdir"] + "cascade_alaro13_01_20170131110000.nc"
start_time = np.datetime64("2017-01-31T11:20")

print(load_NWP(NWP_output, start_time, 4))
