# -*- coding: utf-8 -*-
import os

import json
from jsmin import jsmin

with open(os.path.join("pysteps", "pystepsrc"), "r") as f:
    rcparams = json.loads(jsmin(f.read()))

for key, value in rcparams["data_sources"].items():
    original_path = value["root_path"]

    new_path = os.path.join("pysteps-data", value["root_path"])
    new_path = os.path.abspath(new_path)

    value["root_path"] = new_path

with open("pystepsrc.travis", "w") as f:
    json.dump(rcparams, f, indent=4)
