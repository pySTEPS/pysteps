# -*- coding: utf-8 -*-
"""
Script used to install the pysteps test data in the tox environment.
"""

import json
import os

from git import Repo
from jsmin import jsmin

tox_test_data_dir = os.environ['TOX_TEST_DATA_DIR']
build_dir = os.environ['PACKAGE_ROOT']

if not os.path.isdir(os.path.join(tox_test_data_dir, ".git")):
    Repo.clone_from(
        'https://github.com/pySTEPS/pysteps-data',
        tox_test_data_dir,
        branch='master',
        depth=1
    )
else:
    test_data_repo = Repo(tox_test_data_dir)
    test_data_repo.remotes['origin'].pull()

with open(os.path.join(build_dir, "pysteps", "pystepsrc"), "r") as f:
    rcparams = json.loads(jsmin(f.read()))

for key, value in rcparams["data_sources"].items():
    new_path = os.path.join(tox_test_data_dir, value["root_path"])
    new_path = os.path.abspath(new_path)

    value["root_path"] = new_path

with open(os.path.join(tox_test_data_dir, "pystepsrc.tox"), "w") as f:
    json.dump(rcparams, f, indent=4)
