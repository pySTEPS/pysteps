# -*- coding: utf-8 -*-
"""
Script used to install the pysteps data in a test environment and set a pystepsrc
configuration file that points to that data.

The test data is downloaded in the `PYSTEPS_DATA_PATH` environmental variable.

After this script is run, the `PYSTEPSRC` environmental variable should be set to
PYSTEPSRC=$PYSTEPS_DATA_PATH/pystepsrc for pysteps to use that configuration file.
"""

import os

from pysteps.datasets import create_default_pystepsrc, download_pysteps_data

tox_test_data_dir = os.environ["PYSTEPS_DATA_PATH"]

download_pysteps_data(tox_test_data_dir, force=True)

create_default_pystepsrc(
    tox_test_data_dir, config_dir=tox_test_data_dir, file_name="pystepsrc"
)
