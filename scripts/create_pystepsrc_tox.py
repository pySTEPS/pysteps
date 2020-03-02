# -*- coding: utf-8 -*-
"""
Script used to install the pysteps test data in the tox environment.
"""

import os

from pysteps.dataset import create_default_pystepsrc, download_pysteps_data

tox_test_data_dir = os.environ['TOX_TEST_DATA_DIR']

download_pysteps_data(tox_test_data_dir, force=True)

create_default_pystepsrc(tox_test_data_dir,
                         config_dir=tox_test_data_dir,
                         file_name="pystepsrc.tox")
