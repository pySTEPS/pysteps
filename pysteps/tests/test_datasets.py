# -*- coding: utf-8 -*-
from tempfile import TemporaryDirectory

import pytest

import pysteps
from pysteps import datasets
from pysteps.datasets import download_pysteps_data, create_default_pystepsrc, load_dataset
from pysteps.exceptions import DirectoryNotEmpty


def test_install_example_data():
    """Test the example data installers."""
    temp_dir = TemporaryDirectory()
    try:
        download_pysteps_data(temp_dir.name, force=True)

        with pytest.raises(DirectoryNotEmpty):
            download_pysteps_data(temp_dir.name, force=False)

        params_file = create_default_pystepsrc(temp_dir.name, config_dir=temp_dir.name)

        pysteps.load_config_file(params_file)

        with pytest.raises(ValueError):
            load_dataset(frames=100)

        for case_name in datasets._precip_events.keys():
            load_dataset(case=case_name, frames=24)
    finally:
        temp_dir.cleanup()
