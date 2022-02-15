# -*- coding: utf-8 -*-
import os
from tempfile import TemporaryDirectory

import pytest
from _pytest.outcomes import Skipped

import pysteps
from pysteps.datasets import (
    download_pysteps_data,
    create_default_pystepsrc,
    load_dataset,
)
from pysteps.exceptions import DirectoryNotEmpty


_datasets_opt_deps = dict(
    fmi=["pyproj"],
    mch=["PIL"],
    bom=["netCDF4"],
    knmi=["h5py"],
    opera=["h5py"],
    mrms=["pygrib"],
)


@pytest.mark.parametrize("case_name", _datasets_opt_deps.keys())
def test_load_dataset(case_name):
    """Test the load dataset function."""

    with pytest.raises(ValueError):
        load_dataset(frames=100)

    for mod_name in _datasets_opt_deps[case_name]:
        pytest.importorskip(mod_name)

    try:
        load_dataset(case=case_name, frames=1)
    except Skipped:
        pass


def _test_download_data():
    """Test the example data installers."""
    temp_dir = TemporaryDirectory()

    try:
        download_pysteps_data(temp_dir.name, force=True)
        with pytest.raises(DirectoryNotEmpty):
            download_pysteps_data(temp_dir.name, force=False)

        params_file = create_default_pystepsrc(temp_dir.name, config_dir=temp_dir.name)

        pysteps.load_config_file(params_file)

    finally:
        temp_dir.cleanup()
        pysteps.load_config_file()


def _default_path():
    """
    Default pystepsrc path."""
    home_dir = os.path.expanduser("~")
    if os.name == "nt":
        subdir = "pysteps"
    else:
        subdir = ".pysteps"
    return os.path.join(home_dir, subdir, "pystepsrc")


test_params_paths = [
    (None, "pystepsrc", _default_path()),
    ("/root/path", "pystepsrc", "/root/path/pystepsrc"),
    ("/root/path", "pystepsrc2", "/root/path/pystepsrc2"),
    ("relative/path", "pystepsrc2", "relative/path/pystepsrc2"),
    ("relative/path", "pystepsrc", "relative/path/pystepsrc"),
]


@pytest.mark.parametrize("config_dir, file_name, expected_path", test_params_paths)
def test_params_file_creation_path(config_dir, file_name, expected_path):
    """Test that the default pysteps parameters file is created in the right place."""

    # For windows compatibility
    if config_dir is not None:
        config_dir = os.path.normpath(config_dir)
    expected_path = os.path.normpath(expected_path)

    pysteps_data_dir = "dummy/path/to/data"
    params_file_path = create_default_pystepsrc(
        pysteps_data_dir, config_dir=config_dir, file_name=file_name, dryrun=True
    )

    assert expected_path == params_file_path
