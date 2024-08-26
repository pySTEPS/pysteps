# -*- coding: utf-8 -*-
"""
Script to test the plugin support.

https://github.com/pySTEPS/cookiecutter-pysteps-plugin
"""
import os
import pytest
import subprocess
import sys
import tempfile

__ = pytest.importorskip("cookiecutter")
from cookiecutter.main import cookiecutter

PLUGIN_TEMPLATE_URL = "https://github.com/pysteps/cookiecutter-pysteps-plugin"

from contextlib import contextmanager
import pysteps


def _check_installed_plugin(import_func_name):
    # reload the pysteps module to detect the installed plugin
    pysteps.io.discover_importers()
    assert hasattr(pysteps.io.importers, import_func_name)
    assert import_func_name in pysteps.io.interface._importer_methods
    importer = getattr(pysteps.io.importers, import_func_name)
    importer("filename")


@contextmanager
def _create_and_install_plugin(project_name, importer_name):
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"Installing plugin {project_name} providing the {importer_name} module")
        cookiecutter(
            PLUGIN_TEMPLATE_URL,
            no_input=True,
            overwrite_if_exists=True,
            extra_context={
                "project_name": project_name,
                "importer_name": importer_name,
            },
            output_dir=tmpdirname,
        )
        # Install the plugin
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                os.path.join(tmpdirname, project_name),
            ]
        )

        # The block below, together with the decorator used in this function are used
        # to create a context manager that uninstall the plugin packages after the
        # tests finish (even if they fail).
        # https://docs.pytest.org/en/stable/fixture.html?highlight=context#fixture-finalization-executing-teardown-code
        try:
            yield project_name
        finally:
            _uninstall_plugin(project_name)


def _uninstall_plugin(project_name):
    # Install the plugin
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", project_name]
    )


def test_importers_plugins():
    with _create_and_install_plugin("test_importer_aaa", "importer_aaa"):
        # The default plugin template appends an _xxx to the importer function.
        _check_installed_plugin("importer_aaa_xxx")
        with _create_and_install_plugin("test_importer_bbb", "importer_bbb"):
            _check_installed_plugin("importer_aaa_xxx")
            _check_installed_plugin("importer_bbb_xxx")


def test_nowcasts_plugin_discovery():
    """Testing the discover_nowcast method to effectively see whether the plugin has been installed."""

    from unittest.mock import Mock, patch

    # Ensure we patch the correct location where iter_entry_points is used

    with patch("pkg_resources.iter_entry_points") as mock_iter_entry_points:
        # Create mock plugins
        mock_plugin = Mock()
        mock_plugin.name = "nowcast_method"
        mock_plugin.load.return_value = "mock_module"

        # Set the return value of iter_entry_points to include the mock plugins
        mock_iter_entry_points.return_value = [mock_plugin]

        # Clear the _nowcast_methods dictionary before the test
        pysteps.nowcasts.interface._nowcast_methods.clear()

        # Call the function under test
        pysteps.nowcasts.interface.discover_nowcasts()

        # Print the call arguments for debugging
        print(
            "mock_iter_entry_points.call_args_list:",
            mock_iter_entry_points.call_args_list,
        )

        # Assert that the entry point was called
        mock_iter_entry_points.assert_called_with(
            group="pysteps.plugins.nowcasts", name=None
        )

        # Assert that the _nowcast_methods dictionary is updated correctly
        assert (
            mock_plugin.name in pysteps.nowcasts.interface._nowcast_methods
        ), "Expected 'nowcast_method' to be in _nowcast_methods, but it was not found."
        assert (
            pysteps.nowcasts.interface._nowcast_methods[mock_plugin.name]
            == mock_plugin.load
        ), "Expected the value of 'nowcast_method' to be 'mock_module'."
