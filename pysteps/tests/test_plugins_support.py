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
from pysteps import io, postprocessing


def _check_installed_importer_plugin(import_func_name):
    # reload the pysteps module to detect the installed plugin
    io.discover_importers()
    print(io.importers_info())
    import_func_name = import_func_name.replace("importer_", "import_")
    assert hasattr(io.importers, import_func_name)
    func_name = import_func_name.replace("import_", "")
    assert func_name in io.interface._importer_methods
    importer = getattr(io.importers, import_func_name)
    importer("filename")


def _check_installed_diagnostic_plugin(diagnostic_func_name):
    # reload the pysteps module to detect the installed plugin
    postprocessing.discover_postprocessors()
    assert hasattr(postprocessing.diagnostics, diagnostic_func_name)
    assert diagnostic_func_name in postprocessing.interface._diagnostics_methods
    diagnostic = getattr(postprocessing.diagnostics, diagnostic_func_name)
    diagnostic("filename")


@contextmanager
def _create_and_install_plugin(project_name, plugin_type):
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"Installing plugin {project_name} providing a {plugin_type} module")
        cookiecutter(
            PLUGIN_TEMPLATE_URL,
            no_input=True,
            overwrite_if_exists=True,
            extra_context={
                "project_name": project_name,
                "plugin_type": plugin_type,
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
    with _create_and_install_plugin("pysteps-importer-institution-fun", "importer"):
        _check_installed_importer_plugin("importer_institution_fun")


def test_diagnostic_plugins():
    with _create_and_install_plugin("pysteps-diagnostic-fun", "diagnostic"):
        _check_installed_diagnostic_plugin("diagnostic_fun")
