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
