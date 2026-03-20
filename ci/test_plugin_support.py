# -*- coding: utf-8 -*-

"""
Script to test the plugin support.

This script assumes that a package created with the default pysteps plugin template
(and using the default values) is installed.

https://github.com/pySTEPS/cookiecutter-pysteps-plugin

"""

from pysteps import io

print("Testing plugin support: ", end="")
assert hasattr(io.importers, "import_institution_name")

assert "institution_name" in io.interface._importer_methods

from pysteps.io.importers import import_institution_name

import_institution_name("filename")
print("PASSED")
