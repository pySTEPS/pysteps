# -*- coding: utf-8 -*-

"""
Script to test the plugin support.

This script assumes that a package created with the default pysteps plugin template
(and using the default values) is installed.

https://github.com/pySTEPS/cookiecutter-pysteps-plugin

"""

from pysteps import io

print("Testing plugin support: ", end="")
assert hasattr(io.importers, "import_abc_xxx")
assert hasattr(io.importers, "import_abc_yyy")

assert "abc_xxx" in io.interface._importer_methods
assert "abc_yyy" in io.interface._importer_methods

from pysteps.io.importers import import_abc_xxx, import_abc_yyy

import_abc_xxx("filename")
import_abc_yyy("filename")

print("PASSED")
