# -*- coding: utf-8 -*-

"""
Script to test the plugin support.

This script assumes that a package created with the default Pysteps plugin template
(and using the default values) is installed.

https://github.com/pySTEPS/cookiecutter-pysteps-plugin

"""

from pysteps import io

assert hasattr(io.importers, "import_abc_yyy")
assert hasattr(io.importers, "import_abc_zzz")

assert "abc_yyy" in io.interface._importer_methods
assert "abc_zzz" in io.interface._importer_methods

from pysteps.io.importers import import_abc_yyy
from pysteps.io.importers import import_abc_zzz

import_abc_yyy("filename")
import_abc_zzz("filename")
