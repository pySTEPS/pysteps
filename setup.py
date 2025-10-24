# -*- coding: utf-8 -*-
"""
Setup script for pysteps package.

Note: All project metadata and dependencies are now defined in pyproject.toml
following PEP 621 standards. This setup.py is maintained primarily for building
Cython extensions.
"""

import sys

from setuptools import find_packages, setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    raise RuntimeError(
        "Cython required for running the package installation\n"
        + "Try installing it with:\n"
        + "$> pip install cython"
    )

try:
    import numpy
except ImportError:
    raise RuntimeError(
        "Numpy required for running the package installation\n"
        + "Try installing it with:\n"
        + "$> pip install numpy"
    )

# Define common arguments used to compile the extensions
common_link_args = ["-fopenmp"]
common_compile_args = ["-fopenmp", "-O3", "-ffast-math"]
common_include = [numpy.get_include()]

if sys.platform.startswith("darwin"):
    common_link_args.append("-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/9/")

extensions_data = {
    "pysteps.motion._proesmans": {"sources": ["pysteps/motion/_proesmans.pyx"]},
    "pysteps.motion._vet": {"sources": ["pysteps/motion/_vet.pyx"]},
}

extensions = []

for name, data in extensions_data.items():
    include = data.get("include", common_include)

    extra_compile_args = data.get("extra_compile_args", common_compile_args)

    extra_link_args = data.get("extra_link_args", common_link_args)

    pysteps_extension = Extension(
        name,
        sources=data["sources"],
        depends=data.get("depends", []),
        include_dirs=include,
        language=data.get("language", "c"),
        define_macros=data.get("macros", []),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    extensions.append(pysteps_extension)

external_modules = cythonize(extensions, force=True, language_level=3)

# All metadata and dependencies are now defined in pyproject.toml
# This setup() call is minimal and primarily for Cython extension building
setup(
    ext_modules=external_modules,
    packages=find_packages(),
    include_package_data=True,
)
