# -*- coding: utf-8 -*-

import sys

from setuptools import setup, find_packages
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

requirements = [
    "numpy",
    "jsmin",
    "scipy",
    "matplotlib",
    "jsonschema",
]

setup(
    name="pysteps",
    version="1.6.2",
    author="PySteps developers",
    packages=find_packages(),
    license="LICENSE",
    include_package_data=True,
    description="Python framework for short-term ensemble prediction systems",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    url="https://pysteps.github.io/",
    project_urls={
        "Source": "https://github.com/pySTEPS/pysteps",
        "Issues": "https://github.com/pySTEPS/pysteps/issues",
        "CI": "https://github.com/pySTEPS/pysteps/actions",
        "Changelog": "https://github.com/pySTEPS/pysteps/releases",
        "Documentation": "https://pysteps.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    ext_modules=external_modules,
    setup_requires=requirements,
    install_requires=requirements,
)
