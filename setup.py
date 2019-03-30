# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools.extension import Extension
import sys

try:
    import numpy
except ImportError:
    raise RuntimeError(
        "Numpy required to pior running the package installation\n" +
        "Try installing it with:\n" +
        "$> pip install numpy")

extra_link_args = ['-fopenmp']

if sys.platform.startswith("darwin"):
    extra_link_args.append("-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/8/")

_vet_extension_arguments = dict(extra_compile_args=["-fopenmp"],
                                include_dirs=[numpy.get_include()],
                                language="c",
                                extra_link_args=extra_link_args,
                                )

try:
    from Cython.Build.Dependencies import cythonize

    _vet_lib_extension = Extension(str("pysteps.motion._vet"),
                                   sources=[str("pysteps/motion/_vet.pyx")],
                                   **_vet_extension_arguments)

    external_modules = cythonize([_vet_lib_extension])

except ImportError:
    _vet_lib_extension = Extension(str(str("pysteps.motion._vet")),
                                   sources=[str("pysteps/motion/_vet.c")],
                                   **_vet_extension_arguments)
    external_modules = [_vet_lib_extension]

requirements = ["numpy",
                "attrdict", "jsmin", "scipy", "matplotlib",
                "jsonschema"]

setup(
    name="pysteps",
    version="1.0.0",
    author="PySteps developers",
    packages=find_packages(),
    license="LICENSE",
    include_package_data=True,
    description="Python framework for short-term ensemble prediction systems",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    url="https://pysteps.github.io/",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    ext_modules=external_modules,
    setup_requires=requirements,
    install_requires=requirements
)
