.. _install_pysteps:

Installing pysteps
==================

Dependencies
------------

The pysteps package needs the following dependencies

* `python >=3.8, <3.11 <http://www.python.org/>`_ (lower or higher versions may work but are not tested).
* `jsonschema <https://pypi.org/project/jsonschema/>`_
* `matplotlib <http://matplotlib.org/>`_
* `netCDF4 <https://pypi.org/project/netCDF4/>`_
* `numpy <http://www.numpy.org/>`_
* `opencv <https://opencv.org/>`_
* `pillow <https://python-pillow.org/>`_
* `pyproj <https://jswhit.github.io/pyproj/>`_
* `scipy <https://www.scipy.org/>`_


Additionally, the following packages can be installed for better computational
efficiency:

* `dask <https://dask.org/>`_ and
  `toolz <https://github.com/pytoolz/toolz/>`_ (for code parallelization)
* `pyfftw <https://hgomersall.github.io/pyFFTW/>`_ (for faster FFT computation)


Other optional dependencies include:

* `cartopy >=0.18 <https://scitools.org.uk/cartopy/docs/latest/>`_ (for geo-referenced
  visualization)
* `h5py <https://www.h5py.org/>`_ (for importing HDF5 data)
* `pygrib <https://jswhit.github.io/pygrib/docs/index.html>`_ (for importing MRMS data)
* `gdal <https://gdal.org/>`_ (for importing GeoTIFF data)
* `pywavelets <https://pywavelets.readthedocs.io/en/latest/>`_
  (for intensity-scale verification)
* `pandas <https://pandas.pydata.org/>`_ and
  `scikit-image >=0.19 <https://scikit-image.org/>`_ (for advanced feature detection methods)
* `rasterio <https://rasterio.readthedocs.io/en/latest/>`_ (for the reprojection module)


**Important**: If you only want to use pysteps, you can continue reading below.
But, if you want to contribute to pysteps or edit the package, you need to install
pysteps in development mode: :ref:`Contributing to pysteps <contributor_guidelines>`.

Install with conda/mamba (recommended)
--------------------------------------

`Conda <https://docs.conda.io/>`_ is an open-source package management system and environment
management system that runs on Windows, macOS, and Linux.
`Mamba <https://mamba.readthedocs.io/>`_ is a drop-in replacement for conda offering
better performances and more reliable environment
solutions. Mamba quickly installs, runs, and updates packages and their dependencies.
It also allows you to easily create, save, load, or switch between different
environments on your local computer.

Since version 1.0, pysteps is available on `conda-forge <https://conda-forge.org/>`_,
a community-driven package repository for conda packages.

To install pysteps with mamba in a new environment, run in a terminal::

    mamba create -n pysteps python=3.10
    mamba activate pysteps

This will create and activate the new python environment called 'pysteps' using python 3.10.
The next step is to add the conda-forge channel where the pysteps package is located::

    conda config --env --prepend channels conda-forge

Let's set this channel as the priority one::

    conda config --env --set channel_priority strict

The latter step is not strictly necessary but is recommended since
the conda-forge and the default conda channels are not 100% compatible.

Finally, to install pysteps and all its dependencies run::

    mamba install pysteps

Install pysteps on Apple Silicon Macs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On conda-forge, pysteps is currently compiled for Mac computers with Intel processors (osx-64).
However, thanks to `Rosetta 2 <https://support.apple.com/en-us/HT211861>`_ it is
possible to install the same package on a Mac computers with an Apple Silicon processor
(arm-64).

First, make sure that Rosetta 2 is installed::

    softwareupdate --install-rosetta

Use mamba to create a new environment called 'pysteps' for intel packages with python 3.10::

    CONDA_SUBDIR=osx-64 mamba create -n pysteps python=3.10
    mamba activate pysteps

Make sure that conda/mamba commands in this environment use intel packages::

    conda config --env --set subdir osx-64

Verify that the correct platform is being used::

    python -c "import platform;print(platform.machine())"  # Should print "x86_64"

Finally, run the same pysteps install instructions as given above::

    conda config --env --prepend channels conda-forge
    conda config --env --set channel_priority strict
    mamba install pysteps

We can now verify that pysteps loads correctly::

    python -c "import pysteps"

Note that the first time that pysteps is imported will typically take longer, as Rosetta 2
needs to translate the binary code for the Apple Silicon processor.

Install from source
-------------------

The recommended way to install pysteps from the source is using ``pip``
to adhere to the `PEP517 standards <https://www.python.org/dev/peps/pep-0517/>`_.
Using ``pip`` instead of ``setup.py`` guarantees that all the package dependencies
are properly handled during the installation process.

OSX users: gcc compiler
~~~~~~~~~~~~~~~~~~~~~~~

pySTEPS uses Cython extensions that need to be compiled with multi-threading
support enabled. The default Apple Clang compiler does not support OpenMP.
Hence, using the default compiler would have disabled multi-threading and may raise
the following error during the installation::

    clang: error: unsupported option '-fopenmp'
    error: command 'gcc' failed with exit status 1

To solve this issue, obtain the latest gcc version with
Homebrew_ that has multi-threading enabled::

    brew install gcc

.. _Homebrew: https://brew.sh/

To make sure that the installer uses the homebrew's gcc, export the
following environmental variables in the terminal
(supposing that gcc version 8 was installed)::

    export CC=gcc-8
    export CXX=g++-8

First, check that the homebrew's gcc is detected::

    which gcc-8

This should point to the homebrew's gcc installation.

Under certain circumstances, Homebrew_ does not add the symbolic links for the
gcc executables under /usr/local/bin.
If that is the case, specify the CC and CCX variables using the full path to
the homebrew installation. For example::

    export CC=/usr/local/Cellar/gcc/8.3.0/bin/gcc-8
    export CXX=/usr/local/Cellar/gcc/8.3.0/bin/g++-8

Then, you can continue with the normal installation procedure described next.

Installation using pip
~~~~~~~~~~~~~~~~~~~~~~

The latest pysteps version in the repository can be installed using pip by
simply running in a terminal::

    pip install git+https://github.com/pySTEPS/pysteps

Or, from a local copy of the repo::

    git clone https://github.com/pySTEPS/pysteps
    cd pysteps
    pip install .

The above commands install the latest version of the **master** branch,
which is continuously under development.

.. warning::
    If you are installing pysteps from the sources using pip, the Python interpreter must be launched outside of the pysteps root directory.
    Importing pysteps from a working directory that contains the pysteps source code will raise a ``ModuleNotFoundError``. 
    This error is caused by the root pysteps folder being recognized as the pysteps package, also known as 
    `the double import trap <http://python-notes.curiousefficiency.org/en/latest/python_concepts/import_traps.html#the-double-import-trap>`_.

Setting up the user-defined configuration file
----------------------------------------------

The pysteps package allows the users to customize the default settings
and configuration.
The configuration parameters used by default are loaded from a user-defined
`JSON <https://en.wikipedia.org/wiki/JSON>`_ file and then stored in the **pysteps.rcparams**, a dictionary-like object
that can be accessed as attributes or as items.

.. toctree::
    :maxdepth: 1

    Set-up the user-defined configuration file <set_pystepsrc>
    Example pystepsrc file <pystepsrc_example>

.. _import_pysteps:

Final test: import pysteps in Python
------------------------------------

Activate the pysteps environment::

    conda activate pysteps

Launch Python and import pysteps::

    python
    >>> import pysteps
