.. _install_pysteps:

Installing pysteps
==================

Dependencies
------------

The pysteps package needs the following dependencies

* `python >=3.11, <3.14 <http://www.python.org/>`_ (lower or higher versions may work but are not tested).
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
* `scikit-learn >=1.7 <https://scikit-learn.org/>`_ (for PCA-based blending methods)


**Important**: If you only want to use pysteps, you can continue reading below.
But, if you want to contribute to pysteps or edit the package, you need to install
pysteps in development mode: :ref:`Contributing to pysteps <contributor_guidelines>`.

Install with uv (recommended)
-----------------------------

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package and project manager
that runs on Windows, macOS, and Linux. It takes care of installing the right Python
version, creating a virtual environment, and installing all the dependencies.

First, install uv by following the
`uv installation instructions <https://docs.astral.sh/uv/getting-started/installation/>`_.
macOS users also need a compiler with OpenMP support, see :ref:`install_osx_users`.

Then, get a copy of pysteps and install it with all its dependencies::

    git clone https://github.com/pySTEPS/pysteps
    cd pysteps
    uv sync

This creates a virtual environment in the **.venv** folder inside the pysteps
directory. To also install the optional dependencies listed above, run instead::

    uv sync --extra optional

To use pysteps in your own uv project instead, add it as a dependency from your
project's folder::

    uv add git+https://github.com/pySTEPS/pysteps

Install from source with pip
----------------------------

Alternatively, pysteps can be installed from the source using ``pip``
to adhere to the `PEP517 standards <https://www.python.org/dev/peps/pep-0517/>`_.
Using ``pip`` instead of ``setup.py`` guarantees that all the package dependencies
are properly handled during the installation process.

.. _install_osx_users:

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

    brew install gcc@13

.. _Homebrew: https://brew.sh/

To make sure that the installer uses the homebrew's gcc, export the
following environmental variables in the terminal
(supposing that gcc version 13 was installed)::

    export CC=gcc-13
    export CXX=g++-13

First, check that the homebrew's gcc is detected::

    which gcc-13

This should point to the homebrew's gcc installation.

Under certain circumstances, Homebrew_ does not add the symbolic links for the
gcc executables under /usr/local/bin.
If that is the case, specify the CC and CCX variables using the full path to
the homebrew installation. For example::

    export CC=/usr/local/Cellar/gcc/13.2.0/bin/gcc-13
    export CXX=/usr/local/Cellar/gcc/13.2.0/bin/g++-13

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

From the pysteps folder (or your own uv project), run::

    uv run python -c "import pysteps"

``uv run`` runs a command inside the project's virtual environment. If you prefer
to activate the environment yourself, run ``source .venv/bin/activate``
(Linux/macOS) or ``.venv\Scripts\activate`` (Windows) and use ``python`` as usual.
