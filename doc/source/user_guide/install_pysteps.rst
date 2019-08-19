.. _install_pysteps:

Installing pysteps
==================

Dependencies
------------

The pysteps package needs the following dependencies

* `python >=3.6 <http://www.python.org/>`_
* `attrdict <https://pypi.org/project/attrdict/>`_
* `jsmin <https://pypi.org/project/jsmin/>`_
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

* `cartopy <https://scitools.org.uk/cartopy/docs/v0.16/>`_ or
  `basemap <https://matplotlib.org/basemap/>`_ (for geo-referenced
  visualization)
* `h5py <https://www.h5py.org/>`_ (for importing HDF5 data)
* `pywavelets <https://pywavelets.readthedocs.io/en/latest/>`_
  (for intensity-scale verification)



Anaconda install (recommended)
------------------------------

Conda is an open source package management system and environment management
system that runs on Windows, macOS and Linux.
Conda quickly installs, runs and updates packages and their dependencies.
It also allows to easily create, save, load and switch between different
environments on your local computer.

Since version 1.0, pySTEPS is available in conda-forge, a community-driven
package repository for anaconda.

There are two installation alternatives using anaconda: install pySTEPS in
pre-existing environment, or install it new environment.

New environment
~~~~~~~~~~~~~~~

In a terminal, to create a new conda environment and install pySTEPS, run::

    $ conda create -n pysteps
    $ source activate pysteps

This will create and activate the new python environment. The next step is to
add the conda-forge channel where the pySTEPS package is located::

    $ conda config --env --prepend channels conda-forge

Let's set this channel as the priority one::

    $ conda config --env --set channel_priority strict

The later step is not strictly necessary, but is recommended since
the conda-forge and the default Anaconda channels are not 100% compatible.

Finally, to install pySTEPS and all its dependencies run::

    $ conda install pysteps


Install from source
-------------------

The recommended way to install pysteps from source is using `pip`
to adhere to the [PEP517 standards](https://www.python.org/dev/peps/pep-0517/).
Using `pip` instead of `setup.py` guarantees that all the package dependencies
are properly handled during the installation process.


OSX users
~~~~~~~~~

pySTEPS uses Cython extensions that need to be compiled with multi-threading
support enabled. The default Apple Clang compiler does not support OpenMP,
so using the default compiler would have disabled multi-threading and you will
get the following error during the installation::

    clang: error: unsupported option '-fopenmp'
    error: command 'gcc' failed with exit status 1

To solve this issue, obtain the lastest gcc version with
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

Installation
~~~~~~~~~~~~

The latest pysteps version in the repository can be installed using pip by
simply running in a terminal::

    pip install git+https://github.com/pySTEPS/pysteps

Or, from a local copy of the repo (global installation)::

    git clone https://github.com/pySTEPS/pysteps
    cd pysteps
    pip install .

The above commands will install the latest version of the **master** branch,
which is constantly under development.

.. _development_mode_install:

Development mode
################

The latest version can also be installed in Development Mode, i.e.,
in such a way that the project appears to be installed,
but yet is still editable from the source tree::

    pip install -e <path to local pysteps repo>


Setting up the user-defined configuration file
----------------------------------------------

.. _JSON: https://en.wikipedia.org/wiki/JSON
.. _AttrDict: https://pypi.org/project/attrdict/

The pysteps package allows the users to customize the default settings
and configuration.
The configuration parameters used by default are loaded from a user-defined
JSON_ file and then stored in the **pysteps.rcparams** AttrDict_.

.. toctree::
    :maxdepth: 1

    Set-up the user-defined configuration file <set_pystepsrc>
    Example pystepsrc file <pystepsrc_example>







