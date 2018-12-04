=======
pySTEPS
=======

The pySTEPS initiative is a community that develops and maintains an easy to 
use, modular, free and open source python framework for short-term ensemble 
prediction systems.

The focus is on probabilistic nowcasting of radar precipitation fields,
but pySTEPS is designed to allow a wider range of uses.



Installing pysteps
==================

Dependencies
------------

The pysteps package needs the following dependencies

* python>=3.6
* attrdict_
* jsmin_
* jsonschema_
* matplotlib_
* netCDF4_
* numpy_
* opencv-python_
* pillow_
* pyproj_
* scipy_

.. _attrdict : https://pypi.org/project/attrdict/
.. _jsmin : https://pypi.org/project/jsmin/
.. _jsonschema : https://pypi.org/project/jsonschema/
.. _matplotlib: http://matplotlib.org/
.. _netCDF4: https://pypi.org/project/netCDF4/
.. _numpy: http://www.numpy.org/
.. _opencv-python: https://opencv.org/
.. _pillow: https://python-pillow.org/
.. _pyproj: https://jswhit.github.io/pyproj/
.. _scipy: https://www.scipy.org/

Additionally, the following packages can be installed to provide better computational efficiency:

* dask_ and toolz_ (for code parallelisation)
* pyfftw_ (for faster FFT computations)

.. _dask: https://dask.org/
.. _toolz: https://github.com/pytoolz/toolz/
.. _pyfftw: https://hgomersall.github.io/pyFFTW/

Other optional packages include:

* cartopy_ or basemap_ (for georeferenced visualization) 
* h5py_ (for importing HDF5 data)

.. _basemap: https://matplotlib.org/basemap/
.. _cartopy: https://scitools.org.uk/cartopy/docs/v0.16/
.. _h5py: https://www.h5py.org/

We recommend that you create a conda environment using the available
`environment.yml`_ file to install the most important dependencies::

    conda env create -f environment.yml
    conda activate pysteps
    
.. _environment.yml: \
     https://github.com/pySTEPS/pysteps/blob/master/environment.yml

This will allow running most of the basic functionalities in pysteps.

Install from source
-------------------

**IMPORTANT**: installing from sources requires numpy to be installed
beforehand.

The installer need numpy to compile the Cython extensions.
If numpy is not installed you can running in a terminal::

    pip install numpy

The lastest pysteps version in the repository can be installed using pip by
simply running in a terminal::

    pip install git+https://github.com/pySTEPS/pysteps

Or, to install it using setup.py run (global installation)::

    git clone https://github.com/pySTEPS/pysteps
    cd pysteps
    python setup.py install
    
For `user installation`_::

    python setup.py install --user

.. _user installation: \
    https://docs.python.org/2/install/#alternate-installation-the-user-scheme
    
If you want to install the package in a specific directory run::

    python setup.py install --prefix=/path/to/local/dir


Non anaconda users or minimal anaconda environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The installation using **setup.py** will try to to install the minimum
dependencies needed to run the program correctly.
If you are not using the suggested conda environment (defined in
environment.yml) or you are working with a minimal python distribution,
you may get the following error during the installation::

    ModuleNotFoundError: No module named 'Cython'

This means that Cython is not installed, which is needed to build some of the
dependencies of pysteps.

For non anaconda users, you can install Cython using::

    pip install Cython

Anaconda users can install Cython using::

    conda install cython


Setting up the user-defined configuration file
----------------------------------------------

The pysteps package allows the users to customize the default settings
and configuration.
The configuration parameters used by default are stored in
pysteps.rcparams AttrDict_, which are loaded from a pystepsrc JSON_ file
located in the system.
The configuration parameters can be accessed as attributes or as items
in a dictionary. For e.g., to retrieve the default parameters
the following ways are equivalent::

    import pysteps

    # Retrieve the colorscale for plots
    colorscale = pysteps.rcparams['plot']['colorscale']
    colorscale = pysteps.rcparams.plot.colorscale

    # Retrieve the the root directory of the fmi data
    pysteps.rcparams['data_sources']['fmi']['root_path']
    pysteps.rcparams.data_sources.fmi.root_path

    # -----------------------------------------------------------------
    # A less wordy alternative
    # -----------------------------------------------------------------
    from pysteps import rcparams
    colorscale = rcparams['plot']['colorscale']
    colorscale = rcparams.plot.colorscale

    fmi_root_path = rcparams['data_sources']['fmi']['root_path']
    fmi_root_path = rcparams.data_sources.fmi.root_path

When the pysteps package imported, it looks for **pystepsrc** file in the
following order:

- $PWD/pystepsrc : Looks for the file in the current directory
- $PYSTEPSRC : If the system variable $PYSTEPSRC is defined and it
  points to a file, it is used.
- $PYSTEPSRC/pystepsrc : If $PYSTEPSRC points to a directory, it looks for the
  pystepsrc file inside that directory.
- $HOME/.pysteps/pystepsrc (unix and Mac OS X) : If the system variable $HOME is defined, it looks
  for the configuration file in this path.
- $USERPROFILE/pysteps/pystepsrc (windows only): It looks for the configuration file
  in the pysteps directory located user's home directory.
- Lastly, it looks inside the library in pysteps/pystepsrc for a
  system-defined copy.

.. _JSON: https://en.wikipedia.org/wiki/JSON
.. _AttrDict: https://pypi.org/project/attrdict/


The suggested way to setup the configuration files, is by editing a copy
of the default **pystepsrc** file that is distributed with the package
and place that copy inside the user home folder.


Linux and OSX users
~~~~~~~~~~~~~~~~~~~

For Linux and OSX users, the recommended way to customize the pysteps
configuration is place the pystepsrc parameters file in the users home folder
${HOME} in the following path: **${HOME}/.pysteps/pystepsrc**

This are the steps to setup up the configuration file in that directory:

1. Create the directory if it does not exist. Type in a terminal::

    $> mkdir -p ${HOME}/.pysteps

1. Find the location of the library's pystepsrc file used at the moment.
When we import pysteps in a python interpreter,
the configuration file loaded is shown::

    import pysteps
    "Pysteps configuration file found at: /path/to/pysteps/library/pystepsrc"

1.Copy the library's default rc file to that directory. In a terminal type::

    $> cp /path/to/pysteps/library/pystepsrc ${HOME}/.pysteps/pystepsrc

1. Edit the file with the text editor of your preference
1. Check that the location of the library's pystepsrc file used at the moment.::

     import pysteps
     "Pysteps configuration file found at: /home/user_name/.pysteps/pystepsrc"


Windows
~~~~~~~

For windows users, the recommended way to customize the pysteps
configuration is place the pystepsrc parameters file in the users folder
(defined in the %USERPROFILE% environment variable) in the following path:
**%USERPROFILE%/pysteps/pystepsrc**

This are the steps to setup up the configuration file in that directory:

1. Create the directory if it does not exist. Type in a terminal::

    $> mkdir -p %USERPROFILE%/pysteps

1. Find the location of the library's pystepsrc file used at the moment. When
the pystep is imported, the configuration file loaded is shown::

    import pysteps
    "Pysteps configuration file found at: /path/to/pysteps/library/pystepsrc"

1.Copy the library's default rc file to that directory. In a terminal type::

    $> cp /path/to/pysteps/library/pystepsrc %USERPROFILE%/pysteps/pystepsrc

1. Edit the file with the text editor of your preference
1. Check that the location of the library's pystepsrc file used at the moment::

     import pysteps
     "Pysteps configuration file found at: /home/user_name/.pysteps/pystepsrc"

