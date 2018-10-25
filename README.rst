=======
pySTEPS
=======

The pySTEPS initiative is a community that develops and maintains an easy to 
use, modular, free and open source python framework for short-term ensemble 
prediction systems.

The focus is on probabilistic nowcasting of radar precipitation fields,
but pySTEPS is designed to allow a wider range of uses.



Installing pySTEPS
==================

Dependencies
------------

The pySTEPS package needs the following dependencies

* numpy_
* scipy_
* opencv_
* pillow_
* pyproj_
* matplotlib_ (for examples)

.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy.org/
.. _opencv: https://opencv.org/
.. _pillow: https://python-pillow.org/
.. _pyproj: https://github.com/jswhit/pyproj
.. _matplotlib: http://matplotlib.org/

Additionally, the following packages can be installed to provide better computational efficiency:

* dask_ and toolz_ (for code parallelisation)
* pyfftw_ (for faster FFT computations)

.. _dask: https://dask.org/
.. _toolz: https://github.com/pytoolz/toolz/
.. _pyfftw: https://hgomersall.github.io/pyFFTW/

We recommend that you create a conda environment using the available `environment.yml`_ file to install all the necessary dependencies::

    conda env create -f environment.yml
    
.. _environment.yml: \
     https://github.com/pySTEPS/pysteps/blob/master/environment.yml

Install from source
-------------------

The latest version can be installed manually by downloading the sources using::

    git clone https://github.com/pySTEPS/pysteps


To install using pip run::

    pip install ./pysteps

Or, to install it using setup.py run (global installation)::

    python setup.py install
    
For `user installation`_::

    python setup.py install --user

.. _user installation: \
    https://docs.python.org/2/install/#alternate-installation-the-user-scheme
    
If you want to install the package in a specific directory run::
    
    python setup.py install --prefix=/path/to/local/dir

IMPORTANT: All the dependencies need to be already installed! 
