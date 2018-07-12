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

The pySTEPS package need the following dependencies

* numpy
* scipy
* opencv
* pillow
* pyproj
* matplolib (for examples)


Install from source
-------------------

The latest version can be installed manually by downloading the sources from
https://github.com/pySTEPS/pysteps

Then, for a **global installation** run::

    python setup.py install
    
For `user installation`_::

    python setup.py install --user

.. _user installation: \
    https://docs.python.org/2/install/#alternate-installation-the-user-scheme
    
If you want to put it somewhere different than your system files, you can do::
    
    python setup.py install --prefix=/path/to/local/dir

IMPORTANT: All the dependencies need to be already installed! 
