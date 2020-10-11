.. _importers-plugins:

================================
Create your own importers plugin
================================

Since version 3.4, Pysteps allows the users to add new importers to pysteps by
installing external packages, called plugins, without modifying the pysteps
installation. These plugins need to follow a particular structure (described next) to
allow pysteps to discover and integrate the new importers to the pysteps interface
without any user intervention.

.. contents:: Table of Contents
    :local:
    :depth: 4

How the plugins work?
=====================

When the plugin is installed, it also advertises the new importer functions to other
packages (in our case, pysteps) using the python `entry points specification`_.
After the plugin is installed, when pysteps is imported, it discovers the available
importers advertised via the **entry points**.
These discovered importers are added as attributes to the io.importers module and
automatically registered to the io.get_method interface without any user intervention.
Note that the discovery of the importer plugins is made every time that pysteps is
imported without modifying the actual pysteps installation (i.e., the pysteps sources).
Also, since the importer plugins are external to the pysteps library, the plugins will
keep working if the pysteps version is updated, without requiring any user intervention.

_`entry points specification`:https://packaging.python.org/specifications/entry-points/


Create your own plugin
======================

Let's suppose that we want to add two new importers to pysteps for reading the radar
composites from the "Awesome Bureau of Composites", kindly abbreviated as "abc".
The composites provided by this institution are available in two different
formats: Netcdf and Grib2.

For that, let's create a python package  (a.k.a. the plugin) implementing the two
importers. For simplicity, we will only include the elements that are strictly needed
for the plugin to be installed correctly in the package.

Plugin project structure
------------------------

The simplest python package to implement an importer plugin may have the following
structure:

::

    pysteps-importer-abc        (project name)
    ├── pysteps_importer_abc    (package name)
    │  ├── importer_abc_xyz.py  (importer module)
    │  └── __init__.py          (Initialize the pysteps_importer_abc package)
    ├── setup.py                (Build and installation script)
    └── MANIFEST.in             (manifest template)

Project name
~~~~~~~~~~~~

::

    pysteps-importer-abc        (project name)

For the project name, our example used the following convention:
**pysteps-importer-<institution short name>**.
Note that this convention is not strictly needed and any name can be used.

Package name
~~~~~~~~~~~~

::

    pysteps-importer-abc
    └── pysteps_importer_abc    (package name)

This is the name of our package containing the new importers for pysteps.
The package name should not contain spaces, hyphens or uppercase letters.
For our example, the package name is **pysteps_importer_abc**.

\__init__.py
~~~~~~~~~~~~

::

    pysteps-importer-abc
        ├── pysteps_importer_abc
        └───── __init__.py

The __init__.py files are required inform python that a given directory contains a
python package. Also, this is the first file executed when the importer plugin (i.e.,
the package) is imported.

Importer module
~~~~~~~~~~~~~~~

::

    pysteps-importer-abc
        ├── pysteps_importer_abc
        └───── importer_abc_xyz.py  (importer module)

Inside the package folder (*pysteps_importer_abc*), we place the python module
(or modules) containing the importers' implementation.
Below, there is an example of an importer module that implements the skeleton of two
different importers (the "grib" and "netcdf" importer that we are using as an example):

.. literalinclude:: importers_module_example.py
   :language: python


setup.py
~~~~~~~~

::

    pysteps-importer-abc        (project name)
    └── setup.py                (Build and installation script)

The `setup.py` file contains all the definitions for building, distributing, and
installing the package. A commented example of a setup.py script used for the plugin
installation is shown next:

.. literalinclude:: setup.py
   :language: python


Manifest.in
~~~~~~~~~~~


If you don't supply an explicit list of files, the installation using `setup.py` will
include the minimal files needed for the package to run (the *.py files, for example).
The Manifest.in file contains the list of additional files and directories to be
included in your source distribution.

Next, we show an example of a Manifest file that includes a README and the LICENSE files
located in the project root. Lines starting with **#** indicate comments, and they are
ignored.

::

    # This file contains the additional files included in your plugin package

    include LICENSE

    include README.rst

    ###You can also add directories with data, tests, etc.
    # recursive-include dir_with_data

    ###Include the documentation directory, if any.
    # recursive-include doc

For more information about the manifest file, see
https://docs.python.org/3/distutils/sourcedist.html#specifying-the-files-to-distribute








