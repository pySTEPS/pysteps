.. _importer-plugins:

===============================
Create your own importer plugin
===============================

Since version 1.4, pysteps allows the users to add new importers by installing external
packages, called plugins, without modifying the pysteps installation.
These plugins need to follow a particular structure (described next) to allow pysteps
to discover and integrate the new importers to the pysteps interface without any user
intervention.

.. contents:: Table of Contents
    :local:
    :depth: 3

How do the plugins work?
========================

When the plugin is installed, it advertises the new importers to other packages (in our
case, pysteps) using the python `entry points specification`_.
These new importers are automatically discovered every time that the pysteps library is
imported. The discovered importers are added as attributes to the io.importers module
and also registered to the io.get_method interface without any user intervention.
In addition, since the installation of the plugins does not modify the actual pysteps
installation (i.e., the pysteps sources), the pysteps library can be updated without
reinstalling the plugin.

.. _`entry points specification`: https://packaging.python.org/specifications/entry-points/


Create your own plugin
======================

There are two ways of creating your own plugin.
The first one involves building the importers plugin package from scratch.
An easier alternative is using a `Cookiecutter`_ template that easily create the skeleton
for the new importer plugin.
Although the cookiecutter template is recommended,
in the next section we describe step-by-step how to create an importers plugin to
explain in detail all the elements needed for the plugin to work correctly.

After the reader is familiar with the plugins elements, the easier alternative
using the cookiecutter template is presented in the
`Cookiecutter pysteps-plugin template`_ section.

.. _Cookiecutter: https://cookiecutter.readthedocs.io

Plugin project structure
------------------------

Let's suppose that we want to add two new importers to pysteps for reading the radar
composites from the "Awesome Bureau of Composites", kindly abbreviated as "abc".
The composites provided by this institution are available in two different
formats: Netcdf and Grib2. The details of each format are not important for the rest of
this description. Just remember the names of the two formats.

Without further ado, let's create a python package  (a.k.a. the plugin) implementing the
two importers. For simplicity, we will only include the elements that are strictly
needed for the plugin to be installed and to work correctly.

The minimal python package to implement an importer plugin have the following
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

The __init__.py files are required to inform python that a given directory contains a
python package. Also, this is the first file executed when the importer plugin (i.e.,
the package) is imported.

Importer module
~~~~~~~~~~~~~~~

::

    pysteps-importer-abc
        ├── pysteps_importer_abc
        └───── importer_abc_xyz.py  (importer module)

Inside the package folder (*pysteps_importer_abc*), we place the python module
(or modules) containing the actual implementation of our new importers.
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
include the minimal files needed for the package to run (the \*.py files, for example).
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


Cookiecutter pysteps-plugin template
------------------------------------

`Cookiecutter`_ is a command-line utility to creates python packages projects from
templates, called "cookiecutters".
To facilitate the creation of an imported plugin for pysteps, the following cookiecutter
template is available at
https://github.com/pySTEPS/cookiecutter-pysteps-plugin
.

The first step is to install the latest version of `Cookiecutter`_ using,
for example::

    pip install -U cookiecutter

After cookiecutter is installed, a Pysteps-plugin project can be created by simply
running the following command and answering to the input prompts:

    cookiecutter https://github.com/pySTEPS/cookiecutter-pysteps-plugin

When the above command is run, you are asked to enter the following values
(one at a time):

::

    $> cookiecutter https://github.com/pySTEPS/cookiecutter-pysteps-plugin

    full_name [Your name]:
    email [your@email.com]:
    project_name [pysteps-importer-abc]:
    project_slug [pysteps_importer_abc]:
    project_short_description [Pysteps plugin adding the abc-xyz importers.]:
    importer_name [importer_abc]:
    version [0.1.0]:
    Select open_source_license:
    1 - MIT license
    2 - BSD license
    3 - ISC license
    4 - Apache Software License 2.0
    5 - GNU General Public License v3
    6 - Not open source
    Choose from 1, 2, 3, 4, 5, 6 [1]:

The text inside the brackets indicates the default values to be used if no input is
provided by the user (just pressing the "enter" key). Using the default values may be
useful for quick tests. However, if you want to implement your custom importers, it is
recommended to provide accurate information on each entry.

After all the requested values are provided, the following project is created in the
same directory where the command was run (assuming the default input values):

::

    pysteps-importer-abc        (project name)
    ├── docs                    (documentation directory)
    │     ├── conf.py
    │     ├── index.rst
    │     ├── installation.rst
    │     ├── make.bat
    │     ├── Makefile
    │     ├── readme.rst
    │     └── requirements.txt
    ├── LICENSE
    ├── MANIFEST.in
    ├── pysteps_importer_abc    (package name)
    │     ├── importer_abc.py   (importer_name)
    │     └── __init__.py
    ├── README.rst
    ├── requirements_dev.txt    (package requirements file)
    ├── setup.cfg               (setup configuration file)
    ├── setup.py
    ├── tests                   (tests directory)
    │     ├── __init__.py
    │     └── test_pysteps_importer_abc.py
    └── tox.ini                 (tox automation)

These files created by the cookiecutter provide all the elements to build your own
plugin. The files attempt to be self explanatory to aid with the development of the
plugin. Note that the following elements are optional:

::

    pysteps-importer-abc        (project name)
    ├── docs                    (documentation directory)
    ├── LICENSE
    ├── README.rst
    ├── requirements_dev.txt
    ├── tests                   (tests directory)
    └── tox.ini                 (tox automation)

In particular, the tox.ini file can be used to automatize some tasks, like testing the
plugin, build the documentation, etc.
For additional information about tox, see https://tox.readthedocs.io/en/latest/


Get in touch
============

If you have questions about the plugin implementation, you can get in touch with the
pysteps community on our `pysteps slack`__.
To get access to it, you need to ask for an invitation or you can use the automatic
invitation page `here`__. This invite page can sometimes take a while to load so please
be patient.

__ https://pysteps.slack.com/
__ https://pysteps-slackin.herokuapp.com/

