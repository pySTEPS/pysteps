.. _importer-plugins:

===========================
Create your importer plugin
===========================

Since version 1.4, pysteps allows the users to add new importers by installing external
packages, called plugins, without modifying the pysteps installation. These plugins need
to follow a particular structure to allow pysteps to discover and integrate the new
importers to the pysteps interface without any user intervention.

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


Create your plugin
==================

There are two ways of creating a plugin. The first one is building the importers plugin
from scratch. However, this can be a daunting task if you are creating your first plugin.
To facilitate the creating of new plugins, we provide a `Cookiecutter`_ template, in a
separate project, that creates a template project to be used as a starting point to build
the plugin.

The template for the pysteps plugins is maintained as a separate project at
`cookiecutter-pysteps-plugin <https://github.com/pySTEPS/cookiecutter-pysteps-plugin>`_.
For detailed instruction on how to create a plugin, `check the template's documentation`_.

.. _`check the template's documentation`: https://cookiecutter-pysteps-plugin.readthedocs.io/en/latest

.. _Cookiecutter: https://cookiecutter.readthedocs.io
