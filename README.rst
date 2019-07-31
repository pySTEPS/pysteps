=====================================================================
pySTEPS - Python framework for short-term ensemble prediction systems
=====================================================================

.. start-badges

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - docs
      - |docs|
    * - tests
      - |travis| |requires| |codecov|
    * - package
      - |github| |conda| |pypi|
    * - license
      - |License|

.. |docs| image:: https://readthedocs.org/projects/pysteps/badge/?version=latest
    :alt: Documentation Status
    :target: https://pysteps.readthedocs.io/

.. |travis| image:: https://travis-ci.com/pySTEPS/pysteps.svg?branch=master
    :alt: Travis
    :target: https://travis-ci.com/pySTEPS/pysteps

.. |codecov| image:: https://codecov.io/gh/pySTEPS/pysteps/branch/master/graph/badge.svg
    :alt: Coverage
    :target: https://codecov.io/gh/pySTEPS/pysteps
    
.. |requires| image:: https://requires.io/github/pySTEPS/pysteps/requirements.svg?branch=master
     :target: https://requires.io/github/pySTEPS/pysteps/requirements/?branch=master
     :alt: Requirements Status

.. |github| image:: https://img.shields.io/github/release/pySTEPS/pysteps.svg
    :target: https://github.com/pySTEPS/pysteps/releases/latest
    :alt: Latest github release
    
.. |conda| image:: https://anaconda.org/conda-forge/pysteps/badges/version.svg   
    :target: https://anaconda.org/conda-forge/pysteps
    :alt: Anaconda Cloud
    
.. |pypi| image:: https://badge.fury.io/py/pysteps.svg
    :target: https://pypi.org/project/pysteps/
    :alt: Latest PyPI version
    
.. |license| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :alt: License
    :target: https://opensource.org/licenses/BSD-3-Clause

.. end-badges

What is pysteps?
================

Pysteps is an open-source and community-driven Python library for probabilistic precipitation nowcasting, i.e. short-term ensemble prediction systems.

The aim of pysteps is to serve two different needs. The first is to provide a modular and well-documented framework for researchers interested in developing new methods for nowcasting and stochastic space-time simulation of precipitation. The second aim is to offer a highly configurable and easily accessible platform for practitioners ranging from weather forecasters to hydrologists.

The pysteps library supports standard input/output file formats and implements several optical flow methods as well as advanced stochastic generators to produce ensemble nowcasts. In addition, it includes tools for visualizing and post-processing the nowcasts and methods for deterministic, probabilistic, and neighbourhood forecast verification.

Installation
============

To install pysteps please have a look at the `pysteps user guide`__.

__ https://pysteps.readthedocs.io/en/latest/user_guide/index.html

Use
===

You can have a look at the `gallery of examples`__ to get a better idea of how the library can be used.
 
__ https://pysteps.readthedocs.io/en/latest/auto_examples/index.html

For a more detailed description of the implemented functions, check the `pysteps reference page`__.

__ https://pysteps.readthedocs.io/en/latest/pysteps_reference/index.html

Example data
============

A set of example radar data is available in a separate repository: `pysteps-data`__. More information on how to download and install them are available here__.

__ https://github.com/pySTEPS/pysteps-data
__ https://pysteps.readthedocs.io/en/latest/user_guide/example_data.html#example-data

Contributions
=============

We welcome contributions, feedback, suggestions for developments and bug reports.

Feedback, suggestions for developments and bug reports can use the dedicated `Issues page`__.

__ https://github.com/pySTEPS/pysteps/issues

More information dedicated to developers is available in the `developer guide`__.

__ https://pysteps.readthedocs.io/en/latest/developer_guide/index.html

Reference publications
======================

Pulkkinen, S., D. Nerini, A. Perez Hortal, C. Velasco-Forero, U. Germann, A. Seed, and L. Foresti, 2019:  Pysteps:  an open-source Python library for probabilistic precipitation nowcasting (v1.0). *Geosci. Model Dev. Discuss.*, doi:10.5194/gmd-2019-94 **in review**. [source__]

__ https://www.geosci-model-dev-discuss.net/gmd-2019-94/

Pulkkinen, S., D. Nerini, A. Perez Hortal, C. Velasco-Forero, U. Germann, A. Seed, and
L. Foresti, 2019: pysteps - a Community-Driven Open-Source Library for Precipitation Nowcasting. *Poster presented at the 3rd European Nowcasting Conference, Madrid, ES*, doi: 10.13140/RG.2.2.31368.67840. [source__]

__ https://www.researchgate.net/publication/332781022_pysteps_-_a_Community-Driven_Open-Source_Library_for_Precipitation_Nowcasting
