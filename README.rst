=====================================================================
pySTEPS - Python framework for short-term ensemble prediction systems
=====================================================================

.. start-badges

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - docs
      - |latest| |colab| |gallery|
    * - status
      - |test| |docs| |codecov| |codacy| |black|
    * - package
      - |github| |conda| |pypi|
    * - community
      - |slack| |contributors| |downloads| |license|


.. |docs| image:: https://readthedocs.org/projects/pysteps/badge/?version=latest
    :alt: Documentation Status
    :target: https://pysteps.readthedocs.io/

.. |test| image:: https://github.com/pySTEPS/pysteps/workflows/Test%20Pysteps/badge.svg
    :alt: Test Pysteps
    :target: https://github.com/pySTEPS/pysteps/actions?query=workflow%3A"Test+Pysteps"

.. |black| image:: https://github.com/pySTEPS/pysteps/workflows/Check%20Black/badge.svg
    :alt: Check Black
    :target: https://github.com/pySTEPS/pysteps/actions?query=workflow%3A"Check+Black"

.. |codecov| image:: https://codecov.io/gh/pySTEPS/pysteps/branch/master/graph/badge.svg
    :alt: Coverage
    :target: https://codecov.io/gh/pySTEPS/pysteps

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

.. |slack| image:: https://pysteps-slackin.herokuapp.com/badge.svg
    :alt: Slack invitation page
    :target: https://pysteps-slackin.herokuapp.com/

.. |contributors| image:: https://img.shields.io/github/contributors/pySTEPS/pysteps
    :alt: GitHub contributors
    :target: https://github.com/pySTEPS/pysteps/graphs/contributors

.. |downloads| image:: https://img.shields.io/conda/dn/conda-forge/pysteps
    :alt: Conda downloads
    :target: https://anaconda.org/conda-forge/pysteps

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: My first nowcast
    :target: https://colab.research.google.com/github/pySTEPS/pysteps/blob/master/examples/my_first_nowcast.ipynb

.. |gallery| image:: https://img.shields.io/badge/example-gallery-blue.svg
    :alt: pysteps example gallery
    :target: https://pysteps.readthedocs.io/en/latest/auto_examples/index.html
    
.. |latest| image:: https://img.shields.io/badge/docs-latest-blue.svg
    :alt: pysteps documentation
    :target: https://pysteps.readthedocs.io/en/latest/
    
.. |codacy| image:: https://api.codacy.com/project/badge/Grade/6cff9e046c5341a4afebc0347362f8de
   :alt: Codacy Badge
   :target: https://app.codacy.com/gh/pySTEPS/pysteps?utm_source=github.com&utm_medium=referral&utm_content=pySTEPS/pysteps&utm_campaign=Badge_Grade

.. end-badges

What is pysteps?
================

Pysteps is an open-source and community-driven Python library for probabilistic precipitation nowcasting, i.e. short-term ensemble prediction systems.

The aim of pysteps is to serve two different needs. The first is to provide a modular and well-documented framework for researchers interested in developing new methods for nowcasting and stochastic space-time simulation of precipitation. The second aim is to offer a highly configurable and easily accessible platform for practitioners ranging from weather forecasters to hydrologists.

The pysteps library supports standard input/output file formats and implements several optical flow methods as well as advanced stochastic generators to produce ensemble nowcasts. In addition, it includes tools for visualizing and post-processing the nowcasts and methods for deterministic, probabilistic, and neighbourhood forecast verification.


Run your first nowcast
----------------------

Use pysteps to compute and plot an extrapolation nowcast in Google Colab with `this interactive notebook`__.

__ https://colab.research.google.com/github/pySTEPS/pysteps/blob/master/examples/my_first_nowcast.ipynb

Get in touch
============

You can get in touch with the pysteps community on our `pysteps slack`__. To get access to it, you need to ask for an invitation or you can use the automatic invitation page `here`__. This invite page can sometimes take a while to load so please be patient.

__ https://pysteps.slack.com/
__ https://pysteps-slackin.herokuapp.com/

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

Pulkkinen, S., D. Nerini, A. Perez Hortal, C. Velasco-Forero, U. Germann,
A. Seed, and L. Foresti, 2019:  Pysteps:  an open-source Python library for
probabilistic precipitation nowcasting (v1.0). *Geosci. Model Dev.*, **12 (10)**,
4185–4219, doi:10.5194/gmd-12-4185-2019. [source__]

__ https://doi.org/10.5194/gmd-12-4185-2019

Pulkkinen, S., D. Nerini, A. Perez Hortal, C. Velasco-Forero, U. Germann, A. Seed, and
L. Foresti, 2019: pysteps - a Community-Driven Open-Source Library for Precipitation Nowcasting. *Poster presented at the 3rd European Nowcasting Conference, Madrid, ES*, doi: 10.13140/RG.2.2.31368.67840. [source__]

__ https://www.researchgate.net/publication/332781022_pysteps_-_a_Community-Driven_Open-Source_Library_for_Precipitation_Nowcasting
