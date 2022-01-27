.. _testing_pysteps:

===============
Testing pysteps
===============

The pysteps distribution includes a small test suite for some of the
modules. To run the tests the `pytest <https://docs.pytest.org>`__
package is needed. To install it, in a terminal run::

   pip install pytest


Automatic testing
=================

The simplest way to run the pysteps' test suite is using tox and the tox-conda
plugin (conda needed).
To install these packages activate your conda development environment and run::

    conda install -c conda-forge tox tox-conda

Then, to run the tests, from the repo's root run::

    tox             # Run pytests
    tox -e install  # Test package installation
    tox -e black    # Test for black formatting warnings


Manual testing
==============


Example data
------------

The build-in tests require the pysteps example data installed.
See the installation instructions in the :ref:`example_data` section.

Test an installed package
-------------------------

After the package is installed, you can launch the test suite from any
directory by running::

   pytest --pyargs pysteps

Test from sources
-----------------

Before testing the package directly from the sources, we need to build
the extensions in-place. To do that, from the root pysteps folder run::

   python setup.py build_ext -i

Now, the package sources can be tested in-place using the **pytest**
command on the root of the pysteps source directory. E.g.::

   pytest -v --tb=line

