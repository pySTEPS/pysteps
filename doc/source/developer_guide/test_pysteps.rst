.. _testing_pysteps:

===============
Testing pysteps
===============

The pysteps distribution includes a small test suite for some of the
modules. The tests are run with `pytest <https://docs.pytest.org>`__,
which is installed together with the other development dependencies when you
set up the environment with uv (see :ref:`contributor_guidelines`)::

   uv sync --extra optional

Example data
------------

The build-in tests require the pysteps example data installed.
See the installation instructions in the :ref:`example_data` section.

Run the tests
-------------

From the root of the pysteps repository, run::

   uv run pytest --pyargs pysteps

``uv sync`` already compiles the Cython extensions, so no separate build step is needed.

