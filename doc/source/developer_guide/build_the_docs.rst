.. _build_the_docs:

=================
Building the docs
=================

The pysteps documentations is build using
`Sphinx <http://www.sphinx-doc.org/en/master/>`_,
a tool that makes it easy to create intelligent and beautiful documentation

The documentation is located in the **doc** folder in the pysteps repo.

Build
-----
To build the docs you need a few more dependencies related to Sphinx.
They are listed in the ``doc`` extra in **pyproject.toml** and can be installed
from the repo's root with::

    uv sync --extra doc

In addition to this requirements, to build the example gallery in the
documentation the example pysteps-data is needed. To download and install this
data see the installation instructions in the :ref:`example_data` section.

Once these requirements are met, to build the documentation, in the **doc**
folder run::

    uv run make html

This will build the documentation along with the example gallery.

The build documentation (html web page) will be available in
**doc/_build/html/**.
To correctly visualize the documentation, you need to set up and run a local
HTTP server. To do that, in the **doc/_build/html/** directory run::

    uv run python -m http.server

This will set up a local HTTP server on 0.0.0.0 port 8000.
To see the built documentation open the following url in the browser:
http://0.0.0.0:8000/
