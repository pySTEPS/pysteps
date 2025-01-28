"""
pysteps.postprocessing.diagnostics
====================

Methods for applying diagnostics postprocessing.

The methods in this module implement the following interface::

    diagnostics_xxx(optional arguments)

where **xxx** is the name of the diagnostic to be applied.

Available Diagnostics Postprocessors
------------------------

.. autosummary::
    :toctree: ../generated/

"""


def diagnostics_example1(filename, **kwargs):
    return "Hello, I am an example diagnostics postprocessor."


def diagnostics_example2(filename, **kwargs):
    return [[42, 42], [42, 42]]
