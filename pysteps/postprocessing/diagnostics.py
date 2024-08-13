"""
pysteps.postprocessing.diagnostics
====================

Methods for diagnostic postprocessing.

The methods in this module implement the following interface::

    diagnostic_xyz(optional arguments)

where **xyz** is the name of the diagnostic postprocessing to be applied.

Postprocessor standardizations can be specified here if there is a desired input and output format that all should
adhere to.

Available Postprocessors
------------------------

.. autosummary::
    :toctree: ../generated/

"""


def diagnostics_example1(filename, **kwargs):
    return "Hello, I am an example postprocessor."


def diagnostics_example2(filename, **kwargs):
    return 42
