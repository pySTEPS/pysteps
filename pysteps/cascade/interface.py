# -*- coding: utf-8 -*-
"""
pysteps.cascade.interface
=========================

Interface for the cascade module.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from pysteps.cascade import decomposition, bandpass_filters

_cascade_methods = dict()
_cascade_methods["fft"] = (decomposition.decomposition_fft, decomposition.recompose_fft)
_cascade_methods["gaussian"] = bandpass_filters.filter_gaussian
_cascade_methods["uniform"] = bandpass_filters.filter_uniform


def get_method(name):
    """
    Return a callable function for the bandpass filter or cascade decomposition
    method corresponding to the given name. For the latter, two functions are
    returned: the first is for the decomposing and the second is for recomposing
    the cascade.

    Filter methods:

    +-------------------+------------------------------------------------------+
    |     Name          |              Description                             |
    +===================+======================================================+
    |  gaussian         | implementation of bandpass filter using Gaussian     |
    |                   | weights                                              |
    +-------------------+------------------------------------------------------+
    |  uniform          | implementation of a filter where all weights are set |
    |                   | to one                                               |
    +-------------------+------------------------------------------------------+

    Decomposition/recomposition methods:

    +-------------------+------------------------------------------------------+
    |     Name          |              Description                             |
    +===================+======================================================+
    |  fft              | decomposition into multiple spatial scales based on  |
    |                   | the fast Fourier Transform (FFT) and a set of        |
    |                   | bandpass filters                                     |
    +-------------------+------------------------------------------------------+

    """

    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available names:"
            + str(list(_cascade_methods.keys()))
        ) from None
    try:
        return _cascade_methods[name]
    except KeyError:
        raise ValueError(
            "Unknown method {}\n".format(name)
            + "The available methods are:"
            + str(list(_cascade_methods.keys()))
        ) from None
