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
_cascade_methods['fft'] = decomposition.decomposition_fft
_cascade_methods['gaussian'] = bandpass_filters.filter_gaussian
_cascade_methods['uniform'] = bandpass_filters.filter_uniform


def get_method(name):
    """
    Return a callable function for the bandpass filter or decomposition method
    corresponding to the given name.

    Filter methods:

    +-------------------+------------------------------------------------------+
    |     Name          |              Description                             |
    +===================+======================================================+
    |  gaussian         | implementation of a bandpass filter using Gaussian   |
    |                   | weights                                              |
    +-------------------+------------------------------------------------------+
    |  uniform          | implementation of a filter where all weights are set |
    |                   | to one                                               |
    +-------------------+------------------------------------------------------+

    Decomposition methods:

    +-------------------+------------------------------------------------------+
    |     Name          |              Description                             |
    +===================+======================================================+
    |  fft              | decomposition based on Fast Fourier Transform (FFT)  |
    |                   | and a bandpass filter                                |
    +-------------------+------------------------------------------------------+

    """

    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError("Only strings supported for the method's names.\n"
                        + "Available names:"
                        + str(list(_cascade_methods.keys()))) from None
    try:
        return _cascade_methods[name]
    except KeyError:
        raise ValueError("Unknown method {}\n".format(name)
                         + "The available methods are:"
                         + str(list(_cascade_methods.keys()))) from None
