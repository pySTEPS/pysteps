
from . import bandpass_filters
from . import decomposition

def get_method(name):
    """Return a callable function for the bandpass filter or decomposition method
    corresponding to the given name.\n\

    Filter methods:

    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  gaussian         | implementation of a bandpass filter using Gaussian     |
    |                   | weights                                                |
    +-------------------+--------------------------------------------------------+
    |  uniform          | implementation of a filter where all weights are set   |
    |                   | to one                                                 |
    +-------------------+--------------------------------------------------------+

    Decomposition methods:

    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  fft              | decomposition based on Fast Fourier Transform (FFT)    |
    |                   | and a bandpass filter                                  |
    +-------------------+--------------------------------------------------------+

    """
    if name.lower() == "fft":
        return decomposition.decomposition_fft
    elif name.lower() == "gaussian":
        return bandpass_filters.filter_gaussian
    elif name.lower() == "uniform":
        return bandpass_filters.filter_uniform
    else:
        raise ValueError("unknown method %s, the currently implemented methods are 'fft', 'gaussian' and 'uniform'" % name)
