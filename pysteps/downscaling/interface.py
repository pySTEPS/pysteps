"""
pysteps.downscaling.interface
=============================

Interface for the downscaling module. It returns a callable function for computing
downscaling.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from pysteps.downscaling import rainfarm

_downscale_methods = dict()
_downscale_methods["rainfarm"] = rainfarm.downscale


def get_method(name):
    """Return a callable function for computing downscaling.

    Description:
    Return a callable function for computing deterministic or ensemble
    precipitation downscaling.

    Implemented methods:

    +-----------------+-------------------------------------------------------+
    |     Name        |              Description                              |
    +=================+=======================================================+
    |  rainfarm       | the rainfall downscaling by a filtered autoregressive |
    |                 | model (RainFARM) method developed in                  |
    |                 | :cite:`Rebora2006`                                    |
    +-----------------+-------------------------------------------------------+
    """
    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available names:"
            + str(list(_downscale_methods.keys()))
        ) from None

    try:
        return _downscale_methods[name]
    except KeyError:
        raise ValueError(
            "Unknown downscaling method {}\n".format(name)
            + "The available methods are:"
            + str(list(_downscale_methods.keys()))
        ) from None
