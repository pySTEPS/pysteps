"""
pysteps.tracking.interface
===========================

Interface for the tracking module. It returns a callable function for tracking
features.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from pysteps.tracking import lucaskanade

_tracking_methods = dict()
_tracking_methods["lucaskanade"] = lucaskanade.track_features


def get_method(name):
    """Return a callable function for tracking features.

    Description:
    Return a callable function for tracking features on input images .

    Implemented methods:

    +-----------------+--------------------------------------------------------+
    |     Name        |              Description                               |
    +=================+========================================================+
    | lucaskanade     | Wrapper to the OpenCV implementation of the            |
    |                 | Lucas-Kanade tracking algorithm                        |
    +-----------------+--------------------------------------------------------+
    """
    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available names:"
            + str(list(_tracking_methods.keys()))
        ) from None

    try:
        return _tracking_methods[name]
    except KeyError:
        raise ValueError(
            "Unknown tracking method {}\n".format(name)
            + "The available methods are:"
            + str(list(_tracking_methods.keys()))
        ) from None
