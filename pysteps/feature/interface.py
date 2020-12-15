# -*- coding: utf-8 -*-
"""
pysteps.feature.interface
=========================

Interface for the feature detection module. It returns a callable function for
detecting features.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from pysteps.feature import blob
from pysteps.feature import tstorm
from pysteps.feature import shitomasi

_detection_methods = dict()
_detection_methods["blob"] = blob.detection
_detection_methods["tstorm"] = tstorm.detection
_detection_methods["shitomasi"] = shitomasi.detection


def get_method(name):
    """Return a callable function for computing detection.

    Description:
    Return a callable function for detecting features on input images .

    Implemented methods:

    +-----------------+-------------------------------------------------------+
    |     Name        |              Description                              |
    +=================+=======================================================+
    |  blob           | blob detection in scale space                         |
    +-----------------+-------------------------------------------------------+
    |  tstorm         | Thunderstorm cell detection                           |
    +-----------------+-------------------------------------------------------+
    |  shitomasi      | Shi-Tomasi corner detection                           |
    +-----------------+-------------------------------------------------------+
    """
    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available names:"
            + str(list(_detection_methods.keys()))
        ) from None

    try:
        return _detection_methods[name]
    except KeyError:
        raise ValueError(
            "Unknown detection method {}\n".format(name)
            + "The available methods are:"
            + str(list(_detection_methods.keys()))
        ) from None
