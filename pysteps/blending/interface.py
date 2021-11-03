# -*- coding: utf-8 -*-
"""
pysteps.blending.interface
==========================
Interface for the blending module. It returns a callable function for computing
blended nowcasts with NWP models.
.. autosummary::
    :toctree: ../generated/
    get_method
"""

from pysteps.blending import linear_blending

_blending_methods = dict()
_blending_methods["linear_blending"] = linear_blending.forecast


def get_method(name):
    """Return a callable function for computing nowcasts.
    Description:
    Return a callable function for computing deterministic or ensemble
    precipitation nowcasts.
    Implemented methods:
    +-----------------+-------------------------------------------------------+
    |     Name        |              Description                              |
    +=================+=======================================================+
    +-----------------+-------------------------------------------------------+
    |  linear         | the linear blending of a nowcast method with other    |
    |  blending       | data (e.g. NWP data).                                 |
    +-----------------+-------------------------------------------------------+
    """
    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available names:"
            + str(list(_blending_methods.keys()))
        ) from None

    try:
        return _blending_methods[name]
    except KeyError:
        raise ValueError(
            "Unknown blending method {}\n".format(name)
            + "The available methods are:"
            + str(list(_blending_methods.keys()))
        ) from None