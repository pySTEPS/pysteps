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
from pysteps.blending import steps

_blending_methods = dict()
_blending_methods["linear_blending"] = linear_blending.forecast
_blending_methods["steps"] = steps.forecast


def get_method(name):
    """
    Return a callable function for computing nowcasts blending into an NWP
    forecast.

    Implemented methods:

    +------------------+------------------------------------------------------+
    |     Name         |              Description                             |
    +==================+======================================================+
    | linear_blending  | the linear blending of a nowcast method with other   |
    |                  | data (e.g. NWP data).                                |
    +------------------+------------------------------------------------------+
    | steps            | the STEPS stochastic nowcasting blending method      |
    |                  | described in :cite:`Seed2003`, :cite:`BPS2006` and   |
    |                  | :cite:`SPN2013`. The blending weights approach       |
    |                  | currently follows :cite:`BPS2006`.                   |
    +------------------+------------------------------------------------------+
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
            f"Unknown blending method {name}."
            "The available methods are: "
            f"{*list(_blending_methods.keys()),}"
        ) from None
