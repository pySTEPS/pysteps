# -*- coding: utf-8 -*-
"""
pysteps.combination.interface
==========================
Interface for the combination module. It returns a callable function for different implementations of the ensemble Kalman filter to combine nowcasts with NWP forecasts.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from pysteps.combination import masked_enkf

_enkf_methods = dict()
_enkf_methods["masked_enkf"] = masked_enkf.MaskedEnKF


def get_method(name):
    """
    Return a callable function for different implementations of the ensemble Kalman
    filter to combine Nowcast and NWP ensemble forecasts.

    Implemented methods:

    +------------------+------------------------------------------------------+
    |     Name         |              Description                             |
    +==================+======================================================+
    | masked_enkf      | The ensemble Kalman filter as it is utilized in      |
    |                  | :cite:`Nerini2019`.                                  |
    +------------------+------------------------------------------------------+
    """
    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available names:"
            + str(list(_enkf_methods.keys()))
        ) from None

    try:
        return _enkf_methods[name]
    except KeyError:
        raise ValueError(
            f"Unknown ensemble Kalman filter method {name}."
            "The available methods are: "
            f"{*list(_enkf_methods.keys()),}"
        ) from None
