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

from functools import partial

from pysteps.blending import linear_blending
from pysteps.blending import steps
from pysteps.blending import pca_ens_kalman_filter

_blending_methods = dict()
_blending_methods["linear_blending"] = linear_blending.forecast
_blending_methods["salient_blending"] = partial(linear_blending.forecast, saliency=True)
_blending_methods["steps"] = steps.forecast
_blending_methods["pca_enkf"] = pca_ens_kalman_filter.forecast


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
    | salient_blending | the salient blending of a nowcast method with other  |
    |                  | data (e.g. NWP data) described in :cite:`Hwang2015`. |
    |                  | The blending is based on intensities and forecast    |
    |                  | times. The blended product preserves pixel           |
    |                  | intensities with time if they are strong enough based|
    |                  | on their ranked salience.                            |
    +------------------+------------------------------------------------------+
    | steps            | the STEPS stochastic nowcasting blending method      |
    |                  | described in :cite:`Seed2003`, :cite:`BPS2006` and   |
    |                  | :cite:`SPN2013`. The blending weights approach       |
    |                  | currently follows :cite:`BPS2006`.                   |
    +------------------+------------------------------------------------------+
    | pca_enkf         | the reduced-space EnKF combination method described  |
    |                  | in :cite:`Nerini2019MWR`.                               |
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
