r"""
pysteps.nowcasts.interface
==========================

Interface for the nowcasts module. It returns a callable function for computing
nowcasts.

The v2 nowcasting interface takes an xarray dataset:

    ``forecast(dataset, timesteps, **keywords)``

The dataset contains precipitation, spatial coordinates, observation times,
projection metadata and, when supplied, velocity_x and velocity_y variables.
The global ``precip_var`` attribute identifies the precipitation variable.
The output dataset contains dimensions (time, y, x) for deterministic methods
or (ens_number, time, y, x) for ensemble methods. Method-specific forecast
functions document their input history and optional settings.

The ``parametric`` method requires exactly three complete, untransformed
rain-rate observations in mm/h on a regular grid with square pixels. Pixel
size and cadence are derived from coordinates; the time stepsize is in seconds.
It estimates motion if velocity variables are absent and returns rain rates in
mm/h. Its first release accepts integer timesteps only.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from functools import partial

from pysteps.nowcasts import (
    anvil,
    extrapolation,
    lagrangian_probability,
    linda,
    parametric,
    sprog,
    sseps,
    steps,
)

eulerian_persistence = partial(extrapolation.forecast, extrap_method="eulerian")

_nowcast_methods = dict()
_nowcast_methods["anvil"] = anvil.forecast
_nowcast_methods["eulerian"] = eulerian_persistence
_nowcast_methods["extrapolation"] = extrapolation.forecast
_nowcast_methods["lagrangian"] = extrapolation.forecast
_nowcast_methods["lagrangian_probability"] = lagrangian_probability.forecast
_nowcast_methods["linda"] = linda.forecast
_nowcast_methods["parametric"] = parametric.forecast
_nowcast_methods["probability"] = lagrangian_probability.forecast
_nowcast_methods["sprog"] = sprog.forecast
_nowcast_methods["sseps"] = sseps.forecast
_nowcast_methods["steps"] = steps.forecast


def get_method(name):
    r"""
    Return a callable function for computing nowcasts.

    Description:
    Return a callable function for computing deterministic or ensemble
    precipitation nowcasts.

    Implemented methods:

    +-----------------+-------------------------------------------------------+
    |     Name        |              Description                              |
    +=================+=======================================================+
    |  anvil          | the autoregressive nowcasting using VIL (ANVIL)       |
    |                 | nowcasting method developed in :cite:`PCLH2020`       |
    +-----------------+-------------------------------------------------------+
    |  eulerian       | this approach keeps the last observation frozen       |
    |                 | (Eulerian persistence)                                |
    +-----------------+-------------------------------------------------------+
    |  lagrangian or  | this approach extrapolates the last observation       |
    |  extrapolation  | using the motion field (Lagrangian persistence)       |
    +-----------------+-------------------------------------------------------+
    |  linda          | the LINDA method developed in :cite:`PCN2021`         |
    +-----------------+-------------------------------------------------------+
    |  parametric     | three rain-rate observations to a parametric ensemble |
    |                 | with internal parameter and motion estimation         |
    +-----------------+-------------------------------------------------------+
    |  lagrangian\_   | this approach computes local lagrangian probability   |
    |  probability    | forecasts of threshold exceedences                    |
    +-----------------+-------------------------------------------------------+
    |  sprog          | the S-PROG method described in :cite:`Seed2003`       |
    +-----------------+-------------------------------------------------------+
    |  steps          | the STEPS stochastic nowcasting method described in   |
    |                 | :cite:`Seed2003`, :cite:`BPS2006` and :cite:`SPN2013` |
    |                 |                                                       |
    +-----------------+-------------------------------------------------------+
    |  sseps          | short-space ensemble prediction system (SSEPS).       |
    |                 | Essentially, this is a localization of STEPS          |
    +-----------------+-------------------------------------------------------+
    """
    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available names:"
            + str(list(_nowcast_methods.keys()))
        ) from None

    try:
        return _nowcast_methods[name]
    except KeyError:
        raise ValueError(
            "Unknown nowcasting method {}\n".format(name)
            + "The available methods are:"
            + str(list(_nowcast_methods.keys()))
        ) from None
