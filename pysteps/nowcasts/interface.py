"""
pysteps.nowcasts.interface
==========================

Interface for the nowcasts module. It returns a callable function for computing
nowcasts.

The methods in the nowcasts module implement the following interface:

    ``forecast(precip, velocity, num_timesteps, **keywords)``

where precip is a (m,n) array with input precipitation field to be advected and
velocity is a (2,m,n) array containing  the x- and y-components of
the m x n advection field.
num_timesteps is an integer specifying the number of time steps to forecast.
The interface accepts optional keyword arguments specific to the given method.

The output depends on the type of the method.
For deterministic methods, the output is a three-dimensional array of shape
(num_timesteps,m,n) containing a time series of nowcast precipitation fields.
For stochastic methods that produce an ensemble, the output is a
four-dimensional array of shape (num_ensemble_members,num_timesteps,m,n).
The time step of the output is taken from the inputs.

.. autosummary::
    :toctree: ../generated/
    
    get_method
"""

from pysteps.extrapolation.interface import eulerian_persistence
from pysteps.nowcasts import sprog, steps, sseps, extrapolation

_nowcast_methods = dict()
_nowcast_methods["eulerian"] = eulerian_persistence
_nowcast_methods["lagrangian"] = extrapolation.forecast
_nowcast_methods["extrapolation"] = extrapolation.forecast
_nowcast_methods["sprog"] = sprog.forecast
_nowcast_methods["steps"] = steps.forecast
_nowcast_methods["sseps"] = sseps.forecast


def get_method(name):
    """Return a callable function for computing nowcasts.

    Description:
    Return a callable function for computing deterministic or ensemble
    precipitation nowcasts.

    Implemented methods:

    +-------------------+-------------------------------------------------------+
    |     Name          |              Description                              |
    +===================+=======================================================+
    |  eulerian         | this approach keeps the last observation frozen       |
    |                   | (Eulerian persistence)                                |
    +-------------------+-------------------------------------------------------+
    |  lagrangian or    | this approach extrapolates the last observation       |
    |  extrapolation    | using the motion field (Lagrangian persistence)       |
    +-------------------+-------------------------------------------------------+
    |  sprog            | the S-PROG method described in :cite:`Seed2003`       |
    +-------------------+-------------------------------------------------------+
    |  steps            | the STEPS stochastic nowcasting method described in   |
    |                   | :cite:`Seed2003`, :cite:`BPS2006` and :cite:`SPN2013` |
    |                   |                                                       |
    +-------------------+-------------------------------------------------------+
    |  sseps            | short-space ensemble prediction system (SSEPS).       |
    |                   | Essentially, this is a localization of STEPS.         |
    +-------------------+-------------------------------------------------------+

    steps and sseps produce stochastic nowcasts, and the other methods are
    deterministic.
    """
    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError("Only strings supported for the method's names.\n" +
                        "Available names:" +
                        str(list(_nowcast_methods.keys()))) from None

    try:
        return _nowcast_methods[name]
    except KeyError:
        raise ValueError("Unknown nowcasting method {}\n".format(name) +
                         "The available methods are:" +
                         str(list(_nowcast_methods.keys()))) from None
