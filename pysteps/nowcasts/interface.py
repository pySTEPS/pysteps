import numpy as np

"""The methods in the nowcasts module implement the following interface:

    forecast(R, V, num_timesteps, **kwargs)

where R (m,n) is the input precipitation field and V (2,m,n) is an array
containing  the x- and y-components of the m*n advection field. num_timesteps
is an integer specifying the number of time steps to forecast. The interface
accepts optional keyword arguments specific to the given method.

The output depends on the type of the method. For deterministic methods, the
output is a three-dimensional array of shape (num_timesteps,m,n) containing a
time series of nowcast precipitation fields. For stochastic methods that produce
an ensemble, the output is a four-dimensional array of shape
(num_ensemble_members,num_timesteps,m,n). The time step of the output is taken
from the inputs.

"""

def get_method(name):
    """Return a callable function for computing deterministic or ensemble
    precipitation nowcasts.\n\

    Implemented methods:

    +-------------------+-----------------------------------------------------+
    |     Name          |              Description                            |
    +===================+=====================================================+
    |  eulerian         | this approach keeps the last observation frozen     |
    |                   | (Eulerian persistence)                              |
    +-------------------+-----------------------------------------------------+
    |  lagrangian or    | this approach extrapolates the last observation     |
    |  extrapolation    | using the motion field (Lagrangian persistence)     |
    +-------------------+-----------------------------------------------------+
    |  steps            | implementation of the STEPS stochastic nowcasting   |
    |                   | method described in :cite:`Seed2003`,               |
    |                   | :cite:`BPS2006` and :cite:`SPN2013`                 |
    +-------------------+-----------------------------------------------------+

    steps produces stochastic nowcasts, and the other methods are deterministic.

    """
    if name.lower() in ["eulerian"]:
        def eulerian(R, V, num_timesteps, *args, **kwargs):
            return np.repeat(R[None, :, :,], num_timesteps, axis=0)
        return eulerian
    elif name.lower() in ["extrapolation", "lagrangian"]:
        from . import extrapolation
        return extrapolation.forecast
    elif name.lower() in ["steps"]:
        from . import steps
        return steps.forecast
    else:
        raise ValueError("unknown nowcasting method %s" % name)
