import numpy as np

"""The methods in the extrapolation module implement the following interface:

    extrapolate(R, V, num_timesteps, outval=np.nan, keyword arguments)

where R (m,n) is the input precipitation field to be advected and V (2,m,n) is
an array containing  the x- and y-components of the m*n advection field. num_timesteps
is an integer specifying the number of time steps to extrapolate.
The optional argument outval specifies the value for pixels advected from outside
the domain. Optional keyword arguments that are specific to a given extrapolation
method are passed as a dictionary.

The output of each method is an array R_e that includes the time series of extrapolated
fields of shape (num_timesteps, m, n)."""

def get_method(name):
    """Return a callable function for the extrapolation method corresponding to
    the given name. The available options are:\n\

    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  None             | returns None                                           |
    +-------------------+--------------------------------------------------------+
    |  eulerian         | this methods does not apply any advection to the input |
    |                   | precipitation field (Eulerian persistence)             |
    +-------------------+--------------------------------------------------------+
    |  semilagrangian   | implementation of the semi-Lagrangian method of        |
    |                   | Germann et al. (2002)                                  |
    +-------------------+--------------------------------------------------------+

    """
    if name is None:
        def donothing(R, V, num_timesteps, *args, **kwargs):
            return None
        return donothing
    elif name.lower() in ["eulerian"]:
        def eulerian(R, V, num_timesteps, *args, **kwargs):
            return_displacement = kwargs.get("return_displacement", False)
            R_e = np.repeat(R[None, :, :,], num_timesteps, axis=0)
            if not return_displacement:
                return R_e
            else:
                return R_e, np.zeros((2, R.shape[0], R.shape[1]))
        return eulerian
    elif name.lower() in ["semilagrangian"]:
        from . import semilagrangian
        return semilagrangian.extrapolate
    else:
        raise ValueError("unknown method %s, the only currently implemented method is 'semilagrangian'" % name)
