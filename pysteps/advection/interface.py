"""The extrapolation methods in the advection module implement the following interface:

    extrapolate(R, V, num_timesteps, outval=np.nan, keyword arguments)

where R (m,n) is the input precipitation field to be advected and V (2,m,n) is
an array containing  the x- and y-components of the m*n advection field. num_timesteps
is an integer specifying the number of time steps to extrapolate.
The optional argument outval specifies the value for pixels advected from outside
the domain. Optional keyword arguments that are specific to a given extrapolation
method are passed as a dictionary.
        
The output of each method is an array R_e that includes the time series of extrapolated 
fields of shape (num_timesteps, m, n). 
"""

from . import semilagrangian

def get_method(name):
    """Return a callable function for the extrapolation method corresponding to 
    the given name. The available options are:\n\
    
    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  eulerian or None | Returns the same initial field for eulerian persistence|
    |                   | experiments.                                           |
    +-------------------+--------------------------------------------------------+
    |  semilagrangian   | implementation of the semi-Lagrangian method of        |
    |                   | Germann et al. (2002)                                  |
    +-------------------+--------------------------------------------------------+
    
    """
    if name is None or name.lower() == "eulerian":
        def eulerian(R, V, num_timesteps, *args, **kwargs):
            return np.repeat(R[None, :, :,], num_timesteps, axis=0)
        return eulerian
    elif name.lower() == "semilagrangian":
        return semilagrangian.extrapolate
    else:
        raise ValueError("unknown method %s, the only currently implemented method is 'semilagrangian'" % name)
