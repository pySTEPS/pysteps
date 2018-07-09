
from . import semilagrangian

def get_method(name):
    """Return a callable function for the extrapolation method corresponding to 
    the given name. The available options are:\n\
    
    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  semilagrangian   | implementation of the semi-Lagrangian method of        |
    |                   | Germann et al. (2002)                                  |
    +-------------------+--------------------------------------------------------+
    """
    if name == "semilagrangian":
        return semilagrangian.extrapolate
    else:
        raise ValueError("unknown method %s, the only currently implemented method is 'semilagrangian'" % name)
