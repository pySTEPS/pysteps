
from . import simple_advection
from . import steps

def get_method(name):
    """Return one callable function to produce deterministic or ensemble 
    precipitation nowcasts.\n\
    
    Methods for precipitation nowcasting:
    +-------------------+-------------------------------------------------------+
    |     Name          |              Description                              |
    +===================+=======================================================+
    |  extrapolation    |  this method is a simple advection forecast based     |
    |                   |  on Lagrangian persistence                            |
    +-------------------+-------------------------------------------------------+
    |                   | implementation of the STEPS stochastic nowcasting     |
    |  steps            | method as described in Seed (2003), Bowler et al      |
    |                   | (2006) and Seed et al (2013)                          |
    +-------------------+-------------------------------------------------------+

    """
    if name.lower() == "extrapolation":
        return simple_advection.forecast
    elif name.lower() == "steps":
        return steps.forecast
    else:
        raise ValueError("unknown nowcasting method %s" % name)
