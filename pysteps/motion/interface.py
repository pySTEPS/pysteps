import numpy as np

from pysteps.motion.darts import DARTS
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.motion.vet import vet

_methods = dict()
_methods['lk'] = dense_lucaskanade
_methods['lucaskanade'] = dense_lucaskanade
_methods['darts'] = DARTS
_methods['vet'] = vet
_methods[None] = lambda precip, *args, **kw: np.zeros((2,
                                                       precip.shape[1],
                                                       precip.shape[2]))


def get_method(name):
    """Return a callable function for the optical flow method corresponding to
    the given name. The available options are:\n\

    +--------------------------------------------------------------------------+
    | Python-based implementations                                             |
    +-------------------+------------------------------------------------------+
    |     Name          |              Description                             |
    +===================+======================================================+
    |  None             | Returns a zero motion field                          |
    +-------------------+------------------------------------------------------+
    |  lucaskanade      | OpenCV implementation of the Lucas-Kanade method     |
    |                   | with interpolated motion vectors for areas with no   |
    |                   | precipitation.                                       |
    +-------------------+------------------------------------------------------+
    |  darts            | Implementation of the DARTS method of Ruzanski et al.|
    +-------------------+------------------------------------------------------+
    |  vet              | Implementation of the VET method of                  |
    |                   | Laroche and Zawadzki (1995) and                      |
    |                   | Germann and Zawadzki (2002)                          |
    +-------------------+------------------------------------------------------+

    +--------------------------------------------------------------------------+
    | Methods implemented in C (these require separate compilation and linkage)|
    +-------------------+------------------------------------------------------+
    |     Name          |              Description                             |
    +===================+======================================================+
    |  brox             | implementation of the variational method of          |
    |                   | Brox et al. (2004) from IPOL                         |
    |                   |  (http://www.ipol.im/pub/art/2013/21)                |
    +-------------------+------------------------------------------------------+
    |  clg              | implementation of the Combined Local-Global (CLG)    |
    |                   | method of Bruhn et al., 2005 from IPOL               |
    |                   | (http://www.ipol.im/pub/art/2015/44)                 |
    +-------------------+------------------------------------------------------+

    """

    if isinstance(name, str):
        name = name.lower()

    if name in ["brox", "clg"]:
        raise NotImplementedError("Method %s not implemented" % name)
    else:
        try:
            motion_method = _methods[name]
            return motion_method
        except KeyError:
            raise ValueError("Unknown method {}\n".format(name)
                             + "The available methods are:"
                             + str(list(_methods.keys()))) from None
