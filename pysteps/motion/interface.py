# -*- coding: utf-8 -*-
"""
pysteps.motion.interface
========================

Interface for the motion module. It returns a callable optical flow routine for
computing the motion field.

The methods in the motion module implement the following interface:

    ``motion_method(precip, **keywords)``

where precip is a (T,m,n) array containing a sequence of T two-dimensional input
images of shape (m,n). The first dimension represents the images time dimension
and the value of T depends on the type of the method.

The output is a three-dimensional array (2,m,n) containing the dense x- and
y-components of the motion field in units of pixels / timestep as given by the
input array R.

.. autosummary::
    :toctree: ../generated/

    get_method
"""
import numpy as np

from pysteps.motion.constant import constant
from pysteps.motion.darts import DARTS
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.motion.proesmans import proesmans
from pysteps.motion.vet import vet

_methods = dict()
_methods["constant"] = constant
_methods["lk"] = dense_lucaskanade
_methods["lucaskanade"] = dense_lucaskanade
_methods["darts"] = DARTS
_methods["proesmans"] = proesmans
_methods["vet"] = vet
_methods[None] = lambda precip, *args, **kw: np.zeros(
    (2, precip.shape[1], precip.shape[2])
)


def get_method(name):
    """Return a callable function for the optical flow method corresponding to
    the given name. The available options are:\n

    +--------------------------------------------------------------------------+
    | Python-based implementations                                             |
    +-------------------+------------------------------------------------------+
    |     Name          |              Description                             |
    +===================+======================================================+
    |  None             | returns a zero motion field                          |
    +-------------------+------------------------------------------------------+
    |  constant         | constant advection field estimated by maximizing the |
    |                   | correlation between two images                       |
    +-------------------+------------------------------------------------------+
    |  darts            | implementation of the DARTS method of Ruzanski et    |
    |                   | al. (2011)                                           |
    +-------------------+------------------------------------------------------+
    |  lucaskanade      | OpenCV implementation of the Lucas-Kanade method     |
    |                   | with interpolated motion vectors for areas with no   |
    |                   | precipitation                                        |
    +-------------------+------------------------------------------------------+
    |  proesmans        | the anisotropic diffusion method of Proesmans et     |
    |                   | al. (1994)                                           |
    +-------------------+------------------------------------------------------+
    |  vet              | implementation of the VET method of                  |
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
    |                   | (http://www.ipol.im/pub/art/2013/21)                 |
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
            raise ValueError(
                "Unknown method {}\n".format(name)
                + "The available methods are:"
                + str(list(_methods.keys()))
            ) from None
