
def get_method(name):
    """Return a callable function for the optical flow method corresponding to
    the given name. The available options are:\n\

    +----------------------------------------------------------------------------+
    | Python-based implementations                                               |
    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  lucaskanade      | OpenCV implementation of the Lucas-Kanade method       |
    |                   | with interpolated motion vectors for areas with no     |
    |                   | precipitation.                                         |
    +-------------------+--------------------------------------------------------+
    |     darts         | Implementation of the DARTS method of Ruzanski et al.  |
    +-------------------+--------------------------------------------------------+

    +----------------------------------------------------------------------------+
    | Methods implemented in C (these require separate compilation and linkage)  |
    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  brox             | implementation of the variational method of Brox et al.|
    |                   | (2004) from IPOL (http://www.ipol.im/pub/art/2013/21)  |
    +-------------------+--------------------------------------------------------+
    |  clg              | implementation of the Combined Local-Global (CLG)      |
    |                   | method of Bruhn et al., 2005 from IPOL                 |
    |                   | (http://www.ipol.im/pub/art/2015/44)                   |
    +-------------------+--------------------------------------------------------+
    """
    if name.lower() == "lucaskanade" or name.lower() == "lk":
        from .lucaskanade import dense_lucaskanade 
        return dense_lucaskanade
    elif name.lower() == "darts":
        from .darts import DARTS
        return DARTS
    elif name.lower() in ["brox", "clg"]:
        raise NotImplementedError("method %s is not implemented" % name)
    else:
        raise ValueError("unknown method %s, the only implemented method is 'lucaskanade'" % name)
