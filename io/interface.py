
from . import importers

def get_method(name):
    """Return a callable function for the importer method corresponding to
    the given name. The available options are:\n\

    +----------------------------------------------------------------------------+
    | ...                                                                        |
    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |                   |                                                        |
    +-------------------+--------------------------------------------------------+
    |                   |                                                        |
    +-------------------+--------------------------------------------------------+
    """
    if name.lower() == "aqc":
        return importers.import_aqc
    elif name.lower() == "bom":
        return importers.import_bom
    elif name.lower() == "fmi_pgm":
        return importers.import_fmi_pgm
    elif name.lower() == "odim_hdf5":
        return importers.import_odim_hdf5
    else:
        raise ValueError("unknown method %s" % name)
