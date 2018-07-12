
from . import importers

def get_method(name):
    """Return a callable function for the importer method corresponding to
    the given name. The available options are:\n\

    +-------------------+---------------------------------------------------------+
    |     Name          |              Description                                |
    +===================+=========================================================+
    |     bom_rf3       |  NetCDF files in the Bureau of Meterology (BoM) archive |
    |                   |  containing precipitation intensity composites          |
    +-------------------+---------------------------------------------------------+
    |     fmi_pgm       |  PGM files in the Finnish Meteorological Institute      |
    |                   |  (FMI) archive containing reflectivity composites (dBZ) |
    +-------------------+---------------------------------------------------------+
    |     mch_gif       |  GIF files in the MeteoSwiss archive containing         |
    |                   |  precipitation intensity composites (mm/h)              |
    +-------------------+---------------------------------------------------------+
    |     odim_hdf5     |  ODIM HDF5 file format used by Eumetnet/OPERA           |
    +-------------------+---------------------------------------------------------+
    """
    if name.lower() == "bom_rf3":
        return importers.import_bom
    elif name.lower() == "fmi_pgm":
        return importers.import_fmi_pgm
    if name.lower() == "mch_gif":
        return importers.import_mch_gif
    elif name.lower() == "odim_hdf5":
        return importers.import_odim_hdf5
    else:
        raise ValueError("unknown method %s" % name)
