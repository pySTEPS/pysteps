"""
pysteps.io.interface
====================

Interface for the io module.

.. currentmodule:: pysteps.io.interface

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from pysteps.io import importers, exporters

_importer_methods = dict()
_importer_methods['bom_rf3'] = importers.import_bom_rf3
_importer_methods['fmi_pgm'] = importers.import_fmi_pgm
_importer_methods['mch_gif'] = importers.import_mch_gif
_importer_methods['mch_hdf5'] = importers.import_mch_hdf5
_importer_methods['mch_metranet'] = importers.import_mch_metranet
_importer_methods['odim_hdf5'] = importers.import_odim_hdf5
_importer_methods['knmi_hdf5'] = importers.import_knmi_hdf5

_exporter_methods = dict()
_exporter_methods['netcdf'] = exporters.initialize_forecast_exporter_netcdf
_exporter_methods['kineros'] = exporters.initialize_forecast_exporter_kineros


def get_method(name, method_type):
    """Return a callable function for the method corresponding to the given 
    name.

    Parameters
    ----------
    name : str
        Name of the method. The available options are:\n

        Importers:

        .. tabularcolumns:: |p{2cm}|L|

        +--------------+-------------------------------------------------------+
        |     Name     |              Description                              |
        +==============+=======================================================+
        | bom_rf3      |  NefCDF files used in the Boreau of Meterorology      |
        |              |  archive containing precipitation intensity           |
        |              |  composites.                                          |
        +--------------+-------------------------------------------------------+
        | fmi_pgm      |  PGM files used in the Finnish Meteorological         |
        |              |  Institute (FMI) archive, containing reflectivity     |
        |              |  composites (dBZ).                                    |
        +--------------+-------------------------------------------------------+
        | mch_gif      | GIF files in the MeteoSwiss (MCH) archive containing  |
        |              | precipitation composites.                             |
        +--------------+-------------------------------------------------------+
        | mch_hdf5     | HDF5 file format used by MeteoSiss (MCH).             |
        +--------------+-------------------------------------------------------+
        | mch_metranet | metranet files in the MeteoSwiss (MCH) archive        |
        |              | containing precipitation composites.                  |
        +--------------+-------------------------------------------------------+
        | odim_hdf5    | ODIM HDF5 file format used by Eumetnet/OPERA.         |
        +--------------+-------------------------------------------------------+
        | knmi_hdf5    |  HDF5 file format used by KNMI.                       |
        +--------------+-------------------------------------------------------+

        Exporters:
        
        .. tabularcolumns:: |p{2cm}|L|

        +-------------+--------------------------------------------------------+
        |     Name    |              Description                               |
        +=============+========================================================+
        | kineros     | KINEROS2 Rainfall file as specified in                 |
        |             | https://www.tucson.ars.ag.gov/kineros/.                |
        |             | Grid points are treated as individual rain gauges.     |
        |             | A separate file is produced for each ensemble member.  |
        +-------------+--------------------------------------------------------+
        | netcdf      | NetCDF files conforming to the CF 1.7 specification.   |
        +-------------+--------------------------------------------------------+

    method_type : str
        Type of the method. The available options are 'importer' and 'exporter'.

    """

    if isinstance(method_type, str):
        method_type = method_type.lower()
    else:
        raise TypeError("Only strings supported for for the method_type"
                        + " argument\n"
                        + "The available types are: 'importer' and 'exporter'"
                        ) from None

    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError("Only strings supported for the method's names.\n"
                        + "Available importers names:"
                        + str(list(_importer_methods.keys()))
                        + "\nAvailable exporters names:"
                        + str(list(_exporter_methods.keys()))) from None

    if method_type == "importer":
        methods_dict = _importer_methods
    elif method_type == "exporter":
        methods_dict = _exporter_methods
    else:
        raise ValueError("Unknown method type {}\n".format(name)
                         + "The available types are: 'importer' and 'exporter'"
                         ) from None

    try:
        return methods_dict[name]
    except KeyError:
        raise ValueError("Unknown {} method {}\n".format(method_type, name)
                         + "The available methods are:"
                         + str(list(methods_dict.keys()))) from None
