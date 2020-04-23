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

_importer_methods = dict(
    bom_rf3=importers.import_bom_rf3,
    fmi_geotiff=importers.import_fmi_geotiff,
    fmi_pgm=importers.import_fmi_pgm,
    mch_gif=importers.import_mch_gif,
    mch_hdf5=importers.import_mch_hdf5,
    mch_metranet=importers.import_mch_metranet,
    mrms_grib=importers.import_mrms_grib,
    opera_hdf5=importers.import_opera_hdf5,
    knmi_hdf5=importers.import_knmi_hdf5,
    saf_crri=importers.import_saf_crri,
)

_exporter_methods = dict(
    geotiff=exporters.initialize_forecast_exporter_geotiff,
    kineros=exporters.initialize_forecast_exporter_kineros,
    netcdf=exporters.initialize_forecast_exporter_netcdf,
)


def get_method(name, method_type):
    """Return a callable function for the method corresponding to the given
    name.

    Parameters
    ----------
    name : str

        Name of the method. The available options are:\n

        Importers:

        .. tabularcolumns:: |p{2cm}|L|

        +--------------+------------------------------------------------------+
        |     Name     |              Description                             |
        +==============+======================================================+
        | bom_rf3      |  NefCDF files used in the Boreau of Meterorology     |
        |              |  archive containing precipitation intensity          |
        |              |  composites.                                         |
        +--------------+------------------------------------------------------+
        | fmi_geotiff  |  GeoTIFF files used in the Finnish Meteorological    |
        |              |  Institute (FMI) archive, containing reflectivity    |
        |              |  composites (dBZ).                                   |
        +--------------+------------------------------------------------------+
        | fmi_pgm      |  PGM files used in the Finnish Meteorological        |
        |              |  Institute (FMI) archive, containing reflectivity    |
        |              |  composites (dBZ).                                   |
        +--------------+------------------------------------------------------+
        | knmi_hdf5    |  HDF5 file format used by KNMI.                      |
        +--------------+------------------------------------------------------+
        | mch_gif      | GIF files in the MeteoSwiss (MCH) archive containing |
        |              | precipitation composites.                            |
        +--------------+------------------------------------------------------+
        | mch_hdf5     | HDF5 file format used by MeteoSiss (MCH).            |
        +--------------+------------------------------------------------------+
        | mch_metranet | metranet files in the MeteoSwiss (MCH) archive       |
        |              | containing precipitation composites.                 |
        +--------------+------------------------------------------------------+
        | mrms_grib    | Grib2 files used by the NSSL's MRMS product          |
        +--------------+------------------------------------------------------+
        | opera_hdf5   | ODIM HDF5 file format used by Eumetnet/OPERA.        |
        +--------------+------------------------------------------------------+
        | saf_crri     |  NetCDF SAF CRRI files                               |
        |              |  containing convective rain rate intensity and other |
        +--------------+------------------------------------------------------+

        Exporters:

        .. tabularcolumns:: |p{2cm}|L|

        +-------------+-------------------------------------------------------+
        |     Name    |              Description                              |
        +=============+=======================================================+
        | geotiff     | Export as GeoTIFF files.                              |
        +-------------+-------------------------------------------------------+
        | kineros     | KINEROS2 Rainfall file as specified in                |
        |             | https://www.tucson.ars.ag.gov/kineros/.               |
        |             | Grid points are treated as individual rain gauges.    |
        |             | A separate file is produced for each ensemble member. |
        +-------------+-------------------------------------------------------+
        | netcdf      | NetCDF files conforming to the CF 1.7 specification.  |
        +-------------+-------------------------------------------------------+

    method_type : {'importer', 'exporter'}

        Type of the method (see tables above).

    """

    if isinstance(method_type, str):
        method_type = method_type.lower()
    else:
        raise TypeError(
            "Only strings supported for for the method_type"
            + " argument\n"
            + "The available types are: 'importer' and 'exporter'"
        ) from None

    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available importers names:"
            + str(list(_importer_methods.keys()))
            + "\nAvailable exporters names:"
            + str(list(_exporter_methods.keys()))
        ) from None

    if method_type == "importer":
        methods_dict = _importer_methods
    elif method_type == "exporter":
        methods_dict = _exporter_methods
    else:
        raise ValueError(
            "Unknown method type {}\n".format(name)
            + "The available types are: 'importer' and 'exporter'"
        ) from None

    try:
        return methods_dict[name]
    except KeyError:
        raise ValueError(
            "Unknown {} method {}\n".format(method_type, name)
            + "The available methods are:"
            + str(list(methods_dict.keys()))
        ) from None
