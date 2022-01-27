# -*- coding: utf-8 -*-
"""
pysteps.io.interface
====================

Interface for the io module.

.. currentmodule:: pysteps.io.interface

.. autosummary::
    :toctree: ../generated/

    get_method
"""
import importlib

from pkg_resources import iter_entry_points

from pysteps import io
from pysteps.decorators import postprocess_import
from pysteps.io import importers, exporters
from pprint import pprint

_importer_methods = dict(
    bom_rf3=importers.import_bom_rf3,
    fmi_geotiff=importers.import_fmi_geotiff,
    fmi_pgm=importers.import_fmi_pgm,
    mch_gif=importers.import_mch_gif,
    mch_hdf5=importers.import_mch_hdf5,
    mch_metranet=importers.import_mch_metranet,
    mrms_grib=importers.import_mrms_grib,
    odim_hdf5=importers.import_odim_hdf5,
    opera_hdf5=importers.import_opera_hdf5,
    knmi_hdf5=importers.import_knmi_hdf5,
    saf_crri=importers.import_saf_crri,
)

_exporter_methods = dict(
    geotiff=exporters.initialize_forecast_exporter_geotiff,
    kineros=exporters.initialize_forecast_exporter_kineros,
    netcdf=exporters.initialize_forecast_exporter_netcdf,
)


def discover_importers():
    """
    Search for installed importers plugins in the entrypoint 'pysteps.plugins.importers'

    The importers found are added to the `pysteps.io.interface_importer_methods`
    dictionary containing the available importers.
    """

    # The pkg resources needs to be reload to detect new packages installed during
    # the execution of the python application. For example, when the plugins are
    # installed during the tests
    import pkg_resources

    importlib.reload(pkg_resources)

    for entry_point in pkg_resources.iter_entry_points(
        group="pysteps.plugins.importers", name=None
    ):
        _importer = entry_point.load()

        importer_function_name = _importer.__name__
        importer_short_name = importer_function_name.replace("import_", "")

        _postprocess_kws = getattr(_importer, "postprocess_kws", dict())
        _importer = postprocess_import(**_postprocess_kws)(_importer)
        if importer_short_name not in _importer_methods:
            _importer_methods[importer_short_name] = _importer
        else:
            RuntimeWarning(
                f"The importer identifier '{importer_short_name}' is already available in"
                "'pysteps.io.interface._importer_methods'.\n"
                f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
            )

        if hasattr(importers, importer_function_name):
            RuntimeWarning(
                f"The importer function '{importer_function_name}' is already an attribute"
                "of 'pysteps.io.importers`.\n"
                f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
            )
        else:
            setattr(importers, importer_function_name, _importer)


def importers_info():
    """Print all the available importers."""

    # Importers available in the `io.importers` module
    available_importers = [
        attr for attr in dir(io.importers) if attr.startswith("import_")
    ]

    print("\nImporters available in the pysteps.io.importers module")
    pprint(available_importers)

    # Importers declared in the pysteps.io.get_method interface
    importers_in_the_interface = [
        f.__name__ for f in io.interface._importer_methods.values()
    ]

    print("\nImporters available in the pysteps.io.get_method interface")
    pprint(
        [
            (short_name, f.__name__)
            for short_name, f in io.interface._importer_methods.items()
        ]
    )

    # Let's use sets to find out if there are importers present in the importer module
    # but not declared in the interface, and viceversa.
    available_importers = set(available_importers)
    importers_in_the_interface = set(importers_in_the_interface)

    difference = available_importers ^ importers_in_the_interface
    if len(difference) > 0:
        print("\nIMPORTANT:")
        _diff = available_importers - importers_in_the_interface
        if len(_diff) > 0:
            print(
                "\nIMPORTANT:\nThe following importers are available in pysteps.io.importers module "
                "but not in the pysteps.io.get_method interface"
            )
            pprint(_diff)
        _diff = importers_in_the_interface - available_importers
        if len(_diff) > 0:
            print(
                "\nWARNING:\n"
                "The following importers are available in the pysteps.io.get_method "
                "interface but not in the pysteps.io.importers module"
            )
            pprint(_diff)

    return available_importers, importers_in_the_interface


def get_method(name, method_type):
    """
    Return a callable function for the method corresponding to the given
    name.

    Parameters
    ----------
    name: str
        Name of the method. The available options are:\n

        Importers:

        .. tabularcolumns:: |p{2cm}|L|

        +--------------+------------------------------------------------------+
        |     Name     |              Description                             |
        +==============+======================================================+
        | bom_rf3      | NefCDF files used in the Boreau of Meterorology      |
        |              | archive containing precipitation intensity           |
        |              | composites.                                          |
        +--------------+------------------------------------------------------+
        | fmi_geotiff  | GeoTIFF files used in the Finnish Meteorological     |
        |              | Institute (FMI) archive, containing reflectivity     |
        |              | composites (dBZ).                                    |
        +--------------+------------------------------------------------------+
        | fmi_pgm      | PGM files used in the Finnish Meteorological         |
        |              | Institute (FMI) archive, containing reflectivity     |
        |              | composites (dBZ).                                    |
        +--------------+------------------------------------------------------+
        | knmi_hdf5    | HDF5 file format used by KNMI.                       |
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
        | odim_hdf5    | HDF5 file conforming to the ODIM specification.      |
        +--------------+------------------------------------------------------+
        | opera_hdf5   | Wrapper to "odim_hdf5" to maintain backward          |
        |              | compatibility with previous pysteps versions.        |
        +--------------+------------------------------------------------------+
        | saf_crri     | NetCDF SAF CRRI files containing convective rain     |
        |              | rate intensity and other                             |
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

    method_type: {'importer', 'exporter'}
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
