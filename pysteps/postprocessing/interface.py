# -*- coding: utf-8 -*-
"""
pysteps.postprocessing.interface
====================

Interface for the postprocessing module.

.. currentmodule:: pysteps.postprocessing.interface

.. autosummary::
    :toctree: ../generated/

    get_method
"""
import importlib

import pysteps.postprocessing
from pysteps.postprocessing import diagnostics, ensemblestats
from pprint import pprint

_diagnostics_methods = dict(
    diagnostics_example1=diagnostics.postprocessors_diagnostics_example1,
    diagnostics_example3=lambda x: [x, x],
)

_ensemblestats_methods = dict(
    ensemblestats_example1=ensemblestats.postprocessors_ensemblestats_example1,
    ensemblestats_example3=lambda x, y: [x, y],
)


def discover_postprocessors():
    """
    Search for installed postprocessing plugins in the entrypoint 'pysteps.plugins.postprocessors'

    The postprocessors found are added to the appropriate `_methods`
    dictionary in 'pysteps.postprocessing.interface' containing the available postprocessors.
    """

    # The pkg resources needs to be reloaded to detect new packages installed during
    # the execution of the python application. For example, when the plugins are
    # installed during the tests
    import pkg_resources

    importlib.reload(pkg_resources)

    for entry_point in pkg_resources.iter_entry_points(
        group="pysteps.plugins.postprocessors", name=None
    ):
        _postprocessors = entry_point.load()

        postprocessors_function_name = _postprocessors.__name__
        postprocessors_short_name = postprocessors_function_name.replace(
            "postprocessors_", ""
        )

        if postprocessors_short_name.startswith("diagnostics_"):
            diagnostics_short_name = postprocessors_short_name.replace(
                "diagnostics_", ""
            )
            if diagnostics_short_name not in _diagnostics_methods:
                _diagnostics_methods[diagnostics_short_name] = _postprocessors
            else:
                RuntimeWarning(
                    f"The diagnostics identifier '{diagnostics_short_name}' is already available in "
                    "'pysteps.postprocessing.interface_diagnostics_methods'.\n"
                    f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
                )

            if hasattr(diagnostics, postprocessors_short_name):
                RuntimeWarning(
                    f"The diagnostics function '{diagnostics_short_name}' is already an attribute"
                    "of 'pysteps.postprocessing.diagnostics'.\n"
                    f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
                )
            else:
                setattr(diagnostics, postprocessors_function_name, _postprocessors)

        elif postprocessors_short_name.startswith("ensemblestats_"):
            ensemblestats_short_name = postprocessors_short_name.replace(
                "ensemblestats_", ""
            )
            if ensemblestats_short_name not in _ensemblestats_methods:
                _ensemblestats_methods[ensemblestats_short_name] = _postprocessors
            else:
                RuntimeWarning(
                    f"The ensemblestats identifier '{ensemblestats_short_name}' is already available in "
                    "'pysteps.postprocessing.interface_ensemblestats_methods'.\n"
                    f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
                )

            if hasattr(ensemblestats, postprocessors_short_name):
                RuntimeWarning(
                    f"The ensemblestats function '{ensemblestats_short_name}' is already an attribute"
                    "of 'pysteps.postprocessing.diagnostics'.\n"
                    f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
                )
            else:
                setattr(ensemblestats, postprocessors_function_name, _postprocessors)


def postprocessors_info():
    """Print all the available postprocessors."""

    # diagnostics available in the 'postprocessing.diagnostics' module
    available_diagnostics = [
        attr
        for attr in dir(pysteps.postprocessing.diagnostics)
        if attr.startswith("postprocessors")
    ]

    print("\npostprocessors available in the pysteps.postprocessing.diagnostics module")
    pprint(available_diagnostics)

    # diagnostics postprocessors declared in the pysteps.postprocessing.get_method interface
    diagnostics_in_the_interface = [
        f for f in list(pysteps.postprocessing.interface._diagnostics_methods.keys())
    ]

    print("\ndiagnostics available in the pysteps.postprocessing.get_method interface")
    pprint(
        [
            (short_name, f.__name__)
            for short_name, f in pysteps.postprocessing.interface._diagnostics_methods.items()
        ]
    )

    # Let's use sets to find out if there are postprocessors present in the diagnostics module
    # but not declared in the interface, and vice versa.
    available_diagnostics = set(available_diagnostics)
    diagnostics_in_the_interface = set(diagnostics_in_the_interface)

    difference = available_diagnostics ^ diagnostics_in_the_interface
    if len(difference) > 0:
        print("\nIMPORTANT:")
        _diff = available_diagnostics - diagnostics_in_the_interface
        if len(_diff) > 0:
            print(
                "\nIMPORTANT:\nThe following postprocessors are available in pysteps.postprocessing.diagnostics "
                "module but not in the pysteps.postprocessing.get_method interface"
            )
            pprint(_diff)
        _diff = diagnostics_in_the_interface - available_diagnostics
        if len(_diff) > 0:
            print(
                "\nWARNING:\n"
                "The following postprocessors are available in the pysteps.postprocessing.get_method "
                "interface but not in the pysteps.postprocessing.diagnostics module"
            )
            pprint(_diff)

    # postprocessors available in the 'postprocessing.ensemblestats' module
    available_ensemblestats = [
        attr
        for attr in dir(pysteps.postprocessing.ensemblestats)
        if attr.startswith("postprocessors")
    ]

    print(
        "\npostprocessors available in the pysteps.postprocessing.ensemblestats module"
    )
    pprint(available_ensemblestats)

    # ensemblestats postprocessors declared in the pysteps.postprocessing.get_method interface
    ensemblestats_in_the_interface = [
        f for f in list(pysteps.postprocessing.interface._ensemblestats_methods.keys())
    ]

    print(
        "\npostprocessors available in the pysteps.postprocessing.get_method interface"
    )
    pprint(
        [
            (short_name, f.__name__)
            for short_name, f in pysteps.postprocessing.interface._ensemblestats_methods.items()
        ]
    )

    # Let's use sets to find out if there are postprocessors present in the ensemblestats module
    # but not declared in the interface, and vice versa.
    available_ensemblestats = set(available_ensemblestats)
    ensemblestats_in_the_interface = set(ensemblestats_in_the_interface)

    difference = available_ensemblestats ^ ensemblestats_in_the_interface
    if len(difference) > 0:
        print("\nIMPORTANT:")
        _diff = available_ensemblestats - ensemblestats_in_the_interface
        if len(_diff) > 0:
            print(
                "\nIMPORTANT:\nThe following postprocessors are available in pysteps.postprocessing.ensemblestats "
                "module but not in the pysteps.postprocessing.get_method interface"
            )
            pprint(_diff)
        _diff = ensemblestats_in_the_interface - available_ensemblestats
        if len(_diff) > 0:
            print(
                "\nWARNING:\n"
                "The following postprocessors are available in the pysteps.postprocessing.get_method "
                "interface but not in the pysteps.postprocessing.ensemblestats module"
            )
            pprint(_diff)

    available_postprocessors = available_diagnostics.union(available_ensemblestats)
    postprocessors_in_the_interface = ensemblestats_in_the_interface.union(
        diagnostics_in_the_interface
    )

    return available_postprocessors, postprocessors_in_the_interface


def get_method(name, method_type):
    """
    Return a callable function for the method corresponding to the given
    name.

    Parameters
    ----------
    name: str
        Name of the method. The available options are:\n

        diagnostics:

        .. tabularcolumns:: |p{2cm}|L|

        +---------------+-------------------------------------------------------+
        |     Name      |                   Description                         |
        +===============+=======================================================+
        |  Diagnostics  |           Example that returns a string               |
        |   Example1    |                                                       |
        +---------------+-------------------------------------------------------+
        |  Diagnostics  |           Example that returns an array               |
        |   Example3    |                                                       |
        +---------------+-------------------------------------------------------+

        ensemblestats:

        .. tabularcolumns:: |p{2cm}|L|

        +---------------+-------------------------------------------------------+
        |     Name      |                   Description                         |
        +===============+=======================================================+
        | EnsembleStats |           Example that returns a string               |
        |   Example1    |                                                       |
        +---------------+-------------------------------------------------------+
        | EnsembleStats |           Example that returns an array               |
        |   Example3    |                                                       |
        +---------------+-------------------------------------------------------+

    method_type: {'diagnostics', 'ensemblestats'}
            Type of the method (see tables above).

    """

    if isinstance(method_type, str):
        method_type = method_type.lower()
    else:
        raise TypeError(
            "Only strings supported for for the method_type"
            + " argument\n"
            + "The available types are: 'diagnostics', 'ensemblestats'"
        ) from None

    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "\nAvailable diagnostics names:"
            + str(list(_diagnostics_methods.keys()))
            + "\nAvailable ensemblestats names:"
            + str(list(_ensemblestats_methods.keys()))
        ) from None

    if method_type == "diagnostics":
        methods_dict = _diagnostics_methods
    elif method_type == "ensemblestats":
        methods_dict = _ensemblestats_methods
    else:
        raise ValueError(
            "Unknown method type {}\n".format(method_type)
            + "The available types are: 'diagnostics', 'ensemblestats'"
        ) from None

    try:
        return methods_dict[name]
    except KeyError:
        raise ValueError(
            "Unknown {} method {}\n".format(method_type, name)
            + "The available methods are:"
            + str(list(methods_dict.keys()))
        ) from None
