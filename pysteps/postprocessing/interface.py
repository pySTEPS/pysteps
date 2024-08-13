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

from pkg_resources import iter_entry_points

import pysteps.postprocessing
from pysteps.postprocessing import diagnostics
from pprint import pprint

_diagnostics_methods = dict(
    example1=diagnostics.diagnostics_example1, example3=lambda x: [x, x]
)


def discover_diagnostics():
    """
    Search for installed diagnostics plugins in the entrypoint 'pysteps.plugins.diagnostics'

    The diagnostics found are added to the `pysteps.postprocessing.interface_diagnostics_methods`
    dictionary containing the available diagnostics.
    """

    # The pkg resources needs to be reloaded to detect new packages installed during
    # the execution of the python application. For example, when the plugins are
    # installed during the tests
    import pkg_resources

    importlib.reload(pkg_resources)

    for entry_point in pkg_resources.iter_entry_points(
        group="pysteps.plugins.diagnostics", name=None
    ):
        _diagnostics = entry_point.load()

        diagnostics_function_name = _diagnostics.__name__
        diagnostics_short_name = diagnostics_function_name.replace("diagnostics_", "")

        _diagnostics_kws = getattr(_diagnostics, "diagnostics_kws", dict())
        if diagnostics_short_name not in _diagnostics_methods:
            _diagnostics_methods[diagnostics_short_name] = _diagnostics
        else:
            RuntimeWarning(
                f"The diagnostics identifier '{diagnostics_short_name}' is already available in "
                "'pysteps.postprocessing.interface_diagnostics_methods'.\n"
                f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
            )

        if hasattr(diagnostics, diagnostics_function_name):
            RuntimeWarning(
                f"The diagnostics function '{diagnostics_function_name}' is already an attribute"
                "of 'pysteps.postprocessing.diagnostics'.\n"
                f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
            )
        else:
            setattr(diagnostics, diagnostics_function_name, _diagnostics)


def diagnostics_info():
    """Print all the available diagnostics."""

    # diagnostics available in the 'postprocessing.diagnostics' module
    available_diagnostics = [
        attr
        for attr in dir(pysteps.postprocessing.diagnostics)
        if attr.startswith("diagnostics")
    ]

    print("\ndiagnostics available in the pysteps.postprocessing.diagnostics module")
    pprint(available_diagnostics)

    # diagnostics declared in the pysteps.postprocessing.get_method interface
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

    # Let's use sets to find out if there are diagnostics present in the diagnostics module
    # but not declared in the interface, and vice versa.
    available_diagnostics = set(available_diagnostics)
    diagnostics_in_the_interface = set(diagnostics_in_the_interface)

    available_diagnostics = {s.split("_")[1] for s in available_diagnostics}

    difference = available_diagnostics ^ diagnostics_in_the_interface
    if len(difference) > 0:
        print("\nIMPORTANT:")
        _diff = available_diagnostics - diagnostics_in_the_interface
        if len(_diff) > 0:
            print(
                "\nIMPORTANT:\nThe following diagnostics are available in pysteps.postprocessing.diagnostics "
                "module but not in the pysteps.postprocessing.get_method interface"
            )
            pprint(_diff)
        _diff = diagnostics_in_the_interface - available_diagnostics
        if len(_diff) > 0:
            print(
                "\nWARNING:\n"
                "The following diagnostics are available in the pysteps.postprocessing.get_method "
                "interface but not in the pysteps.postprocessing.diagnostics module"
            )
            pprint(_diff)

    return available_diagnostics, diagnostics_in_the_interface


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

        +-------------+-------------------------------------------------------+
        |     Name    |              Description                              |
        +=============+=======================================================+

        Diagnostic diagnostics:

        .. tabularcolumns:: |p{2cm}|L|

        +-------------+-------------------------------------------------------+
        |     Name    |              Description                              |
        +=============+=======================================================+

    method_type: {'diagnostics', diagnostics_name}
            Type of the method (see tables above).

    """

    if isinstance(method_type, str):
        method_type = method_type.lower()
    else:
        raise TypeError(
            "Only strings supported for for the method_type"
            + " argument\n"
            + "The available types are: 'diagnostics'"
        ) from None

    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "\nAvailable diagnostics names:"
            + str(list(_diagnostics_methods.keys()))
        ) from None

    if method_type == "diagnostics":
        methods_dict = _diagnostics_methods
    else:
        raise ValueError(
            "Unknown method type {}\n".format(method_type)
            + "The available types are: 'diagnostics'"
        ) from None

    try:
        return methods_dict[name]
    except KeyError:
        raise ValueError(
            "Unknown {} method {}\n".format(method_type, name)
            + "The available methods are:"
            + str(list(methods_dict.keys()))
        ) from None
