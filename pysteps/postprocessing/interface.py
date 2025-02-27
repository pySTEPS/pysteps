# -*- coding: utf-8 -*-
"""
pysteps.postprocessing.interface
====================

Interface for the postprocessing module.

Support postprocessing types:
    - ensmeblestats
    - diagnostics

.. currentmodule:: pysteps.postprocessing.interface

.. autosummary::
    :toctree: ../generated/

    get_method
"""
import importlib

from pysteps.postprocessing import diagnostics, ensemblestats
from pprint import pprint
import warnings

_diagnostics_methods = dict()

_ensemblestats_methods = dict(
    mean=ensemblestats.mean,
    excprob=ensemblestats.excprob,
    banddepth=ensemblestats.banddepth,
)


def add_postprocessor(
    postprocessors_function_name, _postprocessors, module, attributes
):
    """
    Add the postprocessor to the appropriate _methods dictionary and to the module.
    Parameters
    ----------

    postprocessors_function_name: str
        for example, e.g. diagnostic_example1
    _postprocessors: function
        the function to be added
    @param module: the module where the function is added, e.g. 'diagnostics'
    @param attributes: the existing functions in the selected module
    """
    # the dictionary where the function is added
    methods_dict = (
        _diagnostics_methods if "diagnostic" in module else _ensemblestats_methods
    )

    # get funtion name without mo
    short_name = postprocessors_function_name.replace(f"{module}_", "")
    if short_name not in methods_dict:
        methods_dict[short_name] = _postprocessors
    else:
        warnings.warn(
            f"The {module} identifier '{short_name}' is already available in "
            f"'pysteps.postprocessing.interface_{module}_methods'.\n"
            f"Skipping {module}:{'.'.join(attributes)}",
            RuntimeWarning,
        )

    if hasattr(globals()[module], postprocessors_function_name):
        warnings.warn(
            f"The {module} function '{short_name}' is already an attribute"
            f"of 'pysteps.postprocessing.{module}'.\n"
            f"Skipping {module}:{'.'.join(attributes)}",
            RuntimeWarning,
        )
    else:
        setattr(globals()[module], postprocessors_function_name, _postprocessors)


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

    # Discover the postprocessors available in the plugins
    for plugintype in ["diagnostic", "ensemblestat"]:
        for entry_point in pkg_resources.iter_entry_points(
            group=f"pysteps.plugins.{plugintype}", name=None
        ):
            _postprocessors = entry_point.load()

            postprocessors_function_name = _postprocessors.__name__

            if plugintype in entry_point.module_name:
                add_postprocessor(
                    postprocessors_function_name,
                    _postprocessors,
                    f"{plugintype}s",
                    entry_point.attrs,
                )


def print_postprocessors_info(module_name, interface_methods, module_methods):
    """
    Helper function to print the postprocessors available in the module and in the interface.

    Parameters
    ----------
    module_name: str
        Name of the module, for example 'pysteps.postprocessing.diagnostics'.
    interface_methods: dict
        Dictionary of the postprocessors declared in the interface, for example _diagnostics_methods.
    module_methods: list
        List of the postprocessors available in the module, for example 'diagnostic_example1'.

    """
    print(f"\npostprocessors available in the {module_name} module")
    pprint(module_methods)

    print(
        "\npostprocessors available in the pysteps.postprocessing.get_method interface"
    )
    pprint([(short_name, f.__name__) for short_name, f in interface_methods.items()])

    module_methods_set = set(module_methods)
    interface_methods_set = set(interface_methods.keys())

    difference = module_methods_set ^ interface_methods_set
    if len(difference) > 0:
        # print("\nIMPORTANT:")
        _diff = module_methods_set - interface_methods_set
        if len(_diff) > 0:
            print(
                f"\nIMPORTANT:\nThe following postprocessors are available in {module_name} module but not in the pysteps.postprocessing.get_method interface"
            )
            pprint(_diff)
        _diff = interface_methods_set - module_methods_set
        if len(_diff) > 0:
            print(
                "\nWARNING:\n"
                f"The following postprocessors are available in the pysteps.postprocessing.get_method interface but not in the {module_name} module"
            )
            pprint(_diff)


def postprocessors_info():
    """Print all the available postprocessors."""

    available_postprocessors = set()
    postprocessors_in_the_interface = set()
    # List the plugins that have been added to the postprocessing.[plugintype] module
    for plugintype in ["diagnostics", "ensemblestats"]:
        # in the dictionary and found by get_methods() function
        interface_methods = (
            _diagnostics_methods
            if plugintype == "diagnostics"
            else _ensemblestats_methods
        )
        # in the pysteps.postprocessing module
        module_name = f"pysteps.postprocessing.{plugintype}"
        available_module_methods = [
            attr
            for attr in dir(importlib.import_module(module_name))
            if attr.startswith(plugintype[:-1])
        ]
        # add the pre-existing ensemblestats functions (see _ensemblestats_methods above)
        # that do not follow the convention to start with "ensemblestat_" as the plugins
        if "ensemblestats" in plugintype:
            available_module_methods += [
                em
                for em in _ensemblestats_methods.keys()
                if not em.startswith("ensemblestat_")
            ]
        print_postprocessors_info(
            module_name, interface_methods, available_module_methods
        )
        available_postprocessors = available_postprocessors.union(
            available_module_methods
        )
        postprocessors_in_the_interface = postprocessors_in_the_interface.union(
            interface_methods.keys()
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
        [nothing pre-installed]

        ensemblestats:
        pre-installed: mean, excprob, banddepth

        Additional options might exist if plugins are installed.

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
