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

from pysteps import postprocessing
from pysteps.postprocessing import postprocessors
from pprint import pprint

_postprocessor_methods = dict()

def discover_postprocessors():
    """
    Search for installed postprocessors plugins in the entrypoint 'pysteps.plugins.postprocessors'

    The postprocessors found are added to the `pysteps.postprocessing.interface_postprocessor_methods`
    dictionary containing the available postprocessors.
    """

    # The pkg resources needs to be reloaded to detect new packages installed during
    # the execution of the python application. For example, when the plugins are
    # installed during the tests
    import pkg_resources

    importlib.reload(pkg_resources)

    for entry_point in pkg_resources.iter_entry_points(group='pysteps.plugins.postprocessors', name=None):
        _postprocessor = entry_point.load()

        postprocessor_function_name = _postprocessor.__name__
        postprocessor_short_name = postprocessor_function_name.replace("postprocess_", "")

        _postprocess_kws = getattr(_postprocessor, "postprocess_kws", dict())
        _postprocessor = postprocess_import(**_postprocess_kws)(_postprocessor)
        if postprocessor_short_name not in _postprocessor_methods:
            _postprocessor_methods[postprocessor_short_name] = _postprocessor
        else:
            RuntimeWarning(
                f"The postprocessor identifier '{postprocessor_short_name}' is already available in "
                "'pysteps.postprocessing.interface_postprocessor_methods'.\n"
                f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
            )

        if hasattr(postprocessors, postprocessor_function_name):
            RuntimeWarning(
                f"The postprocessor function '{postprocessor_function_name}' is already an attribute"
                "of 'pysteps.postprocessing.postprocessors'.\n"
                f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
            )
        else:
            setattr(postprocessors, postprocessor_function_name, _postprocessor)

def postprocessors_info():
    """Print all the available postprocessors."""

    # Postprocessors available in the 'postprocessing.postprocessors' module
    available_postprocessors = [
        attr for attr in dir(postprocessing.postprocessors) if attr.startswith("postprocess_")
    ]

    print("\nPostprocessors available in the pysteps.postprocessing.postprocessors module")
    pprint(available_postprocessors)

    # Postprocessors declared in the pysteps.postprocessing.get_method interface
    postprocessors_in_the_interface = [
        f.__name__ for f in postprocessing.interface._postprocessor_methods.values()
    ]

    print("\nPostprocessors available in the pysteps.postprocessing.get_method interface")
    pprint(
        [
            (short_name, f.__name__)
            for short_name, f in postprocessing.interface._postprocessor_methods.items()
        ]
    )

    # Let's use sets to find out if there are postprocessors present in the postprocessor module
    # but not declared in the interface, and viceversa.
    available_postprocessors = set(available_postprocessors)
    postprocessors_in_the_interface = set(postprocessors_in_the_interface)

    difference = available_postprocessors ^ postprocessors_in_the_interface
    if len(difference) > 0:
        print("\nIMPORTANT:")
        _diff = available_postprocessors - postprocessors_in_the_interface
        if len(_diff) > 0:
            print(
                "\nIMPORTANT:\nThe following postprocessors are available in pysteps.postprocessing.postprocessors module "
                "but not in the pysteps.postprocessing.get_method interface"
            )
            pprint(_diff)
        _diff = postprocessors_in_the_interface - available_postprocessors
        if len(_diff) > 0:
            print(
                "\nWARNING:\n"
                "The following postprocessors are available in the pysteps.postprocessing.get_method "
                "interface but not in the pysteps.postprocessing.postprocessors module"
            )
            pprint(_diff)

    return available_postprocessors, postprocessors_in_the_interface

def get_method(name, method_type):
    """
    Return a callable function for the method corresponding to the given
    name.

    Parameters
    ----------
    name: str
        Name of the method. The available options are:\n

        Postprocessors:

        .. tabularcolumns:: |p{2cm}|L|

        +-------------+-------------------------------------------------------+
        |     Name    |              Description                              |
        +=============+=======================================================+

    method_type: {'postprocessor'}
            Type of the method (see tables above).

    """

    if isinstance(method_type, str):
        method_type = method_type.lower()
    else:
        raise TypeError(
            "Only strings supported for for the method_type"
            + " argument\n"
            + "The available types are: 'postprocessor'"
        ) from None

    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "\nAvailable postprocessors names:"
            + str(list(_postprocessor_methods.keys()))
        ) from None

    if method_type == "postprocessor":
        methods_dict = _postprocessor_methods
    else:
        raise ValueError(
            "Unknown method type {}\n".format(name)
            + "The available types are: 'postprocessor'"
        ) from None

    try:
        return methods_dict[name]
    except KeyError:
        raise ValueError(
            "Unknown {} method {}\n".format(method_type, name)
            + "The available methods are:"
            + str(list(methods_dict.keys()))
        ) from None