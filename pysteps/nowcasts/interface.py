"""
pysteps.nowcasts.interface
==========================

Interface for the nowcasts module. It returns a callable function for computing
nowcasts.

The methods in the nowcasts module implement the following interface:

    ``forecast(precip, velocity, timesteps, **keywords)``

where precip is a (m,n) array with input precipitation field to be advected and
velocity is a (2,m,n) array containing the x- and y-components of
the m x n advection field.
timesteps can be an integer or a list. An integer specifies the number of time
steps to forecast, where the output time step is taken from the inputs.
Irregular time steps can be given in a list.
The interface accepts optional keyword arguments specific to the given method.

The output depends on the type of the method.
For deterministic methods, the output is a three-dimensional array of shape
(num_timesteps,m,n) containing a time series of nowcast precipitation fields.
For stochastic methods that produce an ensemble, the output is a
four-dimensional array of shape (num_ensemble_members,num_timesteps,m,n).
The time step of the output is taken from the inputs.

.. autosummary::
    :toctree: ../generated/

    get_method
"""

from pprint import pprint
from pysteps import nowcasts
import importlib
from pysteps.extrapolation.interface import eulerian_persistence
from pysteps.nowcasts import (
    anvil,
    extrapolation,
    linda,
    sprog,
    steps,
    sseps,
)

from pysteps.nowcasts import lagrangian_probability
import os


_nowcast_methods = dict()
_nowcast_methods["anvil"] = anvil.forecast
_nowcast_methods["eulerian"] = eulerian_persistence
_nowcast_methods["extrapolation"] = extrapolation.forecast
_nowcast_methods["lagrangian"] = extrapolation.forecast
_nowcast_methods["lagrangian_probability"] = lagrangian_probability.forecast
_nowcast_methods["linda"] = linda.forecast
_nowcast_methods["probability"] = lagrangian_probability.forecast
_nowcast_methods["sprog"] = sprog.forecast
_nowcast_methods["sseps"] = sseps.forecast
_nowcast_methods["steps"] = steps.forecast


def discover_nowcasts():
    """
    Search for installed nowcasts plugins in the entrypoint 'pysteps.plugin.nowcasts'

    The nowcasts method found are added to the `pysteps.nowcasts.interface_nowcasts_methods`
    dictionary containing the available nowcasts_methods.
    """

    # The pkg resources needs to be reload to detect new packages installed during
    # the execution of the python application. For example, when the plugins are
    # installed during the tests
    import pkg_resources

    importlib.reload(pkg_resources)

    for entry_point in pkg_resources.iter_entry_points(
        group="pysteps.plugins.nowcasts", name=None
    ):
        _module = entry_point.load()
        nowcast_module_name = entry_point.name

        if nowcast_module_name not in _nowcast_methods:
            _nowcast_methods[nowcast_module_name] = _module

        else:
            RuntimeWarning(
                f"The Nowcasts methode '{nowcast_module_name}' is already available in"
                "'pysteps.nowcasts._nowcasts_methods'.\n"
                f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
            )
        if hasattr(nowcasts, nowcast_module_name):
            RuntimeWarning(
                f"The nowcasts method '{nowcast_module_name}' is already an attribute"
                "of 'pysteps.nowcasts'.\n"
                f"Skipping {entry_point.module_name}:{'.'.join(entry_point.attrs)}"
            )
        else:
            setattr(nowcasts, nowcast_module_name, _module)


def nowcasts_info():
    """Print all the available nowcast methods."""

    # nowcasts methods available in the `nowcasts` package
    available_nowcasts = [
        attr.split(".")[0]
        for attr in os.listdir(" ".join(nowcasts.__path__))
        if not attr.startswith("__") and attr != "interface.py"
    ]
    print("\nMethods available in the pysteps.nowcasts")
    pprint(available_nowcasts)
    # nowcasts declared in the pysteps.nowcast interface

    nowcasts_in_the_interface = [f for f in _nowcast_methods.keys()]

    print("\nMethods available in the pysteps.nowcasts.get_method interface")
    pprint([(short_name, f.__name__) for short_name, f in _nowcast_methods.items()])

    # Let's use sets to find out if there are importers present in the importer module
    # but not declared in the interface, and viceversa.
    available_nowcasts = set(available_nowcasts)
    nowcasts_in_the_interface = set(nowcasts_in_the_interface)

    difference = available_nowcasts ^ nowcasts_in_the_interface
    if len(difference) > 0:
        print("\nIMPORTANT:")
        _diff = available_nowcasts - nowcasts_in_the_interface
        if len(_diff) > 0:
            print(
                "\nIMPORTANT:\nThe following importers are available in pysteps.nowcasts module "
                "but not in the pysteps.nowcasts.get_method interface"
            )
            pprint(_diff)
        _diff = nowcasts_in_the_interface - available_nowcasts
        if len(_diff) > 0:
            print(
                "\nWARNING:\n"
                "The following importers are available in the pysteps.nowcasts.get_method "
                "interface but not in the pysteps.nowcasts module"
            )
            pprint(_diff)

    return available_nowcasts, nowcasts_in_the_interface


def get_method(name):
    """
    Return a callable function for computing nowcasts.

    Description:
    Return a callable function for computing deterministic or ensemble
    precipitation nowcasts.

    Implemented methods:

    +-----------------+-------------------------------------------------------+
    |     Name        |              Description                              |
    +=================+=======================================================+
    |  anvil          | the autoregressive nowcasting using VIL (ANVIL)       |
    |                 | nowcasting method developed in :cite:`PCLH2020`       |
    +-----------------+-------------------------------------------------------+
    |  eulerian       | this approach keeps the last observation frozen       |
    |                 | (Eulerian persistence)                                |
    +-----------------+-------------------------------------------------------+
    |  lagrangian or  | this approach extrapolates the last observation       |
    |  extrapolation  | using the motion field (Lagrangian persistence)       |
    +-----------------+-------------------------------------------------------+
    |  linda          | the LINDA method developed in :cite:`PCN2021`         |
    +-----------------+-------------------------------------------------------+
    |  lagrangian\_   | this approach computes local lagrangian probability   |
    |  probability    | forecasts of threshold exceedences                    |
    +-----------------+-------------------------------------------------------+
    |  sprog          | the S-PROG method described in :cite:`Seed2003`       |
    +-----------------+-------------------------------------------------------+
    |  steps          | the STEPS stochastic nowcasting method described in   |
    |                 | :cite:`Seed2003`, :cite:`BPS2006` and :cite:`SPN2013` |
    |                 |                                                       |
    +-----------------+-------------------------------------------------------+
    |  sseps          | short-space ensemble prediction system (SSEPS).       |
    |                 | Essentially, this is a localization of STEPS          |
    +-----------------+-------------------------------------------------------+

    """
    if isinstance(name, str):
        name = name.lower()
    else:
        raise TypeError(
            "Only strings supported for the method's names.\n"
            + "Available names:"
            + str(list(_nowcast_methods.keys()))
        ) from None

    try:
        return _nowcast_methods[name]
    except KeyError:
        raise ValueError(
            "Unknown nowcasting method {}\n".format(name)
            + "The available methods are:"
            + str(list(_nowcast_methods.keys()))
        ) from None
